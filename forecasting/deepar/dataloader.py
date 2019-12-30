from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import os
import re
from tqdm import trange
from datetime import datetime
from scipy import stats
from torch.utils.data import DataLoader, Dataset, Sampler
import torch

class TrainDataset(Dataset):
    def __init__(self, data, label):
        self.data = data.copy()
        self.label = label.copy()
        self.train_len = self.data.shape[0]

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        return (self.data[index,:,:-1],int(self.data[index,0,-1]), self.label[index])

class TestDataset(Dataset):
    def __init__(self, data, v, label):
        self.data = data.copy()
        self.v = v.copy()
        self.label = label.copy()
        self.test_len = self.data.shape[0]

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        return (self.data[index,:,:-1],int(self.data[index,0,-1]),self.v[index],self.label[index])

class WeightedSampler(Sampler):
    def __init__(self, v, replacement=True):
        v = v.copy()
        self.weights = torch.as_tensor(np.abs(v[:,0]) / np.sum(np.abs(v[:,0])), dtype=torch.double)
        self.num_samples = self.weights.shape[0]
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples

months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# calendar series (day of week)
def get_dow(data_set, month):
    
    year = re.findall(r"\d+", data_set)[0]
    week = []
    for i in range(months[month-1]):
        week_value = datetime.strptime(year + '%02d%02d' %(month, i+1), '%Y%m%d').weekday()
        for _ in range(24):
            week.append(week_value)
    week = np.array(week)   # normalization
    return week

# calendar series (hour of day)
def get_hod(month):
    
    day = []
    for i in range(months[month-1]):
        for _ in range(24):
            day.append(_)
    day = np.array(day)   # normalization
    return day

# weather series
def get_weather(path, data_set, month):
    
    weather_48 = pd.read_csv(os.path.join(path, 'data', 'weather', f'weather_{data_set}_cleaned.csv'))['T'].values
    weather_24 = []
    for i in range(int(len(weather_48)/2)):
        weather_24.append((weather_48[i*2] + weather_48[i*2+1])/2)
    weather_24 = np.array(weather_24)

    weather = []
    for i in range(12):
        weather.append(weather_24[sum(months[:i])*24:sum(months[:i+1])*24])
    weather = weather[month-1]
    
    return weather

def get_data(path, data_set):

    attr = pd.read_csv(os.path.join(path, 'data', f'{data_set}_attr_final.csv'))
    data = []
    for i in trange(len(attr)):
        id = attr['ID'][i]
        df = pd.read_csv(os.path.join(path, 'data', f'{data_set}_monthly', f'{id}.csv'), header = None).values
        data.append(df)
    data = np.array(data)
    return data

def prep_data(data, covariates, data_start, window_size, stride_size, num_covariates, num_series, total_time, train = True):

    time_len = data.shape[0]
    input_size = window_size - stride_size

    windows_per_series = np.full((num_series), (time_len - input_size) // stride_size)
    if train: windows_per_series -= (data_start + stride_size-1) // stride_size
    total_windows = np.sum(windows_per_series)

    x_input = np.zeros((total_windows, window_size, 1 + num_covariates + 1), dtype='float32')
    label = np.zeros((total_windows, window_size), dtype='float32')
    v_input = np.zeros((total_windows, 2), dtype='float32')

    count = 0
    if not train:
        covariates = covariates[-time_len:]
    for series in range(num_series):  #穷举series

        cov_age = stats.zscore(np.arange(total_time - data_start[series]))  # 位置变量,序列中的第几个
        if train:
            covariates[data_start[series]:time_len, 0] = cov_age[:time_len-data_start[series]]  # 位置变量,序列中的第几个(矫正当前序列)
        else:
            covariates[:, 0] = cov_age[-time_len:]  # for test set

        for i in range(windows_per_series[series]):  # 穷举window
            if train:
                window_start = stride_size * i + data_start[series]
            else:
                window_start = stride_size * i  # for test set
            window_end = window_start + window_size  # 定位窗口起始

            x_input[count, 1:, 0] = data[window_start:window_end-1, series]
            x_input[count, :, 1:1+num_covariates] = covariates[window_start:window_end, :]  # 相关变量
            x_input[count, :, -1] = series  # 序列标签
            label[count, :] = data[window_start:window_end, series]  # 输出
            nonzero_sum = (x_input[count, 1:input_size, 0]!=0).sum()  # 非零个数
            if nonzero_sum == 0:
                v_input[count, 0] = 0
            else:
                v_input[count, 0] = np.true_divide(x_input[count, 1:input_size, 0].sum(),nonzero_sum) + 1   # 用非零平均(可以直接平均)
                x_input[count, :, 0] = x_input[count, :, 0] / v_input[count, 0]  # scale
                if train:
                    label[count, :] = label[count, :] / v_input[count, 0]
            count += 1

    return x_input, v_input, label
