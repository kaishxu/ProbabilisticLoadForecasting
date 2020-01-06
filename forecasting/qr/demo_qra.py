from tqdm import trange
import numpy as np
import pandas as pd
import os
import gc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')

import qreg
from dataloader import get_data, get_train_set_qra, get_test_set_qra, get_weather, get_hod, get_dow

def train_model_1(train, test, week, day):
    
    d = 2
    
    # to get the num of samples
    max_lag = 24
    trainX, trainTlag, trainTd, trainY = get_train_set_qra(train, week, day, max_lag, d)
    n_samples = trainY.shape[0]
    
    pred_train = np.zeros((max_lag, n_samples))
    pred_test = np.zeros((max_lag, 168))
    
    for lag in trange(1, max_lag+1):
        trainX, trainTlag, trainTd, trainY = get_train_set_qra(train, week, day, lag, d)
        testX, testTlag, testTd, testY = get_test_set_qra(train, test, week, day, lag, d)

        linreg = LinearRegression()
        model = linreg.fit(np.hstack((trainX, trainTlag, trainTd)), trainY)

        # Predict (train)
        pred = linreg.predict(np.hstack((trainX, trainTlag, trainTd)))
        pred_train[lag-1] = np.squeeze(pred[-n_samples:, :])

        # Predict (test)
        pred = linreg.predict(np.hstack((testX, testTlag, testTd)))
        pred_test[lag-1] = np.squeeze(pred)

    # clear
    del model, linreg, pred
    gc.collect()
    return pred_train, pred_test, trainY[-n_samples:, :], testY

def train_model_2(trainX, trainY, testX):
    
    tau = np.linspace(0.01, 0.99, 99)
    beta = qreg.linear(trainX.T, np.squeeze(trainY), tau)
    
    testX_ = np.hstack((np.ones((testX.shape[1], 1)), testX.T))
    
    pred = np.zeros((99, 168))
    for q in range(99):
        pred[q] = np.sum(testX_ * beta[q], axis=1)

    return pred


if __name__ == "__main__":

    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    #methods = ['hierarchical/euclidean', 'hierarchical/cityblock', 'hierarchical/DTW', 'kmeans']
    methods = ['hierarchical/euclidean']
    #data_sets = ['Irish_2010', 'London_2013']
    data_sets = ['Irish_2010']

    path = os.path.abspath(os.path.join(os.getcwd()))

    for times in range(1, 2):
        for data_set in data_sets:

            data = get_data(path, data_set)
            print(data.shape)
            for method in methods:
                for n_clusters in range(2, 11):
                    for month in range(1, 13):
                        
                        weather = get_weather(path, data_set, month)
                        week = get_dow(data_set, month)
                        day = get_hod(month)
                        
                        path_cluster = os.path.join(path, 'result', data_set, 'clustering', 'point', method, f'n_clusters_{n_clusters}.csv')
                        clusters = pd.read_csv(path_cluster, header=None)
                        
                        series = data[:, month-1, :months[month-1]*24]
                        
                        print('times:', times, ', data_set:', data_set, ', method:', method, ', n_clusters:', n_clusters, ', month:', month, ', series shape:', series.shape)

                        total_scale = []

                        path_result = os.path.join(path, 'result', data_set, 'forecasting', 'qra', f'times_{times}', method)
                        if not os.path.exists(path_result):
                            os.makedirs(path_result)

                        total_pred_series = []
                        for i in range(n_clusters):

                            index = list(clusters[month-1] == i)
                            sub_series = series[index]
                            sub_series = np.sum(sub_series, axis=0)
                            
                            total_series = np.vstack((sub_series, weather))
                            
                            test = total_series[:, -168:]
                            train = total_series[:, :-168]
                            
                            scale = np.zeros(2)
                            scale[0] = np.max(train[0])
                            scale[1] = np.min(train[0])
                            total_scale.append(scale)
                            
                            train[0] = (train[0] - scale[1]) / (scale[0] - scale[1])
                            test[0] = (test[0] - scale[1]) / (scale[0] - scale[1])
                            
                            trainX_, testX_, trainY_, testY_ = train_model_1(train, test, week, day)
                            pred_series = train_model_2(trainX_, trainY_, testX_)
                            
                            print('cluster:', i)
                            
                            total_pred_series.append(pred_series)
                            del sub_series, train, test
                            gc.collect()

                        total_pred_series = np.array(total_pred_series)
                        np.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}.npy'), total_pred_series)

                        del series, total_scale
