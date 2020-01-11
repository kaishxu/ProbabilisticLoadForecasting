import numpy as np
import pandas as pd
import os
from scipy import stats
import utils
import torch.optim as optim
import torch
from torch.utils.data.sampler import RandomSampler

import model.net as net
from dataloader import *
from train import train_and_evaluate

months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

data_set = 'Irish_2010'
path = os.path.abspath(os.path.join(os.getcwd()))

data = get_data(path, data_set)

method = 'hierarchical/euclidean'

for times in range(1, 11):

    for month in range(1, 13):
        for n_clusters in range(2, 11):

            print('times:', times, ', data_set:', data_set, ', method:', method, ', n_clusters:', n_clusters, ', month:', month)

            path_cluster = os.path.join(path, 'result', data_set, 'clustering', 'point', method, f'n_clusters_{n_clusters}.csv')
            clusters = pd.read_csv(path_cluster, header=None)
            path_data = os.path.join(path, 'data', 'deepar')

            series = data[:, month-1, :months[month-1]*24].T.copy()

            total_time = series.shape[0]
            num_series = series.shape[1]

            weather = get_weather(path, data_set, month)
            week = get_dow(data_set, month)
            day = get_hod(month)

            num_covariates = 4
            covariates = np.zeros((num_covariates, len(series)))
            covariates[1] = stats.zscore(weather)
            covariates[2] = stats.zscore(week)
            covariates[3] = stats.zscore(day)
            cov_age = stats.zscore(np.arange(total_time))
            covariates[0] = cov_age
            covariates = covariates.T.copy()

            train_data = series[:-9*24, :].copy()
            test_data = series[-7*24-168:, :].copy()
            val_data = series[-9*24-168:-7*24, :].copy()

            window_size = 192
            stride_size = 24

            # prepare data
            cov = covariates[:-9*24, :].copy()
            train_x_input, train_v_input, train_label = prep_data(train_data, cov, window_size, stride_size, num_covariates, num_series, clusters[month-1])
            cov = covariates[-7*24-168:, :].copy()
            test_x_input, test_v_input, test_label = prep_data(test_data, cov, window_size, stride_size, num_covariates, num_series, clusters[month-1], train=False)
            cov = covariates[-9*24-168:-7*24, :].copy()
            val_x_input, val_v_input, val_label = prep_data(val_data, cov, window_size, stride_size, num_covariates, num_series, clusters[month-1], train=False)

            # params
            json_path = os.path.join(path, 'forecasting', 'deepar', 'params24.json')
            params = utils.Params(json_path)

            params.num_class = n_clusters
            params.relative_metrics = False
            params.sampling = False
            params.one_step = True

            model_dir = os.path.join(path, 'result', data_set, 'forecasting', 'deepar', f'times_{times}', method)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            params.model_dir = os.path.join(model_dir, f'n_clusters_{n_clusters}_month_{month}.pth.tar')

            # use GPU if available
            cuda_exist = torch.cuda.is_available()

            # Set random seeds for reproducible experiments if necessary
            if cuda_exist:
                params.device = torch.device('cuda')
                # torch.cuda.manual_seed(240)
                model = net.Net(params).cuda()
            else:
                params.device = torch.device('cpu')
                # torch.manual_seed(230)
                model = net.Net(params)

            # dataset
            train_set = TrainDataset(train_x_input, train_label)
            test_set = TestDataset(test_x_input, test_v_input, test_label)
            val_set = TestDataset(val_x_input, val_v_input, val_label)

            # sampler
            train_sampler = WeightedSampler(train_v_input) # Use weighted sampler instead of random sampler

            # loader
            train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=train_sampler, num_workers=4)
            test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
            val_loader = DataLoader(val_set, batch_size=params.predict_batch, sampler=RandomSampler(val_set), num_workers=4)

            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            loss_fn = net.loss_fn

            restore_file = None
            train_and_evaluate(model,
                            train_loader,
                            test_loader,
                            val_loader,
                            optimizer,
                            loss_fn,
                            params,
                            restore_file)
            break
        break
