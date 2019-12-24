import numpy as np
import pandas as pd
from iholt import Holt_model
import os
from tqdm import trange
import json

months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
methods = ['hierarchical/euclidean', 'hierarchical/cityblock', 'hierarchical/hausdorff', 'kmeans']
data_sets = ['Irish_2010', 'London_2013']

path = os.path.abspath(os.path.join(os.getcwd()))

for data_set in data_sets:

    attr = pd.read_csv(os.path.join(path, 'data', f'{data_set}_attr_final.csv'))
    data = []
    for i in trange(len(attr)):
        id = attr['ID'][i]
        df = pd.read_csv(os.path.join(path, 'data', f'{data_set}_monthly_interval', f'{id}.csv'), header = None).values
        data.append(df)
    data = np.array(data)

    for method in methods:
        for n_clusters in range(2, 11):
            for month in range(1, 13):

                path_cluster = os.path.join(path, 'result', data_set, 'clustering', 'interval', method, f'n_clusters_{n_clusters}.csv')
                clusters = pd.read_csv(path_cluster, header=None)
                series = data[:, (month-1)*2:month*2, :months[month-1]*24]

                total_pred_series = []
                total_xs = []
                total_scale = []
                for i in range(n_clusters):
                    index = list(clusters[month-1] == i)
                    sub_series = series[index]
                    sub_series = np.sum(sub_series, axis=0)
                    test = sub_series[:, -168:]
                    train = sub_series[:, :-168]

                    scale = np.zeros(2)
                    scale[0] = np.max(train)
                    scale[1] = np.min(train)
                    total_scale.append(scale)
                    train = (train - scale[1])/(scale[0] - scale[1])
                    test = (test - scale[1])/(scale[0] - scale[1])
                    
                    # test window (h = 1, 2, ..., 7)
                    pred_series = []
                    xs = []
                    for h in range(1, 8):

                        # Build model
                        holt_model = Holt_model(np.hstack((train, test[:, :(h-1)*24])))

                        # Optimize
                        bnds = [[0, 1]] * 8
                        x0 = np.ones(8) * 0.5   # Parameters [a11, a12, a21, a22, b11, b12, b21, b22]
                        result = holt_model.train(x0, bnds)
                        It, Lt, Tt = holt_model.pred(result.x, 24, test[:, (h-1)*24:h*24])
                        pred_series.append(np.squeeze(np.array(It)).T[:, -24:])
                        xs.append(result.x)
                    
                    pred_series = np.array(pred_series)
                    xs = np.array(xs)
                    total_pred_series.append(pred_series)
                    total_xs.append(xs)

                total_pred_series = np.array(total_pred_series)
                total_xs = np.array(total_xs)
                total_scale = np.array(total_scale)
                path_result = os.path.join(path, 'result', data_set, 'forecasting', 'interval', method)
                np.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}.npy'), total_pred_series)
                np.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}_params.npy'), total_xs)
                np.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}_scale.npy'), total_scale)

                print('data_set:', data_set, ', method:', method, ', n_clusters:', n_clusters, ', month:', month, ', series shape:', series.shape)
