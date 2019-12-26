import numpy as np
import pandas as pd
from iholt import Holt_model
import os
import gc

def train_model(train, test):
    # Model
    # build model
    holt_model = Holt_model(train)

    # optimize
    bnds = [[0, 1]] * 8
    x0 = np.ones(8) * 0.5   # Parameters [a11, a12, a21, a22, b11, b12, b21, b22]
    result = holt_model.train(x0, bnds)
    
    # predict
    It, Lt, Tt = holt_model.pred(result.x, 168, test)

    return result, It

if __name__ == "__main__":
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    methods = ['hierarchical/euclidean', 'hierarchical/cityblock', 'hierarchical/hausdorff', 'kmeans']
    data_sets = ['Irish_2010', 'London_2013']

    path = os.path.abspath(os.path.join(os.getcwd()))

    for times in range(1, 11):
        for data_set in data_sets:

            data = get_data(path, data_set)

            for method in methods:
                for n_clusters in range(2, 11):
                    for month in range(1, 13):
                        
                        path_cluster = os.path.join(path, 'result', data_set, 'clustering', 'interval', method, f'n_clusters_{n_clusters}.csv')
                        clusters = pd.read_csv(path_cluster, header=None)
                        
                        series = data[:, (month-1)*2:month*2, :months[month-1]*24]
                        
                        print('times:', times, ', data_set:', data_set, ', method:', method, ', n_clusters:', n_clusters, ', month:', month, ', series shape:', series.shape)

                        total_pred_series = []
                        total_xs = []
                        total_scale = []
                        for i in range(n_clusters):

                            index = list(clusters[month-1] == i)
                            sub_series = series[index]
                            sub_series = np.sum(sub_series, axis=0)

                            # split
                            test = sub_series[:, -168:]
                            train = sub_series[:, :-168]

                            # normalize
                            scale = np.zeros(2)
                            scale[0] = np.max(train)
                            scale[1] = np.min(train)
                            total_scale.append(scale)
                            train = (train - scale[1])/(scale[0] - scale[1])
                            test = (test - scale[1])/(scale[0] - scale[1])
                            
                            result, It = train_model(train, test)

                            total_pred_series.append(np.squeeze(np.array(It)).T[:, -168:])
                            total_xs.append(result.x)
                            print('cluster:', i, 'train status:', result.success)
                            del result, It
                            gc.collect()

                        total_pred_series = np.array(total_pred_series)
                        total_xs = np.array(total_xs)
                        total_scale = np.array(total_scale)
                        
                        path_result = os.path.join(path, 'result', data_set, 'forecasting', 'iholt', f'times_{times}', method)
                        if not os.path.exists(path_result):
                            os.makedirs(path_result)
                            
                        np.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}.npy'), total_pred_series)
                        np.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}_params.npy'), total_xs)
                        np.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}_scale.npy'), total_scale)

                        del series, sub_series, train, test, total_pred_series, total_xs, total_scale
