from tqdm import trange
import numpy as np
import pandas as pd
import os
import gc

from l1qr import L1QR

def qloss(y_true, y_pred, q):
    tmp1 = (q / 100 - 1) * (y_true - y_pred)
    tmp2 = q / 100 * (y_true - y_pred)
    return np.mean(np.maximum(tmp1, tmp2))

def train_model_2(trainX_, trainY_, testX_):
    
    trainY = pd.Series(np.squeeze(trainY_))
    trainX = pd.DataFrame(trainX_.T)
    
    pred = []
    for q in tqdm(np.linspace(0.01, 0.99, 99)):
        
        mdl = L1QR(y=trainY, x=trainX, alpha=q)
        mdl.fit(s_max=3)
        b0 = mdl.b0.to_numpy()
        b = mdl.b.to_numpy()

        loss_train = np.zeros(len(b0))
        for i in range(len(b0)):
            tmp = b0[i] + np.sum(b[i] * (trainX_.T), axis=1)
            loss_train[i] = qloss(trainY_.reshape(-1), tmp, q)
        b0 = b0[np.argmin(loss_train)]
        b = b[np.argmin(loss_train)]
        
        pred.append(b0 + np.sum(b * (testX_.T), axis=1))
    
    return np.array(pred)


if __name__ == "__main__":

    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    #methods = ['hierarchical/euclidean', 'hierarchical/cityblock', 'hierarchical/DTW', 'kmeans']
    methods = ['hierarchical/euclidean']
    #data_sets = ['Irish_2010', 'London_2013']
    data_sets = ['Irish_2010']

    path = os.path.abspath(os.path.join(os.getcwd()))

    for times in range(1, 11):
        for data_set in data_sets:

            for method in methods:
                for n_clusters in range(2, 11):
                    for month in range(1, 13):
                        
                        print('times:', times, ', data_set:', data_set, ', method:', method, ', n_clusters:', n_clusters, ', month:', month)
                        
                        path_result1 = os.path.join(path, 'result', data_set, 'forecasting', 'qra', 'step_1', method)
                        path_result2 = os.path.join(path, 'result', data_set, 'forecasting', 'qra', 'step_2', method)
                        if not os.path.exists(path_result2):
                            os.makedirs(path_result2)
                        
                        trainX_ = np.load(os.path.join(path_result1, f'n_clusters_{n_clusters}_month_{month}_trainX.npy'))
                        trainY_ = np.load(os.path.join(path_result1, f'n_clusters_{n_clusters}_month_{month}_trainY.npy'))
                        testX_ = np.load(os.path.join(path_result1, f'n_clusters_{n_clusters}_month_{month}_testX.npy'))

                        total_pred_series = []
                        for i in range(n_clusters):
                            
                            pred_series = train_model_2(trainX_, trainY_, testX_)
                            
                            total_pred_series.append(pred_series)
                            print('cluster:', i)

                        total_pred_series = np.array(total_pred_series)
                        np.save(os.path.join(path_result2, f'n_clusters_{n_clusters}_month_{month}.npy'), total_pred_series)

                        del total_pred_series
