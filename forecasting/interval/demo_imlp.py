import os
import numpy as np
import pandas as pd
from tqdm import trange

from dataloader import get_train_set, get_test_set
from imlp import iAct, iLoss, get_model

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
                
                print('data_set:', data_set, ', method:', method, ', n_clusters:', n_clusters, ', month:', month, ', series shape:', series.shape)
                print('Start!')

                total_pred_series = []
                total_scale = []

                path_result = os.path.join(path, 'result', data_set, 'forecasting', 'imlp', method)
                if not os.path.exists(path_result):
                    os.makedirs(path_result)

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
                    
                    # recency effect
                    lag = 12
                    d = 1
                    
                    trainX_c, trainX_r, trainY_c, trainY_r = get_train_set(train, lag, d)
                    testX_c, testX_r, testY_c, testY_r = get_test_set(train, test, lag, d)
                    
                    # Parameters
                    input_dim = lag + d
                    output_dim = 1
                    num_hidden_layers = 1
                    num_units = [6]
                    act = ['tanh']
                    beta = 0.5

                    # Get model
                    model = get_model(input_dim, output_dim, num_units, act, beta, num_hidden_layers)

                    # Train
                    model.fit(x=[trainX_c, trainX_r], y=[trainY_c, trainY_r], epochs=500, verbose=0)
                    
                    pred_c, pred_r = model.predict(x=[testX_c, testX_r])

                    model.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}_for_{i}.h5'))
                    pred_series = np.vstack((np.squeeze((pred_c - pred_r) / 2), np.squeeze((pred_c + pred_r) / 2)))
                    total_pred_series.append(pred_series)

                total_pred_series = np.array(total_pred_series)
                total_scale = np.array(total_scale)

                np.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}.npy'), total_pred_series)
                np.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}_scale.npy'), total_scale)

                del series, sub_series, train, test, total_pred_series, total_scale
                print('Finish!')