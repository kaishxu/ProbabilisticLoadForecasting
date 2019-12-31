import numpy as np
import pandas as pd
import os
from keras.callbacks import EarlyStopping
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from qrnn import get_model, qloss
from dataloader import get_data, get_train_set_qrnn, get_test_set_qrnn, get_weather, get_hod, get_dow
import tensorflow as tf

#gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

def train_model(lag, d, trainX, trainY, testX, path_result, n_clusters, month, t):
    
    # Parameters
    input_dim = (lag + d) * 2 + 1 + 7 + 24
    num_hidden_layers = 2
    num_unit = 10
    num_units = [num_unit, num_unit]
    act = ['relu', 'relu']

    # Get model
    model = get_model(input_dim, num_units, act, num_hidden_layers)

    # Train
    hist = model.fit(x=trainX, y=trainY, epochs=300, verbose=0)
    pred = model.predict(x=testX)

    model.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}_for_{t}.h5'))

    return pred


if __name__ == "__main__":

    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    methods = ['hierarchical/euclidean', 'hierarchical/cityblock', 'hierarchical/DTW', 'kmeans']
    data_sets = ['Irish_2010', 'London_2013']

    path = os.path.abspath(os.path.join(os.getcwd()))
    
    for times in range(1, 11):
        for data_set in data_sets:

            data = get_data(path, data_set)

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

                        total_pred_series = []
                        total_scale = []

                        path_result = os.path.join(path, 'result', data_set, 'forecasting', 'qrnn', f'times_{times}', method)
                        if not os.path.exists(path_result):
                            os.makedirs(path_result)

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
                            
                            # recency effect
                            lag = 24
                            d = 1
                            
                            trainX, trainY = get_train_set_qrnn(train, week, day, lag, d)
                            testX, testY = get_test_set_qrnn(train, test, week, day, lag, d)
                            
                            pred_series = train_model(lag, d, trainX, trainY, testX, path_result, n_clusters, month, i)
                            
                            total_pred_series.append(pred_series)
                            print('cluster:', i)
                            
                            tf.keras.backend.clear_session()
                            del sub_series, train, test, trainX, trainY, testX, testY
                            gc.collect()

                        total_pred_series = np.array(total_pred_series)
                        total_scale = np.array(total_scale)

                        np.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}.npy'), total_pred_series)
                        np.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}_scale.npy'), total_scale)

                        del series, total_pred_series, total_scale
