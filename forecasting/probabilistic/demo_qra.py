from tqdm import trange
import numpy as np
import pandas as pd
import os
import gc

from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential
from keras import backend as K
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from dataloader import get_data, get_train_set_qra, get_test_set_qra, get_weather, get_hod, get_dow

#gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

def qloss(y_true, y_pred, q):
    tmp1 = (q / 100 - 1) * (y_true - y_pred)
    tmp2 = q / 100 * (y_true - y_pred)
    return K.mean(K.maximum(tmp1, tmp2))

def train_model(train, test):
    
    # to get the length of samples
    max_lag = 24
    max_d = 2
    trainX, trainTlag, trainTd, trainY = get_train_set_qra(train, max_lag, max_d)
    l = trainY.shape[0]
    
    num_best = 8
    error_train_step1 = np.zeros((24, 2))
    pred_train = np.zeros((24, 2, l))
    pred_test = np.zeros((24, 2, 168))
    total_pred = []

    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    for lag in trange(1, 25):
        for d in range(1, 3):
            
            trainX, trainTlag, trainTd, trainY = get_train_set_qra(train, lag, d)
            testX, testTlag, testTd, testY = get_test_set_qra(train, test, lag, d)

            ## QRA step 1
            # linear model
            inputs = Input((5 + lag*3 + d*3,), name='input')
            x = Dense(1, activation=None, use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal')(inputs)
            model = Model(inputs=inputs, outputs=x)

            # Train
            model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
            hist1 = model.fit(x=np.hstack((trainX, trainTlag, trainTd)), y=trainY, validation_split=0.2, epochs=1000, verbose=0, callbacks=[early_stopping])

            # Predict (train)
            pred = model.predict(x=np.hstack((trainX, trainTlag, trainTd))[-l:, :])
            error_train_step1[lag-1, d-1] = np.sum(np.abs(pred - trainY[-l:, :]))
            pred_train[lag-1, d-1] = np.squeeze(pred)
            
            # Predict (test)
            pred = model.predict(x=np.hstack((testX, testTlag, testTd)))
            pred_test[lag-1, d-1] = np.squeeze(pred)
    
    # prepare for step 2
    series_train_1 = pred_train[np.argsort(error_train_step1[:,0])[:num_best//2], 0]
    series_train_2 = pred_train[np.argsort(error_train_step1[:,1])[:num_best//2], 1]

    trainX_ = np.vstack((series_train_1, series_train_2)).T
    trainY_ = trainY[-l:, :]
    
    series_test_1 = pred_test[np.argsort(error_train_step1[:,0])[:num_best//2], 0]
    series_test_2 = pred_test[np.argsort(error_train_step1[:,1])[:num_best//2], 1]
    
    testX_ = np.vstack((series_test_1, series_test_2)).T
    testY_ = testY
    
    # clear
    tf.keras.backend.clear_session()
    
    ## QRA step 2
    # qr model
    for q in range(1, 100):
        
        input_dim = num_best
        model = Sequential([Dense(1, input_shape=(input_dim,))])

        # Train
        model.compile(loss=lambda y_true, y_pred: qloss(y_true, y_pred, q), optimizer='adam')
        hist2 = model.fit(x=trainX_, y=trainY_, validation_split=0.2, epochs=1000, verbose=0, callbacks=[early_stopping])

        # Predict (train)
        pred = model.predict(x=trainX_)
        error_train_step2 = qloss(trainY_, pred, q)

        # Predict (test)
        pred = model.predict(x=testX_)
        error_test_step2 = qloss(testY_, pred, q)
        total_pred.append(np.squeeze(pred))
    
    total_pred = np.array(total_pred)
    tf.keras.backend.clear_session()

    return total_pred


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

                        path_result = os.path.join(path, 'result', data_set, 'forecasting', 'qra', f'times_{times}', method)
                        if not os.path.exists(path_result):
                            os.makedirs(path_result)

                        for i in range(n_clusters):

                            index = list(clusters[month-1] == i)
                            sub_series = series[index]
                            sub_series = np.sum(sub_series, axis=0)
                            
                            total_series = np.vstack((sub_series, weather, week, day))
                            
                            test = total_series[:, -168:]
                            train = total_series[:, :-168]
                            
                            scale = np.zeros(2)
                            scale[0] = np.max(train[0])
                            scale[1] = np.min(train[0])
                            total_scale.append(scale)
                            train[0] = (train[0] - scale[1]) / (scale[0] - scale[1])
                            test[0] = (test[0] - scale[1]) / (scale[0] - scale[1])
                            
                            pred_series = train_model(train, test)
                            
                            total_pred_series.append(pred_series)
                            print('cluster:', i)
                            
                            tf.keras.backend.clear_session()
                            del sub_series, train, test
                            gc.collect()

                        total_pred_series = np.array(total_pred_series)
                        total_scale = np.array(total_scale)

                        np.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}.npy'), total_pred_series)
                        np.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}_scale.npy'), total_scale)

                        del series, total_pred_series, total_scale
