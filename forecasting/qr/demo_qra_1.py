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

def train_model_1(train, test, week, day):
    
    # to get the num of samples
    max_lag = 24
    max_d = 2
    trainX, trainTlag, trainTd, trainY = get_train_set_qra(train, week, day, max_lag, max_d)
    n_samples = trainY.shape[0]
    
    pred_train = np.zeros((24, 2, n_samples))
    pred_test = np.zeros((24, 2, 168))

    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    
    for lag in trange(1, 25):
        for d in range(1, 3):
            
            trainX, trainTlag, trainTd, trainY = get_train_set_qra(train, week, day, lag, d)
            testX, testTlag, testTd, testY = get_test_set_qra(train, test, week, day, lag, d)

            ## QRA step 1
            # linear model
            inputs = Input((7 + 24 + 3 + lag*3 + d*3,), name='input')
            x = Dense(1, use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal')(inputs)
            model = Model(inputs=inputs, outputs=x)

            # Train
            model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
            hist1 = model.fit(x=np.hstack((trainX, trainTlag, trainTd)), y=trainY, validation_split=0.2, epochs=1500, verbose=0, callbacks=[early_stopping])

            # Predict (train)
            pred = model.predict(x=np.hstack((trainX, trainTlag, trainTd)))
            pred_train[lag-1, d-1] = np.squeeze(pred[-n_samples:, :])
            
            # Predict (test)
            pred = model.predict(x=np.hstack((testX, testTlag, testTd)))
            pred_test[lag-1, d-1] = np.squeeze(pred)
    
    # clear
    del model, pred, hist1
    tf.keras.backend.clear_session()
    gc.collect()
    return pred_train, pred_test, trainY[-n_samples:, :], testY


if __name__ == "__main__":

    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    #methods = ['hierarchical/euclidean', 'hierarchical/cityblock', 'hierarchical/DTW', 'kmeans']
    methods = ['kmeans']
    #data_sets = ['Irish_2010', 'London_2013']
    data_sets = ['London_2013']

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

                        total_trainX_ = []
                        total_trainY_ = []
                        total_testX_ = []
                        total_testY_ = []

                        total_scale = []

                        path_result = os.path.join(path, 'result', data_set, 'forecasting', 'qra', 'step_1', f'times_{times}', method)
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
                            
                            trainX_, testX_, trainY_, testY_ = train_model_1(train, test, week, day)
                            
                            total_trainX_.append(trainX_)
                            total_trainY_.append(trainY_)
                            total_testX_.append(testX_)
                            total_testY_.append(testY_)
                            print('cluster:', i)
                            
                            del sub_series, train, test
                            gc.collect()

                        total_trainX_ = np.array(total_trainX_)
                        total_trainY_ = np.array(total_trainY_)
                        total_testX_ = np.array(total_testX_)
                        total_testY_ = np.array(total_testY_)
                        total_scale = np.array(total_scale)

                        np.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}_trainX.npy'), total_trainX_)
                        np.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}_trainY.npy'), total_trainY_)
                        np.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}_testX.npy'), total_testX_)
                        np.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}_testY.npy'), total_testY_)
                        np.save(os.path.join(path_result, f'n_clusters_{n_clusters}_month_{month}_scale.npy'), total_scale)

                        del series, total_scale, total_trainX_, total_trainY_, total_testX_, total_testY_
