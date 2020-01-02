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

#gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

def qloss(y_true, y_pred, q):
    tmp1 = (q / 100 - 1) * (y_true - y_pred)
    tmp2 = q / 100 * (y_true - y_pred)
    return K.mean(K.maximum(tmp1, tmp2))

def train_model_2(trainX, trainY, testX, num_best):
    
    total_pred = []
    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    
    for q in trange(1, 100):
        
        input_dim = num_best
        model = Sequential([Dense(1, use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal', input_shape=(input_dim,))])

        # Train
        model.compile(loss=lambda y_true, y_pred: qloss(y_true, y_pred, q), optimizer='adam')
        hist2 = model.fit(x=trainX, y=trainY, validation_split=0.2, epochs=1500, verbose=0, callbacks=[early_stopping])

        # Predict (test)
        pred = model.predict(x=testX)
        total_pred.append(np.squeeze(pred))
    
    total_pred = np.array(total_pred)
    
    del model, pred, hist2
    tf.keras.backend.clear_session()
    gc.collect()
    return total_pred


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
                        
                        path_result1 = os.path.join(path, 'result', data_set, 'forecasting', 'qra', 'step_1', f'times_{times}', method)
                        path_result2 = os.path.join(path, 'result', data_set, 'forecasting', 'qra', 'step_2', f'times_{times}', method)
                        if not os.path.exists(path_result2):
                            os.makedirs(path_result2)
                        
                        trainX_ = np.load(os.path.join(path_result1, f'n_clusters_{n_clusters}_month_{month}_trainX.npy'))
                        trainY_ = np.load(os.path.join(path_result1, f'n_clusters_{n_clusters}_month_{month}_trainY.npy'))
                        testX_ = np.load(os.path.join(path_result1, f'n_clusters_{n_clusters}_month_{month}_testX.npy'))
                        
                        num_best = 8

                        total_pred_series = []
                        for i in range(n_clusters):
                            
                            error = np.sum(np.abs(trainX_[i] - np.squeeze(trainY_[i])), axis=2)
                            trainX = np.vstack((trainX_[i, np.argsort(error[:, 0])[:num_best//2], 0, :], trainX_[i, np.argsort(error[:, 1])[:num_best//2], 1, :])).T
                            trainY = trainY_[i].copy()
                            testX = np.vstack((testX_[i, np.argsort(error[:, 0])[:num_best//2], 0, :], testX_[i, np.argsort(error[:, 1])[:num_best//2], 1, :])).T

                            pred_series = train_model_2(trainX, trainY, testX, num_best)
                            
                            total_pred_series.append(pred_series)
                            print('cluster:', i)

                        total_pred_series = np.array(total_pred_series)
                        np.save(os.path.join(path_result2, f'n_clusters_{n_clusters}_month_{month}.npy'), total_pred_series)

                        del total_pred_series
