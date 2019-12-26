import numpy as np
import pandas as pd
import os
from tqdm import trange

def get_train_set(data, lag, d):
    l = np.maximum(d * 24, lag)

    total_X = []
    total_Y = []
    for i in range(len(data[0]) - l):

        X = np.zeros((2, d + lag))
        X[:, :lag] = data[:, i+l-lag:i+l]

        for j in range(d):
            X[:, lag+j] = np.mean(data[:, i+l-(j+1)*24:i+l-j*24], axis=1)

        Y = data[:, i+l]
        total_X.append(X)
        total_Y.append(Y)
    
    total_X = np.array(total_X)
    total_Y = np.array(total_Y)
    
    X_c = (total_X[:, 1, :] + total_X[:, 0, :]) / 2
    X_r = (total_X[:, 1, :] - total_X[:, 0, :]) / 2
    Y_c = ((total_Y[:, 1] + total_Y[:, 0]) / 2).reshape(-1, 1)
    Y_r = ((total_Y[:, 1] - total_Y[:, 0]) / 2).reshape(-1, 1)
    
    return X_c, X_r, Y_c, Y_r

def get_test_set(data, test, lag, d):
    l = np.maximum(d * 24, lag)
    
    data = np.hstack((data[:, -l:], test))
    
    total_X = []
    total_Y = []
    for i in range(len(data[0]) - l):

        X = np.zeros((2, d + lag))
        X[:, :lag] = data[:, i+l-lag:i+l]

        for j in range(d):
            X[:, lag+j] = np.mean(data[:, i+l-(j+1)*24:i+l-j*24], axis=1)

        Y = data[:, i+l]
        total_X.append(X)
        total_Y.append(Y)
    
    total_X = np.array(total_X)
    total_Y = np.array(total_Y)
    
    X_c = (total_X[:, 1, :] + total_X[:, 0, :]) / 2
    X_r = (total_X[:, 1, :] - total_X[:, 0, :]) / 2
    Y_c = ((total_Y[:, 1] + total_Y[:, 0]) / 2).reshape(-1, 1)
    Y_r = ((total_Y[:, 1] - total_Y[:, 0]) / 2).reshape(-1, 1)
    
    return X_c, X_r, Y_c, Y_r


def get_train_set_(data, lag, d):
    l = np.maximum(d * 24, lag)

    total_X = []
    total_Y = []
    for i in range(len(data[0]) - l):

        X = np.zeros((2, d + lag))
        X[:, :lag] = data[:, i+l-lag:i+l]

        for j in range(d):
            X[:, lag+j] = np.mean(data[:, i+l-(j+1)*24:i+l-j*24], axis=1)

        Y = data[:, i+l]
        total_X.append(X)
        total_Y.append(Y)
    
    total_X = np.array(total_X)
    total_Y = np.array(total_Y)
    
    return total_X.reshape(total_X.shape[0], -1), total_Y.reshape(total_Y.shape[0], -1)

def get_test_set_(data, test, lag, d):
    l = np.maximum(d * 24, lag)
    
    data = np.hstack((data[:, -l:], test))
    
    total_X = []
    total_Y = []
    for i in range(len(data[0]) - l):

        X = np.zeros((2, d + lag))
        X[:, :lag] = data[:, i+l-lag:i+l]

        for j in range(d):
            X[:, lag+j] = np.mean(data[:, i+l-(j+1)*24:i+l-j*24], axis=1)

        Y = data[:, i+l]
        total_X.append(X)
        total_Y.append(Y)
    
    total_X = np.array(total_X)
    total_Y = np.array(total_Y)
    
    return total_X.reshape(total_X.shape[0], -1), total_Y.reshape(total_Y.shape[0], -1)

def get_data(path, data_set):

    attr = pd.read_csv(os.path.join(path, 'data', f'{data_set}_attr_final.csv'))
    data = []
    for i in trange(len(attr)):
        id = attr['ID'][i]
        df = pd.read_csv(os.path.join(path, 'data', f'{data_set}_monthly_interval', f'{id}.csv'), header = None).values
        data.append(df)
    data = np.array(data)
    return data