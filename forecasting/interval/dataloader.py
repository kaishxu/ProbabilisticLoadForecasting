import numpy as np

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
    
    data = np.hstack((train[:, -l:], test))
    
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