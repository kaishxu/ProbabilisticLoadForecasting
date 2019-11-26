import pandas as pd
import numpy as np
from dtw import accelerated_dtw
import multiprocessing
import scipy.spatial.distance as dm

def euclidean(a, b, alpha, beta):
    tmp = (a - alpha) ** 2 + (b - beta) ** 2
    tmp = tmp.sum(axis=1)
    return tmp

def cityblock(a, b, alpha, beta):
    tmp = abs(a - alpha) + abs(b - beta)
    tmp = tmp.sum(axis=1)
    return tmp

data_set = 'London_2013'
dist = 'hierarchical/DTW'
attr = pd.read_csv('./data/' + data_set + '_attr_final.csv')

# Hierarchical clustering (Construct the matrix)
labels = []
for month in range(12):

    X = []
    for i in range(len(attr)):
        id = attr['ID'][i]
        df = pd.read_csv('./data/' + data_set + '_profiles/' + str(id) + '.csv', header = None).values
        X.append(df[month])
    X = np.array(X)
    
    print('Month: ' + str(month+1) + ', Finish constructing X!')
    
    mat = [(X[i], X[j]) for i in range(len(attr)) for j in range(i+1, len(attr))]

    print('Month: ' + str(month+1) + ', Finish constructing mat(1)!')
    
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    mat = pool.starmap(fastdtw, mat)
    mat = dm.squareform(mat)

    print('Month: ' + str(month+1) + ', Finish constructing mat(2)!')
    
    pd.DataFrame(mat).to_csv('./result/' + data_set + '/cluster/point/' + dist + '/mat_month_' + str(month+1) + '.csv', header=None, index=False)