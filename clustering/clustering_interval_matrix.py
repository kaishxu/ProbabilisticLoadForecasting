import pandas as pd
import numpy as np
import multiprocessing
import scipy.spatial.distance as dm
from hausdorff import hausdorff_distance

def distance(x, y, dist):
    if 'euclidean' in dist:
        tmp = np.sum((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
    if 'cityblock' in dist:
        tmp = np.sum(abs(x[0] - y[0]) + abs(x[1] - y[1]))
    if 'hausdorff' in dist:
        tmp = hausdorff_distance(x.T, y.T)
    return tmp

data_set = 'Irish_2010'

#1
dist = 'hierarchical/euclidean'
attr = pd.read_csv('./data/' + data_set + '_attr_final.csv')

# Hierarchical clustering (Construct the matrix)
for month in range(12):

    X = []
    for i in range(len(attr)):
        id = attr['ID'][i]
        df = pd.read_csv('./data/' + data_set + '_profiles_interval/' + str(id) + '.csv', header = None).values
        X.append(df[month*2:(month+1)*2])
    X = np.array(X)
    
    if 'hausdorff' not in dist:
        X = (X - np.min(X))/(np.max(X) - np.min(X))

    print('Month: ' + str(month+1) + ', Finish constructing X!')
    
    mat = [(X[i], X[j], dist) for i in range(len(attr)) for j in range(i+1, len(attr))]

    print('Month: ' + str(month+1) + ', Finish constructing mat(1)!')
    
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    mat = pool.starmap(distance, mat)
    mat = dm.squareform(mat)

    print('Month: ' + str(month+1) + ', Finish constructing mat(2)!')
    
    pd.DataFrame(mat).to_csv('./result/' + data_set + '/cluster/interval/' + dist + '/mat_month_' + str(month+1) + '.csv', header=None, index=False)

#2
dist = 'hierarchical/cityblock'
attr = pd.read_csv('./data/' + data_set + '_attr_final.csv')

# Hierarchical clustering (Construct the matrix)
for month in range(12):

    X = []
    for i in range(len(attr)):
        id = attr['ID'][i]
        df = pd.read_csv('./data/' + data_set + '_profiles_interval/' + str(id) + '.csv', header = None).values
        X.append(df[month*2:(month+1)*2])
    X = np.array(X)
    
    if 'hausdorff' not in dist:
        X = (X - np.min(X))/(np.max(X) - np.min(X))

    print('Month: ' + str(month+1) + ', Finish constructing X!')
    
    mat = [(X[i], X[j], dist) for i in range(len(attr)) for j in range(i+1, len(attr))]

    print('Month: ' + str(month+1) + ', Finish constructing mat(1)!')
    
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    mat = pool.starmap(distance, mat)
    mat = dm.squareform(mat)

    print('Month: ' + str(month+1) + ', Finish constructing mat(2)!')
    
    pd.DataFrame(mat).to_csv('./result/' + data_set + '/cluster/interval/' + dist + '/mat_month_' + str(month+1) + '.csv', header=None, index=False)

#3
dist = 'hierarchical/hausdorff'
attr = pd.read_csv('./data/' + data_set + '_attr_final.csv')

# Hierarchical clustering (Construct the matrix)
for month in range(12):

    X = []
    for i in range(len(attr)):
        id = attr['ID'][i]
        df = pd.read_csv('./data/' + data_set + '_profiles_interval/' + str(id) + '.csv', header = None).values
        X.append(df[month*2:(month+1)*2])
    X = np.array(X)
    
    if 'hausdorff' not in dist:
        X = (X - np.min(X))/(np.max(X) - np.min(X))

    print('Month: ' + str(month+1) + ', Finish constructing X!')
    
    mat = [(X[i], X[j], dist) for i in range(len(attr)) for j in range(i+1, len(attr))]

    print('Month: ' + str(month+1) + ', Finish constructing mat(1)!')
    
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    mat = pool.starmap(distance, mat)
    mat = dm.squareform(mat)

    print('Month: ' + str(month+1) + ', Finish constructing mat(2)!')
    
    pd.DataFrame(mat).to_csv('./result/' + data_set + '/cluster/interval/' + dist + '/mat_month_' + str(month+1) + '.csv', header=None, index=False)