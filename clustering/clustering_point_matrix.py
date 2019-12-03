# Hierarchical clustering (point)
# Construct the similarity matrix for DTW distance

import pandas as pd
import numpy as np
from dtw import accelerated_dtw
import multiprocessing
import scipy.spatial.distance as dm

def fastdtw(x, y):
    euclidean = lambda x, y: (x - y) ** 2
    dist, cost_matrix, acc_cost_matrix, path = accelerated_dtw(x, y, euclidean)
    return dist

def main(data_set, attr, dist):
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

if __name__ == '__main__':

    # data_set: Irish_2010, London_2013
    data_set = 'Irish_2010'
    dist = 'hierarchical/DTW'
    attr = pd.read_csv('./data/' + data_set + '_attr_final.csv')
    main(data_set, attr, dist)