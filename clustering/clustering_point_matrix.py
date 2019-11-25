import pandas as pd
import numpy as np
from dtw import accelerated_dtw
import multiprocessing
import scipy.spatial.distance as dm

def fastdtw(x, y):
    euclidean = lambda x, y: (x - y) ** 2
    dist, cost_matrix, acc_cost_matrix, path = accelerated_dtw(x, y, euclidean)
    return dist

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

# import pandas as pd
# import numpy as np
# from dtw import accelerated_dtw
# import multiprocessing
# import scipy.spatial.distance as dm

# data_set = 'London_2013'
# dist = 'hierarchical/DTW'
# attr = pd.read_csv('./data/' + data_set + '_attr_final.csv')

# month = 0
# count = 1
# X = pd.read_csv('./result/London_2013/cluster/point/hierarchical/DTW/month_' + str(month+1) + '.csv', header=None).values

# def fastdtw(a, b):
#     euclidean = lambda x, y: (x - y) ** 2
#     global X
#     dist, cost_matrix, acc_cost_matrix, path = accelerated_dtw(X[a], X[b], euclidean)
#     global count
#     count +=1
#     if count % 10000==0:
#         print(count)
#     return dist

# mat = [(i, j) for i in range(len(attr)) for j in range(i+1, len(attr))]

# print('Month: ' + str(month+1) + ', Finish constructing mat(1)!')

# cores = multiprocessing.cpu_count()
# pool = multiprocessing.Pool(processes=cores)
# mat = pool.starmap(fastdtw, mat)
# mat = dm.squareform(mat)

# print('Month: ' + str(month+1) + ', Finish constructing mat(2)!')

# pd.DataFrame(mat).to_csv('./result/' + data_set + '/cluster/point/' + dist + '/mat_month_' + str(month+1) + '.csv', header=None, index=False)