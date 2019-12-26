import numpy as np
import pandas as pd
import os
from tqdm import trange

def get_data(path, data_set):

    attr = pd.read_csv(os.path.join(path, 'data', f'{data_set}_attr_final.csv'))
    data = []
    for i in trange(len(attr)):
        id = attr['ID'][i]
        df = pd.read_csv(os.path.join(path, 'data', f'{data_set}_monthly_interval', f'{id}.csv'), header = None).values
        data.append(df)
    data = np.array(data)
    return data