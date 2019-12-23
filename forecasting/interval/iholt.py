from scipy.optimize import minimize
import numpy as np

def get_Lt(a, Xt, Lt_1, Tt_1):
    return np.dot(a, Xt) + np.dot((np.diag([1, 1]) - a), Lt_1 + Tt_1)

def get_Tt(b, Lt, Lt_1, Tt_1):
    return np.dot(b, Lt - Lt_1) + np.dot((np.diag([1, 1]) - b), Tt_1)

class Holt_model(object):
    def __init__(self, s):
        self.s = s
    def fun(self, x):
        a = x[:4].reshape((2, 2))
        b = x[4:].reshape((2, 2))
        s = self.s
        
        # initialize
        Lt_1 = s[:, 1:2]
        Tt_1 = s[:, 1:2] - s[:, 0:1]
        
        e = np.sum((s[:, 2:3] - Lt_1 - Tt_1) ** 2)
        for i in range(len(self.s[0]) - 3):
            Lt = get_Lt(a, s[:, i+2:i+3], Lt_1, Tt_1)
            Tt = get_Tt(b, Lt, Lt_1, Tt_1)
            e = e + np.sum((s[:, i+3:i+4] - Lt - Tt) ** 2)
            Lt_1 = Lt
            Tt_1 = Tt
        return e
    def pred(self, x, len_pred, test_sample=None):
        a = x[:4].reshape((2, 2))
        b = x[4:].reshape((2, 2))
        
        s = self.s
        Lt_1 = s[:, 1:2]
        Tt_1 = s[:, 1:2] - s[:, 0:1]
        
        list_It = []
        list_Lt = []
        list_Tt = []
        
        # predict (in sample)
        for i in range(len(self.s[0]) - 2):
            Lt = get_Lt(a, s[:, i+2:i+3], Lt_1, Tt_1)
            Tt = get_Tt(b, Lt, Lt_1, Tt_1)
            list_Lt.append(Lt)
            list_Tt.append(Tt)
            list_It.append(Lt + Tt)
            Lt_1 = Lt
            Tt_1 = Tt
        
        # predict (out of sample)
        if test_sample is not None:
            for i in range(len_pred - 1):
                Lt = get_Lt(a, test_sample[:, i:i+1], Lt_1, Tt_1)
                Tt = get_Tt(b, Lt, Lt_1, Tt_1)
                list_Lt.append(Lt)
                list_Tt.append(Tt)
                list_It.append(Lt + Tt)
                Lt_1 = Lt
                Tt_1 = Tt
        else:
            for i in range(len_pred - 1):
                Lt = get_Lt(a, Lt_1 + Tt_1, Lt_1, Tt_1)
                Tt = get_Tt(b, Lt, Lt_1, Tt_1)
                list_Lt.append(Lt)
                list_Tt.append(Tt)
                list_It.append(Lt + Tt)
                Lt_1 = Lt
                Tt_1 = Tt
        return list_It, list_Lt, list_Tt
    def train(self, x0, bnds, mtd='L-BFGS-B'):
        result = minimize(self.fun, x0, method=mtd, bounds=bnds)
        return result
