"""
Python module to carry out quantile regression

This module can be used to estimate linear regression coefficients for
different quantiles for a give data set.

"""
# Created: Fri Feb 22, 2019  11:52pm
# Last modified: Fri Mar 22, 2019  03:53pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>

from tqdm import trange
import numpy as np
import sys
from scipy.optimize import linprog


def linear(X, Y, tau=[]):
    """
    Estimates quantile regression coefficients for given data and quantiles.

    Parameters
    ----------
    Y : numpy.ndarray of shape (N,)
        response variable sample of length N
    K : numpy.ndarray with shape (N, K), or (N,) if K = 1
        N samples of K independent variables used to explain Y
    tau: array like
        list of Ntau quantiles for which to estimate linear model coefficients

    Returns
    -------
    beta : numpy.ndarray of shape (Ntau, K + 1)
        array of length Ntau where i-th row has the linear model coefficients
        corresponding to the i-th quantile in the input array `tau`

    Raises
    ------
    AssertionError : error
                    quantiles are not provided
    AssertionError : error
                    length of sample sizes in X and Y are unequal

    Notes
    -----
    a) As of now, the only documentation is the one in this help text for the
    `linear` function. However, the formulation of the linear program in the
    function is based on the description provided in [1]. More detailed
    documentation based on this will come soon.
    b) Normal minimization using the cost function formulated by Koenker along
    with the scipy.optimize.minimize function does not converge and does not
    give the correct result.

    References
    ----------
    [1] https://stats.stackexchange.com/a/384913

    """
    ## INPUT HANDLING
    ## --------------
    # check if the list of quantiles is provided
    assert(len(tau) > 0), "Please give a list of quantiles"
    Ntau = len(tau)

    # check if sample lengths of X and Y are same
    N = Y.shape[0]
    assert(X.shape[0] == N), "Length of samples for X and Y are not same"

    # get the dimension of X, i.e. the number of explanatory variables
    try:
        K = X.shape[1]
    except IndexError:
        K = 1

    # add a column of 1's to X to account for the 0-th coefficient (intercept)
    X = np.c_[np.ones(N), X]

    ## SET UP LINEAR PROGRAM
    ## ---------------------
    # create appropriate quantities for the linear program
    i_N = np.ones(N)
    I_N = np.diagflat(i_N)
    A = np.c_[X, -X, I_N, -I_N]
    b = Y

    ## MAIN LOOP
    ## ---------
    # loop over the provided quantiles and estimate coefficients
    K_ = K + 1
    beta = np.zeros((Ntau, K + 1))
    for i in trange(Ntau):
        # create the coefficient matrix
        c = np.r_[
                  np.zeros(2 * K_),     # 2 times the number of coefficients
                  tau[i] * i_N,         # positive part of residuals
                  (1. - tau[i]) * i_N   # negative part of residuals
                  ]

        # use scipy.optimize.linprog to get solutions of the linear program
        res = linprog(c=c,
                      A_eq=A, b_eq=b,
                      method="interior-point",      # simplex doesn't work
                      bounds=(0, None),
#                       options = {"sym_pos":False}
                      )
        z = res.x

        # use the first 2K_ entries to estimate the desired coefficients
        beta[i] = z[0:K_] - z[K_:(2 * K_)]

    return beta
