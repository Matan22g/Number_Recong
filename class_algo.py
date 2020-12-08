import matplotlib.pylab as plt
import numpy as np
from scipy.linalg import expm
from numpy.linalg import inv


def calc_matrix_exp(A):
    exp_A = expm(A)
    return exp_A


def calc_y_n_k(W, x, k):
    # W is the matrix of all the weight vectors
    # x is the image vector
    x = np.array([x])
    w_k = np.array([W[k]]).T

    numerator_mul = w_k @ x

    numerator_res = calc_matrix_exp(numerator_mul)

    denominator_sum = 0
    for j in range(W.shape[0]):

        w_j = np.array([W[j]]).T
        denominator_mul = w_j @ x
        denominator_res = calc_matrix_exp(denominator_mul)
        denominator_sum += denominator_res

    denominator_inverse = inv(denominator_sum)
    print(numerator_res.shape)
    print(denominator_inverse.shape)

    y_n_k = numerator_res @ denominator_inverse
    return y_n_k

def calc_loss_entropy_cross(W,t,X,N):

    for n in range(N):
        t_n = t[n]

        for k in range(W.shape[0]):
            x_n = X[n]
            y_n_k = calc_y_n_k(W,x_n,k)

def classiffy(data):
    X_train, X_test, X_valid, t_train, t_test, t_valid = data

    N = X_train.shape[0]  # Number of samples to train

    W = np.random.random((10, 784))
    W = np.column_stack((W, [1] * W.shape[0]))  # adding bias 1
    calc_y_n_k(W,X_train[0],0)