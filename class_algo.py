#### Algorithm implementation

import matplotlib.pylab as plt
import numpy as np
from scipy.linalg import logm, expm
from numpy.linalg import inv


#Get a Matrix A and returns exp(A)
def calc_matrix_exp(A):
    exp_A = expm(A)
    return exp_A

#Get a Matrix A and returns ln(A)
def calc_matrix_ln(A):
    ln_A = logm(A)
    return ln_A

#getting W matrix of 10X(img_dim^2) and x is vector of 1X(img_dim^2)
#k is the number of the iteration to take the weight for
#then calc the y_n_k = exp(w_k.T .* x_N) / sum (w_k_j.T .* x_N)
#result in (img_dim^2)X(img_dim^2) matrix

def calc_y_n_k(W, x, k):
    # W is the matrix of all the weight vectors
    # x is the image vector
    x = np.array([x])
    w_k = np.array([W[k]]).T

    numerator_mul = w_k @ x  # numerator_mul shape:  (785, 785)

    numerator_res = calc_matrix_exp(numerator_mul)

    denominator_sum = 0
    for j in range(W.shape[0]):

        w_j = np.array([W[j]]).T
        denominator_mul = w_j @ x
        denominator_res = calc_matrix_exp(denominator_mul) # denominator_res shape:  (785, 785)
        denominator_sum += denominator_res

    denominator_inverse = inv(denominator_sum)

    y_n_k = numerator_res @ denominator_inverse # y_n_j shape:  (785, 785)
    return y_n_k

#getting W matrix of 10X(img_dim^2) and X is vector of training data
#t is the real result of the training
#then calc the E = SUM(SUM(t_n_k*ln(y_n_k))
def calc_loss_entropy_cross(W,t,X,N):

    E_w = 0

    for n in range(N):
        t_n = int(t[n])
        for k in range(W.shape[0]):
            x_n = X[n] # x_n shape:  (785,)
            y_n_k = calc_y_n_k(W,x_n,k) # y_n_j shape:  (785, 785)
            ln_y_n_k = calc_matrix_ln(y_n_k) # ln_y_n_k.shape (785, 785)
            iter_mul = t_n*y_n_k # iter_mul.shape (785, 785)
            E_w += iter_mul # E_w.shape (785, 785)


    return E_w

def calc_grad(W,t,X,N,j):
    grad = 0

    for n in range(N):
        t_n = int(t[n])
        x_n = X[n] # x_n shape:  (785,)
        y_n_j = calc_y_n_k(W, x_n, j) # y_n_j shape:  (785, 785)
        sub = y_n_j - t_n # sub shape:  (785, 785)
        grad += sub@x_n # grad shape (785,)

    return grad

def update_weights(W,h,t,X,N):
    for j in range(W.shape[0]):
        old_weight_j = W[j]
        grad = calc_grad(W,t,X,N,j)
        new_weight_j = W[j] - h*grad
        W[j] = new_weight_j
    return W

def calc_percision():
    pass

def classiffy(data):
    print("start classifiy")
    X_train, X_test, X_valid, t_train, t_test, t_valid = data
    h = 0.01 # heta
    N = X_train.shape[0]  # Number of samples to train

    W = np.random.random((10, 784))
    W = np.column_stack((W, [1] * W.shape[0]))  # adding bias 1

    iter_num = 100
    percision_delta = 0

    # for i in range(iter_num):
    #     W = update_weights(W,h,t_train,X_train,N)
    #     loss = calc_loss_entropy_cross(W,t_train,X_train,N)
    #     percision = calc_percision()
    #     if