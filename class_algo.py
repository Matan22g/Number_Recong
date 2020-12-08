#### Algorithm implementation
import sys

import matplotlib.pylab as plt
import numpy as np
from scipy.linalg import logm, expm
from numpy.linalg import inv



def calc_y_n_k(W, x, k):
    # W is the matrix of all the weight vectors
    # x is the image vector
    x = np.array([x]).T
    w_k = np.array([W[k]])

    numerator_mul = w_k @ x  # numerator_mul shape:  1X1
    # numerator_res = calc_matrix_exp(numerator_mul)
    try:
        numerator_res = np.exp(numerator_mul[0][0])
    except RuntimeWarning as e:
        return 0

    denominator_sum = 0

    for j in range(W.shape[0]):

        w_j = np.array([W[j]])
        denominator_mul = w_j @ x
        try:
            denominator_res = np.exp(denominator_mul[0][0])
        except RuntimeWarning as e: # skipping that iter
            denominator_res = 0  #
        denominator_sum += denominator_res

    try:
        y_n_k = numerator_res / denominator_sum

    except RuntimeWarning as e:
        return 0

    return y_n_k


# getting W matrix of 10X(img_dim^2) and X is vector of training data
# t is the real result of the training
# then calc the E = SUM(SUM(t_n_k*ln(y_n_k))
def calc_loss_entropy_cross(W, t, X, N):
    E_w = 0
    for n in range(N):
        t_n = int(t[n])

        for k in range(W.shape[0]):
            x_n = X[n]  # x_n shape:  (785,)
            y_n_k = calc_y_n_k(W, x_n, k)
            try:
                ln_y_n_k = np.log(y_n_k)
            except RuntimeWarning as e:
                ln_y_n_k = 0
            iter_mul = t_n * y_n_k
            E_w += iter_mul
    return E_w


def progressBar(i, total, percision):
    percent = (i / total) * 100
    sys.stdout.write("\rPercent Done %i%% Percision %i" % (percent,percision))
    if i == total - 1:
        print("\n")
    sys.stdout.flush()


def calc_grad(W, t, X, N, j):
    grad = 0
    for n in range(N):
        t_n = int(t[n])
        x_n = X[n]
        y_n_j = calc_y_n_k(W, x_n, j)
        sub = y_n_j - t_n
        grad += sub * x_n
    return grad

def update_weights(W, h, t, X, N):
    for j in range(W.shape[0]):
        grad = calc_grad(W, t, X, N, j)

        new_weight_j = W[j] - h * grad
        W[j] = new_weight_j
    return W

def predict(x_n, W):
    j_max = 0
    max_val = 0
    for j in range(W.shape[0]):
        y_n_j = calc_y_n_k(W, x_n, j)
        if y_n_j > max_val:
            max_val = y_n_j
            j_max = j
    return j_max


def calc_percision(X_valid, t_valid, W):
    num_of_valid = X_valid.shape[0]
    correct_amount = 0
    valid_amount = X_valid.shape[0]
    for i in range(valid_amount):
        prediction = predict(X_valid[i], W)
        real = int(t_valid[i])
        # print("prediction", prediction)
        # print("t_valid["+str(i)+"] " + str(t_valid[i]))
        if prediction == real:
            correct_amount += 1
    # print("correct_amount",correct_amount)
    percision = (correct_amount / valid_amount) * 100
    # print("percision: " + str(percision)+ "%")
    return percision



def classiffy(data):
    print("start classifiy")
    X_train, X_test, X_valid, t_train, t_test, t_valid = data
    h = 0.1  # heta
    N = X_train.shape[0]  # Number of samples to train

    W = np.random.random((10, 784))
    W = np.column_stack((W, [1] * W.shape[0]))  # adding bias 1

    # im = X_valid[0]
    # im = np.reshape(im[:-1], (-1, 28))
    # plt.imshow(im, cmap = 'gray')
    # plt.show()

    iter_num = 100
    percision_delta = 2.5
    last_percision = 0
    percision = 0
    num_of_consistency = 0

    for i in range(iter_num):
        progressBar(i, iter_num, percision)

        # old_W = np.copy(W)
        W = update_weights(W, h, t_train, X_train, N)
        # print("for: ", i , (old_W==W).all())

        # loss = calc_loss_entropy_cross(W, t_train, X_train, N)
        percision = calc_percision(X_valid, t_valid, W)

        if last_percision == 0:
            last_percision = percision
            continue

        if percision - last_percision < percision_delta:
            num_of_consistency += 1
        else:
            num_of_consistency = 0
        if num_of_consistency == 10:
            print("10 iter without change stoping..")
