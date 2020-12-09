#### Algorithm implementation

import sys
import matplotlib.pylab as plt
import numpy as np

def progressBar(i, total, percision):
    percent = (i / total) * 100
    sys.stdout.write("\rIter Done %i/%i, Percision %i%%" % (i, total, percision))
    if i == total - 1:
        print("\n")
    sys.stdout.flush()


def my_softmax(x):
    # Compute softmax values for each row in matrix x
    e_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return e_x / e_x.sum(axis=1).reshape(-1, 1)


def predict(im, W):
    W_xn_mat = (im / 255) @ W.T

    im = np.reshape(im[:-1], (-1, 28))
    plt.imshow(im, cmap='gray')
    plt.show()

    y_nk = my_softmax([W_xn_mat])
    return np.argmax(y_nk, axis=1)[0]


# getting W matrix of 10X(img_dim^2) and X is vector of training data
# t is the real result of the training
# then calc the E = SUM(SUM(t_n_k*ln(y_n_k))

def calc_loss_entropy_cross(y_nk, t):
    N = t.shape[0]
    E_w = 0
    for n in range(N):
        for k in range(10):
            E_w += t[n][k] * y_nk[n][k]
    return -E_w


def update_weights(y_nk, X, W, t, h):
    grad = X.T @ (y_nk - t)
    W = W - (h * grad).T
    return W


def calc_percision(X_valid, t_valid, W):
    valid_amount = X_valid.shape[0]

    W_xn_mat = X_valid @ W.T
    y_nk = my_softmax(W_xn_mat)

    num_Correct = (np.argmax(y_nk, axis=1) == np.argmax(t_valid, axis=1))

    acc = num_Correct.sum() / valid_amount
    percision = (acc) * 100

    return percision


#### Method to run it all

def classiffy(data, heta, iter):
    # print("Start classifiy")
    X_train, X_test, X_valid, t_train, t_test, t_valid = data
    h = heta  # heta
    N = X_train.shape[0]  # Number of samples to train

    W = np.random.random((10, 784))
    W = np.column_stack((W, [1] * W.shape[0]))  # adding bias 1

    iter_lim = iter

    percision_delta = 0.5
    last_percision = 0
    percision = 0
    num_of_consistency = 0
    max_val = np.amax(X_train)
    # print("Heta:", h)

    # Convert str to int
    t_train_int = t_train.astype(np.int)
    t_valid_int = t_valid.astype(np.int)

    # Build t matrices
    t_train_mat = np.zeros((t_train.shape[0], 10), dtype=int)
    t_valid_mat = np.zeros((t_valid.shape[0], 10), dtype=int)
    t_train_mat[[range(t_train.shape[0])], [t_train_int]] = 1
    t_valid_mat[[range(t_valid.shape[0])], [t_valid_int]] = 1

    X_train, X_test, X_valid = X_train / 255, X_test / 255, X_valid / 255

    percision_array = []
    loss_array = []
    i = 0
    while True:
        i += 1
        progressBar(i, iter_lim, percision)

        W_xn_mat = X_train @ W.T
        y_nk = my_softmax(W_xn_mat)

        loss = calc_loss_entropy_cross(y_nk, t_train_mat)
        loss_array.append(loss)

        percision = calc_percision(X_valid, t_valid_mat, W)
        percision_array.append(percision)

        W = update_weights(y_nk, X_train, W, t_train_mat, h)

        if last_percision == 0:
            last_percision = percision
            continue

        if percision - last_percision < percision_delta:
            num_of_consistency += 1
        else:
            num_of_consistency = 0

        if num_of_consistency == 35:
            print("\n35 iter without change stoping..")
            break

        if i > iter_lim:
            print("\nencounter iter limit stoping..")
            break

    print("heta: ", h)
    print("iter: ", i)
    print("percision: ", percision)
    return W, percision_array, loss_array


def main_classiffy(data):

    heta = 1.3
    iteration = 20
    W, percision_array, loss_array = classiffy(data, heta, iteration)

    X_train, X_test, X_valid, t_train, t_test, t_valid = data

    #### Further amusement
    x1_for_plot = [i + 1 for i in range(len(percision_array))]
    x2_for_plot = [i + 1 for i in range(len(loss_array))]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(x1_for_plot, percision_array, label='Percision Percantage')
    ax2.plot(x2_for_plot, loss_array, label='loss_array')
    ax1.set_xlabel('Iteration (#)')
    ax1.set_ylabel('Percision (%)')
    ax1.set_title('Percision Percantage')
    ax1.legend()
    ax2.set_xlabel('Iteration (#)')
    ax2.set_ylabel('loss_array')
    ax2.set_title('loss_array')
    ax2.legend()
    plt.show()


    # Convert string to int
    t_train_int = t_train.astype(np.int)
    t_valid_int = t_valid.astype(np.int)
    t_test_int = t_test.astype(np.int)

    # Build t matrices
    t_train_mat = np.zeros((t_train.shape[0], 10), dtype=int)
    t_valid_mat = np.zeros((t_valid.shape[0], 10), dtype=int)
    t_test_mat = np.zeros((t_test.shape[0], 10), dtype=int)

    t_train_mat[[range(t_train.shape[0])], [t_train_int]] = 1
    t_valid_mat[[range(t_valid.shape[0])], [t_valid_int]] = 1
    t_test_mat[[range(t_test.shape[0])], [t_test_int]] = 1

    X_train, X_test, X_valid = X_train / 255, X_test / 255, X_valid / 255

    percision_t_test = calc_percision(X_train, t_train_mat, W)
    percision_t_train = calc_percision(X_test, t_test_mat, W)
    percision_t_valid = calc_percision(X_valid, t_valid_mat, W)

    print("percision_t_train: ", percision_t_train)
    print("percision_t_valid: ", percision_t_valid)
    print("percision_t_test: ", percision_t_test)


def test(W,x,t):

    t_valid_int = t.astype(np.int)

    t_valid_mat = np.zeros((14000, 10), dtype=int)
    t_valid_mat[[range(14000)], [t_valid_int]] = 1

    X_valid = x / 255

    valid_amount = X_valid.shape[0]
    W_xn_mat = X_valid @ W.T
    y_nk = my_softmax(W_xn_mat)
    num_Correct = (np.argmax(y_nk, axis=1) == np.argmax(t_valid_mat, axis=1))
    acc = num_Correct.sum() / valid_amount

    real = np.argmax(t_valid_mat, axis=1)
    guess = np.argmax(y_nk, axis=1)
    X_valid = X_valid * 255

    for i in range(10):
        print("real: ", real[i])
        print("guess: ", guess[i])

        im = X_valid[i]
        im = np.reshape(im[:-1], (-1, 28))
        plt.imshow(im, cmap='gray')
        plt.show()