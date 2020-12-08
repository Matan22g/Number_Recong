#### Importing the Database

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import matplotlib.pylab as plt
import numpy as np




cut = 1

def get_data():
    mnist = fetch_openml('mnist_784')
    X = mnist['data'].astype('float64')
    t = mnist['target']

    amount = int(X.shape[0]/cut)
    # amount = X.shape[0]

    random_state = check_random_state(1)
    permutation = random_state.permutation(amount)
    X = X[permutation]
    # X = np.insert(X,X.shape[1],1,axis=1)
    t = t[permutation]

    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2)
    X_train, X_valid,t_train, t_valid = train_test_split(X_train, t_train, test_size=0.25)

    # The next lines standardize the images

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_valid = scaler.transform(X_valid)

    X_train = np.column_stack((X_train, [1] * X_train.shape[0]))
    X_test = np.column_stack((X_test, [1] * X_test.shape[0]))
    X_valid = np.column_stack((X_valid, [1] * X_valid.shape[0]))

    return [X_train, X_test, X_valid, t_train, t_test, t_valid]
