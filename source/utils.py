"""
Module containing some useful routines.
"""

import os
import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def avoid_duplicate_name(filename):
    """
    If filename does already exist suggest a filename using an incrementer.
    """

    name, fmt = filename.split('.')
    # Check for a match in which case we need to find an alternative.
    match = os.path.exists(filename)
    idx = 1

    while match:
        filename = "%s_%s.%s" % (name, idx, fmt)
        match = os.path.exists(filename)
        idx += 1

        # Safety check.
        if idx > 50:
            break

    return filename


def compute_indicator_matrix_c(xp):
    """
    Compute an indicator matrix for each protected feature and return them as a dictionary.
    """
    Ip = dict()
    for i in range(xp.shape[1]):
        Ip[i] = xp[:, i].flatten()

    return Ip


def compute_indicator_matrix_r(xp):
    """
    Compute an indicator array for each protected feature and return them as a dictionary.
    """
    Ip = dict()
    protected_values = np.unique(xp)
    for p in protected_values:
        val = 1 * (xp == p)
        Ip[int(p)] = val.reshape(-1)

    return Ip


def compute_auxiliary_matrices_c(xnp, xp):
    """
    Compute the auxiliary matrices needed to define the disparate treatment index.
    """

    n_points = len(xnp)

    # Turn input dataframes into numpy arrays.
    xnp = xnp.values
    xp = xp.values

    # Compute matrix D.
    D = np.zeros((n_points, n_points))
    # for i in range(n_points):
    #     for j in range(i + 1):
    #         val = inverse_distance(xnp[i, :], xnp[j, :])
    #         D[i, j] = val
    #         D[j, i] = val

    # Compute matrix I.
    # here I assume for simplicity dim({xp}) = 1.
    # (i.e. there is only one protected feature)
    Ip = dict()
    protected_values = np.unique(xp).tolist()
    for i in range(xp.shape[1]):
        Ip[i] = xp[:, i].flatten()

    return D, Ip


def compute_auxiliary_matrices_r(xnp, xp):
    """
    Compute the auxiliary matrices needed to define the disparate treatment index.
    """

    n_points = len(xnp)

    # Turn input dataframes into numpy arrays.
    xnp = xnp.values
    xp = xp.values

    # Compute matrix D.
    D = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(i + 1):
            # val = inverse_distance(xnp[i, :], xnp[j, :])
            val = exp_distance(xnp[i, :], xnp[j, :])
            D[i, j] = val
            D[j, i] = val


    # Compute matrix I.
    # here I assume for simplicity dim({xp}) = 1.
    Ip = dict()
    protected_values = np.unique(xp).tolist()
    for p in protected_values:
        val = 1 * (xp == p)
        Ip[int(p)] = val.flatten()

    return D, Ip


def compute_auxiliary_matrices_r2(xnp, xp):
    """
    Compute the auxiliary matrices needed to define the disparate treatment index.
    """

    n_points = len(xnp)

    # Turn input dataframes into numpy arrays.
    xnp = xnp.values
    xp = xp.values

    # Compute matrix D.
    D = np.zeros((n_points, n_points))
    vfunc = np.vectorize(exp_distance)
    for i in range(n_points):
        D[i, i:] = vfunc(xnp[i, ], xnp[i:, ])
    D = D + D.T - D.diagonal()



    # Compute matrix I.
    # here I assume for simplicity dim({xp}) = 1.
    Ip = dict()
    protected_values = np.unique(xp).tolist()
    for p in protected_values:
        val = 1 * (xp == p)
        Ip[p] = val.flatten()

    return D, Ip


def compute_discrimination_index_c(y, D, Ip):
    """
    Compute the discrimination index of a given dataset.
    """

    n_points = len(y)
    eps = 1e-6
    Iy = dict()
    # Since the targets are either 0 or 1, I_1 is y itself, in matrix form, and I_0 its complementary.
    target_values = [0, 1]
    for p in target_values:
        if p == 1:
            Iy[p] = y
        else:
            Iy[p] = 1-y

    tot = .0
    for xp in Ip.values():
        for y in Iy.values():
            DI = D * xp
            for i in range(n_points):
                tmp = (1.0 / (eps + np.sum(D[i, :]))) * (D[i, :].dot(y)) - \
                      (1.0 / (eps + np.sum(DI[i, :]))) * (DI[i, :].dot(y))
                tot += np.abs(tmp)

    return tot


def compute_discrimination_index_r(y, D, I):
    """
    Compute the disparate treatment discrimination index of a given dataset.
    """

    n_points = len(y)
    tot = .0
    eps = 1e-6
    for p, Ip in I.items():
        DI = D * Ip
        for i in range(n_points):
            tmp = (1.0 / (eps + np.sum(D[i, :]))) * (D[i, :].dot(y)) - \
                  (1.0 / (eps + np.sum(DI[i, :]))) * (DI[i, :].dot(y))
            tot += np.abs(tmp)

    return tot


def didi_c(y, I):

    y = np.array(y).flatten()
    didi = 0
    N = len(y)
    for key, Ip in I.items():
        Np = np.sum(Ip)
        if Np > 0:
            val = 2 * (np.sum(y) / N - Ip.dot(y) / Np)
            didi += np.abs(val)

    return didi


def didi_r(y, I):
    """
    Compute the disparate impact discrimination index of a given dataset.
    """

    n_points = len(y)
    tot = .0
    for val in I.values():
        Np = np.sum(val)
        if Np > 0:
            tmp = (1.0 / Np) * (val.dot(y)) - \
                  (1.0 / n_points) * np.sum(y)
            tot += np.abs(tmp)

    return float(tot)


def euclidean_dist(x, y):
    """
    Compute the euclidean distance between x and y.
    """
    x = np.array(x).ravel()
    y = np.array(y).ravel()

    return np.sqrt(np.sum(np.square(x - y)))


def get_top_features(x, y, n=10):
    """
    Get the n most relevant features of the input dataset.

    In order to do feature selection, we have to fix a model: here we use a random forest for simplicity,
    since feature importance is already implemented in sklearn routines.
    """

    x_scaled = MinMaxScaler().fit_transform(x)
    y_scaled = MinMaxScaler().fit_transform(y)
    features = list(x.columns)
    tmp = SelectKBest(k=n).fit(x_scaled, y_scaled)
    feat_ranking = tmp.scores_

    return list(np.take(features, np.argsort(feat_ranking)))[:n]


def inverse_distance(x, y):
    """
    The inverse of the euclidean distance.
    """
    return 1.0 / (1 + euclidean_dist(x, y))


def exp_distance(x, y):
    """
    The exponential of the inverse of the euclidean distance.
    """
    return np.exp(-euclidean_dist(x, y))


def mean_squared_error(x, y):
    """
    Compute the mean squared error between two arrays.
    """

    x = np.array(x).ravel()
    y = np.array(y).ravel()

    return np.mean(np.square(x - y))


def mean_absolute_error(x, y):
    """
    Compute the mean squared error between two arrays.
    """

    x = np.array(x).ravel()
    y = np.array(y).ravel()

    return np.mean(np.abs(x - y))


def accuracy(x, y):
    """
    Compute the accuracy between the two arrays.
    """

    x = np.array(x).ravel()
    y = np.array(y).ravel()

    return np.mean(np.where(x == y, 1, 0))


def mean_absolute_perc_error(y, yhat):
    """
    Compute the mean squared error between two arrays.
    """
    eps = 1e-6
    y = np.array(y).ravel()
    yhat = np.array(yhat).ravel()

    return np.mean(np.abs((y - yhat) / (eps + y)))


def read_from_pickle(filename):
    """
    Read results from pickle and returns them as a list.
    """
    pickle_in = list()

    with open(filename, "rb") as file:
        try:
            while True:
                tmp_in = pickle.load(file)
                pickle_in.append(tmp_in)
        except EOFError:
            pass

    data = pickle_in[0]

    return data
