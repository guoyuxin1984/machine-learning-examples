"""
created on 2018-9-24
created by GUO Yuxin
"""
import numpy as np


def centralize(x):
    avg = np.mean(x, axis=0)
    cen = x - avg
    if len(x.shape) == 1:
        if list(x) == list(np.zeros((1, len(x)))):
            return x
        return cen
    for i in range(cen.shape[1]):
        if list(cen[:, i]) == list(np.zeros(cen[:, i].shape)):
            cen[:, i] = x[:, i]
    return cen


def standardize(x):
    """
    standardize the input to have mean 0 and L2 norm 1
    :param x:
    :return: standardize the input to have mean 0 and L2 norm 1
    """
    cen = centralize(x)
    std = np.std(cen, axis=0) * np.sqrt(x.shape[0])
    std = cen/std
    return std


def standardize_z_score(x):
    """
    standardize the input to have mean 0 and std 1
    :param x:
    :return: standardize the input to have mean 0 and std 1
    """
    cen = centralize(x)
    std = np.std(cen, axis=0)
    std = cen / std
    return std


def load_data_prostate():
    """
    to get the prostate data
    :return:  return input matrix x, response vector y and the is training vector
    """
    inputs = np.genfromtxt("data/prostate.data", skip_header=1, dtype=float, usecols=[1, 2, 3, 4, 5, 6, 7, 8])
    outputs = np.genfromtxt("data/prostate.data", dtype=float, skip_header=1, usecols=[9])
    is_train = np.genfromtxt("data/prostate.data", dtype='|S', skip_header=1, usecols=[10])
    return inputs, outputs, is_train


def load_data_diabetes():
    """
    load the diabetes data with 10 predictors and 1 response
    :return: return input matrix x and response vector y
    """
    x = np.genfromtxt('data/diabetes.data', skip_header=1, dtype=float, usecols=range(10))
    y = np.genfromtxt('data/diabetes.data', skip_header=1, dtype=float, usecols=[10])
    return x, y
