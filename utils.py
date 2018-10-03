"""
created on 2018-9-24
created by GUO Yuxin
"""
import numpy as np


def centralize(x):
    avg = np.mean(x, axis=0)
    cen = x - avg
    for i in range(cen.shape[1]):
        if list(cen[:, i]) == list(np.zeros(cen[:, i].shape)):
            cen[:, i] = x[:, i]
    return cen


def standardize(x):
    cen = centralize(x)
    std = []
    for i in range(x.shape[1]):
        std.append(np.sqrt(np.sum(cen[:, i] ** 2)/x.shape[0]))
    std = cen/std
    return std





