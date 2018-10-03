"""
created on 2018-9-23
created by GUO Yuxin
"""

import numpy as np


def load_data():
    inputs = np.genfromtxt("data/prostate.data", skip_header=1, dtype=float, usecols=[1, 2, 3, 4, 5, 6, 7, 8])
    outputs = np.genfromtxt("data/prostate.data", dtype=float, skip_header=1, usecols=[9])
    is_train = np.genfromtxt("data/prostate.data", dtype='|S', skip_header=1, usecols=[10])
    return inputs, outputs, is_train

