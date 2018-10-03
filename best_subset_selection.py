"""
created on 2018-9-24
created by GUO Yuxin
"""
import numpy as np
import prostate
from ols import OLS
import matplotlib.pyplot as plt


class BestSubset:
    def __init__(self):
        pass

    def best_subset(self, x, y):
        n_features = x.shape[1]
        lst = []
        for k in range(1, n_features):
            lst.extend(combination(n_features, k))
        lst.append(list(range(n_features)))

        rss = []
        beta_hat = []

        for i in lst:
            selected_x = x[:, i]
            ols = OLS(selected_x, y)
            beta = ols.ols()
            rss.extend(ols.rss(beta))
            beta_hat.append(beta)

        for k in range(0, n_features):
            rss_list = [rss[i] for i, v in enumerate(lst) if len(v) == k+1]
            rss_list = np.array(rss_list)
            rss_list = rss_list.reshape(rss_list.shape[1], rss_list.shape[0])
            plt.plot([k+1], rss_list, 'go')
        plt.axis([0, 9, 0, 100])
        plt.show()


def combination(n_features, order):
    lst = []
    length = n_features
    pos = np.zeros((length, 1), int)
    pos[:order] = 1
    lst.append(list(range(order)))
    while np.sum(pos[-order:]) != order:
        for i in range(length-1):
            if pos[i] == 1 and pos[i+1] == 0:
                pos[i] = 0
                pos[i + 1] = 1
                one_count = np.sum(pos[:i])
                for k in range(one_count):
                    pos[:one_count] = 1
                    pos[one_count:i] = 0
                lst.append([index for index, v in enumerate(pos) if v == 1])
                break
    return lst


if __name__ == '__main__':
    inputs, outputs, is_train = prostate.load_data()
    x = inputs[is_train == b'T']
    y = outputs[is_train == b'T']
    if len(y.shape) == 1:
        y = y.reshape((x.shape[0], 1))

    bs = BestSubset()
    bs.best_subset(x, y)


