"""
created on 2018-9-25
created by GUO Yuxin
"""

import numpy as np
import prostate
import utils


def forward_stepwise(x, y, k=5):
    r = y - np.mean(y)
    n_cols = x.shape[1]
    used_col = np.arange(n_cols, n_cols+k)
    cos = float('-inf')
    x_selected = np.ones((x.shape[0], 1))
    for index in range(k):
        for i in range(n_cols):
            if i in used_col:
                continue
            x_temp = x[:, i:i+1]
            cos_temp = x_temp.T.dot(r)/np.linalg.norm(x_temp, axis=0)
            if cos_temp > cos:
                used_col[index] = i
                cos = cos_temp

        cos = float('-inf')
        x_selected = np.column_stack((x_selected, x[:, used_col[index]:used_col[index]+1]))
        u, s, vt = np.linalg.svd(x_selected.T.dot(x_selected))
        s = np.diag(s)
        inv_xtx = vt.T.dot(np.linalg.pinv(s)).dot(u.T)
        r = y - x_selected.dot(inv_xtx).dot(x_selected.T).dot(y)
    return used_col, x_selected


if __name__ == '__main__':
    x, y, is_train = prostate.load_data()
    x = utils.standardize(x)
    used_col, x_select = forward_stepwise(x, y)
