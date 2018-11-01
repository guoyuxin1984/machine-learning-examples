"""
created on 2018-10-10
created by GUO Yuxin
"""

import numpy as np
import prostate
import utils


def forward_stagewise(x, y, learning_rate=0.01):
    n_samples = x.shape[0]
    x = np.column_stack((np.ones((n_samples, 1), dtype=float), x))
    n_features = x.shape[1]
    beta_hat = np.zeros((n_features, 1), dtype=float)
    beta_hat[0] = np.mean(y)
    residual = y - beta_hat[0]
    selected_feature = 0
    corr = float('-Inf')
    count = 0
    while np.abs(corr) > 0.001:
        count += 1
        corr = 0.0
        for i in range(1, n_features):
            temp_corr = np.abs(residual.T.dot(x[:, i]))
            if temp_corr > corr:
                corr = temp_corr
                selected_feature = i
        beta_hat[selected_feature] += learning_rate * residual.T.dot(x[:, selected_feature])/residual.T.dot(residual)
        residual = y - x.dot(beta_hat).reshape(n_samples,)
    return beta_hat


if __name__ == '__main__':
    x, y, is_train = prostate.load_data()
    x = x[is_train == b'T']
    y = y[is_train == b'T']

    x = utils.standardize_z_score(x)
    forward_stagewise(x, y, 0.8)
