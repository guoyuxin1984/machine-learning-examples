# --*-- coding:utf-8 --*--
"""
created on 2018-9-18
created by GUO Yuxin
"""

import numpy as np
import matplotlib.pyplot as plt
import prostate
import utils


class OLS:
    def __init__(self, train_input, train_output):
        self.x_org = train_input
        self.x = np.insert(self.x_org, obj=0, values=1, axis=1)
        self.x = utils.standardize(self.x)
        self.y = train_output
        self.xtx = self.x.T.dot(self.x)
        self.u, self.s, self.vt = np.linalg.svd(self.xtx)
        self.s = np.diag(self.s)
        self.xtx_inv = self.vt.T.dot(np.linalg.pinv(self.s)).dot(self.u.T)

    def ols(self):
        beta_hat = self.xtx_inv.dot(self.x.T).dot(self.y)
        self.y_hat = self.x.dot(beta_hat)
        return beta_hat

    def ols_gradient(self, learning_rate=0.001, loop_count=1000, gradient_error=0.001):
        beta_hat = np.ones((self.x.shape[1], 1), dtype=float)
        for i in range(loop_count):
            self.y_hat = self.x.dot(beta_hat)
            gradient = -self.x.T.dot(self.y - self.y_hat)
            beta_hat -= learning_rate * gradient
            if np.linalg.norm(gradient, ord=2, axis=0) < gradient_error:
                break
        return beta_hat

    def y_var_hat(self):
        y_variance_hat = np.sum((self.y - self.y_hat) ** 2)/(self.x.shape[0] - self.x.shape[1] - 1)
        return y_variance_hat

    def z_score(self, y_std_hat=None, beta_hat=None):
        diag_xtx = np.diag(self.xtx_inv)
        if y_std_hat is None:
            y_std_hat = np.sqrt(self.y_var_hat())
        if beta_hat is None:
            beta_hat = self.ols()
        z = beta_hat/(y_std_hat * np.sqrt(diag_xtx)).reshape(self.x.shape[1], 1)
        return z

    def rss(self, beta_hat=None):
        if beta_hat is None:
            beta_hat = self.ols()
        rss = (self.y - self.x.dot(beta_hat)).T.dot(self.y - self.x.dot(beta_hat))
        return rss


if __name__ == '__main__':
    # toll trail
    '''
    x = np.linspace(-2, 2, 100)
    x1 = x
    x2 = x
    for i in range(2):
        x1 = x1 * x2
        x = np.column_stack((x, x1))

    beta = np.random.rand(3, 1)
    y = x.dot(beta) + np.random.rand(100, 1)
    plt.plot(x2, y, 'ro')

    ols_ = OLS(x, y)
    beta_hat = ols_.ols()
    y_std_hat = np.sqrt(ols_.y_var_hat())
    plt.plot(x2, ols_.y_hat, 'b-')
    plt.plot(x2, ols_.y_hat - y_std_hat * 2, 'y--')
    plt.plot(x2, ols_.y_hat + y_std_hat * 2, 'g--')
    print(ols_.z_score(y_std_hat))
    plt.show()
    '''
    inputs, outputs, is_train = prostate.load_data()
    x = inputs[is_train == b'T']
    y = outputs[is_train == b'T']
    if len(y.shape) == 1:
        y = y.reshape((x.shape[0], 1))
    ols = OLS(x, y)

    print(ols.ols_gradient())
    print(ols.ols())
    print(ols.z_score())

