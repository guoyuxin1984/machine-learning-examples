import numpy as np
import utils
import prostate


def ridge_regression(x, y, lam=0.1):
    x = utils.standardize(x)
    eye = np.eye(x.shape[1])
    beta_0 = np.mean(y)
    u, s, vt = np.linalg.svd(x)
    # s = np.diag(s)
    beta = np.linalg.pinv(x.T.dot(x) + lam * eye).dot(x.T).dot(y)
    beta = np.insert(beta, 0, beta_0)
    df = np.sum(s**2/(s**2 + lam))
    return beta, df


def ridge_reg_with_df(x, y, df=5, error=0.001):
    x = utils.standardize(x)
    eye = np.eye(x.shape[1])
    beta_0 = np.mean(y)
    u, s, vt = np.linalg.svd(x)
    # newton-raphson method to solve lambda
    lam = 1
    while (np.sum(s**2/(s**2 + lam)) - df) > error:
        yx = np.sum(s**2/(s**2 + lam)) - df
        yx_slop = -np.sum(s**2/(s**2 + lam)**2)
        lam = lam - yx/yx_slop
    # end of newton-raphson method
    beta = np.linalg.pinv(x.T.dot(x) + lam * eye).dot(x.T).dot(y)
    beta = np.insert(beta, 0, beta_0)
    return beta, lam


if __name__ == '__main__':
    x, y, is_train = prostate.load_data()
    x = x[is_train == b'T']
    y = y[is_train == b'T']
    ridge_regression(x, y, 23.12)
    ridge_reg_with_df(x, y)
    pass