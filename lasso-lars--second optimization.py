"""
created on 2018-10-18
created by GUO Yuxin
"""
import numpy as np
import utils
import matplotlib.pyplot as plt


def get_gamma_next_index(used_x, c_max, c_vec, a, a_vec):
    gamma = []
    if len(used_x) == len(c_vec):
        return c_max / a, -1
    for i in range(len(a_vec)):
        if i in used_x:
            gamma.append(float('Inf'))
            continue
        gamma_add = (c_max + c_vec[i]) / (a + a_vec[i])
        gamma_sub = (c_max - c_vec[i]) / (a - a_vec[i])
        if (gamma_add < gamma_sub) and gamma_add > 0:
            gamma.extend(gamma_add)
        elif gamma_sub > 0:
            gamma.extend(gamma_sub)
        else:
            gamma.extend(gamma_add)
    return min(gamma), np.argmin(gamma)


if __name__ == '__main__':
    x, y = utils.load_data_diabetes()
    n_samples = x.shape[0]
    n_features = x.shape[1]
    x = utils.standardize(x)
    y = utils.centralize(y)
    y = y.reshape(n_samples, 1)
    y_hat = np.zeros((n_samples, 1))
    active_index = []
    axis_active_index = []
    active_x = []
    beta_hat = np.zeros((1, n_features))
    x_axis = [0]
    beta_ols = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y).reshape(n_features)
    next_index = -1
    remove_index = -1
    while len(active_index) < n_features:
        i = len(active_index)
        residue = y - y_hat
        c = x.T.dot(residue)
        c_abs = np.abs(c)
        c_max = max(c_abs)

        if i == 0:
            g = 1
            g_inv = g
            e = 1
            a = 1
            u = x[:, np.argmax(c_abs)] * np.sign(c[np.argmax(c_abs)])
            w = a * g_inv * e
            active_index.append(np.argmax(c_abs))
            axis_active_index = np.copy(active_index)
        else:
            if remove_index == -1:
                active_index.append(next_index)
                if next_index not in axis_active_index:
                    axis_active_index = np.copy(active_index)
            else:
                remove_index = -1
            active_x = x[:, active_index] * np.sign(c[active_index]).reshape(1, len(active_index))
            g = active_x.T.dot(active_x)
            g_inv = np.linalg.pinv(g)
            e = np.ones((g.shape[0], 1))
            a = 1 / np.sqrt(e.T.dot(g_inv).dot(e))
            u = active_x.dot(g_inv).dot(e) * a
            w = a * g_inv.dot(e)

        a_vec = x.T.dot(u)
        gamma, next_index = get_gamma_next_index(active_index, c_max, c, a, a_vec)
        d = np.sign(c[active_index]) * w
        gamma_hat = list(- beta_hat[i, active_index] / d.reshape(len(active_index), ))
        gamma_hat_pos = [x for x in gamma_hat if x > 0]

        if len(gamma_hat_pos) > 0:
            gamma_hat_min = min(gamma_hat_pos)
            if gamma_hat_min < gamma:
                remove_index = gamma_hat.index(gamma_hat_min)
                gamma = gamma_hat_min
            #elif len(active_index) == n_features:
            #    break

        temp_beta = np.zeros((1, n_features))
        if i == 0:
            temp_beta[0, active_index] = gamma * np.sign(c[active_index]).reshape((len(active_index),))
        elif i < n_features:
            temp_beta[0, active_index] = (gamma * a * g_inv.dot(e) * np.sign(c[active_index])).reshape(
                len(active_index), )

        temp_beta = temp_beta + beta_hat[-1]
        beta_hat = np.row_stack((beta_hat, temp_beta))
        x_axis.append(np.sum(np.abs(beta_hat[-1, active_index])))
        y_hat += gamma * u.reshape((n_samples, 1))

        if remove_index > -1:
            active_index.pop(remove_index)
            '''
            residue = y - y_hat
            c = x.T.dot(residue)
            active_x = x[:, active_index] * np.sign(c[active_index]).reshape(1, len(active_index))
            g = active_x.T.dot(active_x)
            g_inv = np.linalg.pinv(g)
            e = np.ones((g.shape[0], 1))
            a = 1 / np.sqrt(e.T.dot(g_inv).dot(e))
            u = active_x.dot(g_inv).dot(e) * a
            w = a * g_inv.dot(e)
            a_vec = x.T.dot(u)
            gamma, next_index = get_gamma_next_index(active_index, c_max, c, a, a_vec)
            temp_beta[0, active_index] = (gamma * a * g_inv.dot(e) * np.sign(c[active_index])).reshape(
                len(active_index), )
            temp_beta = temp_beta + beta_hat[-1]
            beta_hat = np.row_stack((beta_hat, temp_beta))
            x_axis.append(np.sum(np.abs(beta_hat[-1, active_index])))
            '''

    #beta_hat = np.row_stack((beta_hat, beta_ols))
    #x_axis.append(np.sum(np.abs(beta_hat[-1, :])))

    axis_active_index = list(axis_active_index)
    for i in axis_active_index:
        plt.plot(x_axis, beta_hat[:, i])
        plt.plot(x_axis[i], [0], 'o', markersize=3)
        plt.text(x_axis[axis_active_index.index(i)], 0, str(i + 1))

    plt.ylabel(r'$\beta_j$')
    plt.xlabel(r'$t=\sum|\beta_j|$')
    plt.show()

    pass
