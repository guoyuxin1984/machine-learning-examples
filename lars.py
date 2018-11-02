"""
created on 2018-10-18
created by GUO Yuxin
"""
import numpy as np
import utils
import matplotlib.pyplot as plt


def get_gamma_next_index(used_x, c_max, c_vec, a, a_vec):
    gamma = []
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
    active_x = []
    x_axis = [0]
    beta_hat = np.zeros((n_features + 1, n_features))
    for i in range(n_features):
        residue = y - y_hat
        c = x.T.dot(residue)
        c_abs = np.abs(c)
        c_max = max(c_abs)

        for k in range(n_features):
            if np.round(c_abs[k], 4) == np.round(c_max, 4):
                if k not in active_index:
                    active_index.append(k)

        active_x = x[:, active_index] * np.sign(c[active_index]).reshape(1, len(active_index))
        if i == 0:
            g = 1
            g_inv = g
            e = 1
            a = 1
            u = x[:, active_index] * np.sign(c[active_index])
        else:
            g = active_x.T.dot(active_x)
            g_inv = np.linalg.pinv(g)
            e = np.ones((g.shape[0], 1))
            a = 1 / np.sqrt(e.T.dot(g_inv).dot(e))
            u = active_x.dot(g_inv).dot(e) * a

        a_vec = x.T.dot(u)
        gamma, next_index = get_gamma_next_index(active_index, c_max, c, a, a_vec)
        y_hat += gamma * u

        if i == 0:
            beta_hat[1, active_index] = gamma * np.sign(c[active_index])
            x_axis.append(np.abs(beta_hat[1, active_index]))
        elif i < n_features - 1:
            temp_beta = gamma * a * g_inv.dot(e)
            beta_hat[i + 1, active_index] = beta_hat[i, active_index] + (temp_beta * np.sign(c[active_index])).reshape(len(active_index),)
            x_axis.append(np.sum(np.abs(beta_hat[i + 1, active_index])))
            
    beta_hat[-1, :] = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y).reshape(len(active_index))
    x_axis.append(np.sum(np.abs(beta_hat[-1, :])))

    for i in active_index:
        plt.plot(beta_hat[:, i])
        plt.plot([i], [0], 'o', markersize=3)
        plt.text(active_index.index(i), 0, str(i + 1))
    plt.ylabel(r'$\beta_j$')
    plt.xlabel(r'$t=\sum|\beta_j|$')
    plt.show()

    pass
