from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy import matmul, sqrt
from numpy.linalg import multi_dot

matplotlib.use("Qt5Agg")


def read_X_y():
    X = pd.read_csv(r"d:/w/X.csv").values
    y = pd.read_csv(r"d:/w/y.csv").values
    return X, y


X, y = read_X_y()


def plsdof(X, y, m=X.shape[1],
           compute_jacobian=True,
           DoF_max=min(X.shape[1] + 1, X.shape[0] - 1),
           n_comp=3,
           scatter_y_yfit=False):
    def dA(w, A, dw):
        wa = sqrt(np.sum(w * matmul(A, w)))
        dummy = matmul((1 / wa) \
                       * (np.diag([1] * len(w))
                          - multi_dot(
                    (w[:, np.newaxis],
                     w[:, np.newaxis].transpose(),
                     A / (wa ** 2)))), dw)
        return dummy

    # v, z, dv, dz = v[:,i], b, dv[i, :, :], X.transpose()
    def dvvtz(v, z, dv, dz):
        if v.shape.__len__() < 2:
            v = v[:, np.newaxis]
            dv = dv[np.newaxis, :, :]
        k = v.shape[1]
        p = v.shape[0]
        n = dv.shape[2]
        dummy = np.zeros((dv.shape[1], dv.shape[2]))
        for i in range(k):  # ind in python, minus 1
            d = matmul((matmul(v[:, i][:, np.newaxis], z.transpose()) \
                        + np.sum(v[:, i].flatten() * z.flatten()) * np.diag([1] * p)),
                       dv[i, :, :]) \
                + multi_dot((v[:, i][:, np.newaxis], v[:, i][:, np.newaxis].transpose(), dz))
            dummy = dummy + d
        return dummy

    def vvtz(v, z):
        dummy = matmul(v, matmul(v.transpose(), z))
        dummy = dummy.flatten()
        return dummy

    p = X.shape[1]
    n = X.shape[0]
    m = min(m, DoF_max)
    Beta = np.zeros((p, m))
    V = Beta.copy()
    W = V.copy()
    dV = None
    dBeta = None
    dW = None
    if compute_jacobian:
        dW = np.zeros((m, p, n))
        dBeta = np.zeros((m, p, n))
        dV = np.zeros((m, p, n))
    X0 = X
    y0 = y
    mean_y = np.mean(y)
    y = scale(y, with_std=False)
    mean_X = np.mean(X, axis=0)
    sd_X = np.std(X, axis=0, ddof=1)  # ddof=1, so that the result is the same with R code
    sd_X[sd_X == 0] = 1
    X = X - np.outer(np.ones((1, X.shape[0])), mean_X[:, np.newaxis])
    X = X / np.outer(np.ones((1, X.shape[0])), sd_X[:, np.newaxis])
    dcoefficients = None
    A = matmul(X.transpose(), X)
    b = matmul(X.transpose(), y)
    for i in range(m):
        # python index from 0 to m-1, i is 1 smaller than R code.
        # in the numpy_array, i is good;
        # in comparison with absolute value, i should be i+1
        if (i + 1) == 1:  # python index from 0
            W[:, i] = b.flatten()
            if compute_jacobian:
                dW[i, :, :] = X.transpose()
                dW[i, :, :] = dA(W[:, i], A, dW[i, :, :])
                dV[i, :, :] = dW[i, :, :]
            W[:, i] = W[:, i] / sqrt((np.sum((W[:, i]) * matmul(A, W[:, i]))))
            V[:, i] = W[:, i]
            Beta[:, i] = np.sum(V[:, i].flatten() * b.flatten()) * V[:, i]
            if compute_jacobian:
                dBeta[i, :, :] = dvvtz(V[:, i], b, dV[i, :, :], X.transpose())
        if (i + 1) > 1:  # python index from 0
            W[:, i] = (b - matmul(A, Beta[:, i - 1][:, np.newaxis])).flatten()
            if compute_jacobian:
                dW[i, :, :] = X.transpose() - matmul(A, dBeta[i - 1, :, :])
            V[:, i] = W[:, i] - vvtz(V[:, 0:i], matmul(A, W[:, i][:, np.newaxis]))
        if compute_jacobian:
            dV[i, :, :] = dW[i, :, :] - dvvtz(V[:, 0:i],
                                              matmul(A, W[:, i][:, np.newaxis]),
                                              dV[0:i, :, :],
                                              matmul(A, dW[i, :, :]))
            # range-index in python needs to add 1 (because i has beed reduced by 1) in the ending; 'drop=false' is in default
            dV[i, :, :] = dA(V[:, i], A, dV[i, :, :])
        V[:, i] = V[:, i] / sqrt((np.sum(multi_dot((
            V[:, i][:, np.newaxis].transpose(),
            A,
            V[:, i][:, np.newaxis]
        )))))
        Beta[:, i] = Beta[:, i - 1] + np.sum(V[:, i][:, np.newaxis] * b) * V[:, i]
        if compute_jacobian:
            dBeta[i, :, :] = dBeta[i - 1, :, :] + dvvtz(V[:, i], b, dV[i, :, :], X.transpose())
    dcoefficients = None
    if compute_jacobian:
        dcoefficients = np.zeros((m + 1, p, n))
        dcoefficients[1:(m + 1), :, :] = dBeta
    sigmahat = np.zeros((m + 1))
    RSS = np.zeros((m + 1))
    yhat = np.zeros((m + 1))
    DoF = np.arange(1, m + 2).astype(float)
    Yhat = np.zeros((n, m + 1))
    dYhat = np.zeros((m + 1, n, n))
    coefficients = np.zeros((p, m + 1))
    coefficients[:, 1:(m + 1)] = Beta / matmul(sd_X[:, np.newaxis], np.ones((1, m)))
    intercept = np.array([[mean_y]] * (m + 1)) - matmul(coefficients.transpose(), mean_X[:, np.newaxis])
    covariance = None
    if compute_jacobian:
        covariance = np.zeros((m + 1, p, p))
        DD = np.diag(1 / sd_X)
    for i in range(m + 1):
        Yhat[:, i] = matmul(X0, coefficients[:, i]) + intercept[i]
        # assignment to numpy slice, right side of equation should be flattened
        res = y0 - Yhat[:, i][:, np.newaxis]
        yhat[i] = np.sum((Yhat[:, i]) ** 2)
        RSS[i] = np.sum(res ** 2)
        if compute_jacobian:
            dYhat[i, :, :] = matmul(X, dcoefficients[i, :, :]) \
                             + np.ones((n, n)) / n
            DoF[i] = np.sum(np.diag(dYhat[i, :, :]))
            dummy = matmul((np.diag([1] * n) - dYhat[i, :, :]),
                           (np.diag([1] * n) - dYhat[i, :, :].transpose()))
            # cannot check in i=0
            sigmahat[i] = sqrt(RSS[i] / np.sum(np.diag(dummy)))
            if i > (1 - 1):
                # index transfer between r and python
                covariance[i, :, :] = multi_dot((
                    sigmahat[i] ** 2 * DD,
                    dcoefficients[i, :, :],
                    dcoefficients[i, :, :].transpose(),
                    DD
                ))
    if not compute_jacobian:
        sigmahat = sqrt(RSS / (n - DoF))
    TT = matmul(X, V)
    DoF[DoF > DoF_max] = DoF_max

    if scatter_y_yfit:
        X, y = read_X_y()
        y_fit = matmul(X, coefficients[:, n_comp][:, np.newaxis]) + intercept[n_comp]
        fig, ax = plt.subplots()
        ax.scatter(y, y_fit)

    return DoF


if __name__ == '__main__':
    plsdof(X, y)
