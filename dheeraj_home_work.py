import numpy as np
import cvxpy as cp
import scipy.misc
import matplotlib.pyplot as plt


def loss_fn(X, beta):
    return 0.5 * cp.norm(X - beta, "fro") ** 2


def regularizer(beta):
    rows, cols = beta.shape
    sum_across_rows = cp.tv(beta[:, 0])
    for i in range(1, cols):
        sum_across_rows = sum_across_rows + cp.tv(beta[:, i])
    sum_across_cols = cp.tv(beta[0, :])
    for i in range(1, rows):
        sum_across_cols = sum_across_cols + cp.tv(beta[i, :])
    return sum_across_rows + sum_across_cols


def objective_fn(X, beta, lambd):
    return loss_fn(X, beta) + lambd * regularizer(beta)


def process(X, lambda_val):
    # X = np.genfromtxt('HW1_data/toy.csv', delimiter=',')
    nrow, ncol = X.shape
    beta = cp.Variable(X.shape)
    lambd = cp.Parameter(nonneg=True)
    lambd.value = lambda_val
    problem = cp.Problem(cp.Minimize(objective_fn(X, beta, lambd)))
    f = problem.solve()
    print(f)
    out = beta.value
    return out  # plt.imsave('toy_out.png',out)


if __name__ == '__main__':
    import cv2
    noisy_im = cv2.imread('noisy_k0_im.png', cv2.IMREAD_GRAYSCALE)
    assert noisy_im.shape[0] == noisy_im.shape[1]

    X = noisy_im

    res = process(X, 50)

    res_im = res.astype(np.uint8)
    cv2.imwrite('res_k0_im___.png', res_im)