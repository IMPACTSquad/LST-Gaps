import numpy as np
from tqdm import tqdm

import common.tensor_utils as tu
import common.image_utils as iu


def complete(incomplete_tensor, mask, alpha_vec=None, mu_K=None, c=0.6, K=100, L=1e-5, verbose=False):
    N = incomplete_tensor.ndim
    shape = incomplete_tensor.shape

    if alpha_vec is None:
        alpha_vec = np.ones(N) / N

    if mu_K is None:
        # from Section 7.1 p.g. 215 (value 5 chosen from range [1, 10])
        mu_K = 5 * alpha_vec / np.sqrt(shape)

    # initialize
    X = incomplete_tensor.copy()
    W = incomplete_tensor.copy()
    Z = incomplete_tensor.copy()
    B = 0
    Ldash = L

    # starting mu
    mu_0 = 0.4 * alpha_vec * np.array([tu.spectral_norm(tu.unfold(X, i, "kolda")) for i in range(N)])
    p = 1.15
    a = (mu_0 - mu_K) / (1 - K ** -p)
    b = mu_0 - a

    # step size denominator
    normX = np.linalg.norm(X)

    # for k in tqdm(range(K)):
    for k in range(K):
        mu_k = a / (k + 1) ** p + b
        while True:
            # print("f0")
            theta = L * (1 + (1 + 4 * Ldash * B) ** 0.5) / 2 / Ldash
            W = (theta / L) / (B + theta / L) * Z + B * X / (B + theta / L)
            # print("f1")
            f_W, grad_f_W = tensor_func_mu_and_grad(W, mask, alpha_vec, mu_k)  # slowest lines

            # compute f_mu(X)
            f_X = tensor_func_mu(X, alpha_vec, mu_k)  # slowest lines
            # print("f2")

            if f_X <= f_W - (grad_f_W ** 2).sum() / 2 / Ldash:
                break

            # compute f_mu(X')
            Xdash = W - grad_f_W / Ldash
            f_X = tensor_func_mu(Xdash, alpha_vec, mu_k)  # second slowest lines
            # print("f3")

            if f_X <= f_W - (grad_f_W ** 2).sum() / 2 / Ldash:
                if verbose:
                    print("Iteration: %s" % k)
                    print("Step magnitude: %s" % (np.linalg.norm(X - Xdash) / normX))
                X = Xdash.copy()
                break

            Ldash = Ldash / c

        L = Ldash
        Z -= theta * grad_f_W / L
        B += theta / L

    return X


def tensor_func_mu_and_grad(tensor, mask, alphas, mu_list):
    gradient = 0 * tensor
    f_mu = 0
    for i in range(tensor.ndim):
        tensor_unf = tu.unfold(tensor, i, "kolda")
        f_i, grad_f_i = func_mu_and_grad(tensor_unf, alphas[i], mu_list[i])
        gradient += tu.fold(grad_f_i, i, gradient.shape, "kolda")
        f_mu += f_i
    return f_mu, iu.apply_mask(gradient, ~mask.astype(bool))


def tensor_func_mu(tensor, alphas, mu_list):
    f_mu = 0
    for i in range(tensor.ndim):
        tensor_unf = tu.unfold(tensor, i, "kolda")
        f_mu += func_mu(tensor_unf, alphas[i], mu_list[i])
    return f_mu


def theta(L, Ldash, B):
    return L * (1 + (1 + 4 * Ldash * B) ** 0.5) / 2 / Ldash


def tensorW(theta, L, B, tensorZ, tensorX):
    return ((theta / L) / (B + theta / L)) * tensorZ + ((B) / (B + theta / L)) * tensorX


def func_mu_and_grad(matrix, alpha, mu):
    U, S, Vh = tu.svd(matrix)
    shrink_S = tu.shrinked_sigma(S, mu / alpha)
    trunc_S = tu.truncated_sigma(S, mu / alpha)
    f_mu = alpha ** 2 * (np.sum(S ** 2) - np.sum(shrink_S ** 2)) / 2 / mu
    grad_f_mu = np.linalg.multi_dot([U, np.diag(trunc_S), Vh]) * alpha ** 2 / mu
    return f_mu, grad_f_mu


def func_mu(matrix, alpha, mu):
    _, S, _ = tu.svd(matrix)
    shrink_S = tu.shrinked_sigma(S, mu / alpha)
    return alpha ** 2 * (np.sum(S ** 2) - np.sum(shrink_S ** 2)) / 2 / mu
