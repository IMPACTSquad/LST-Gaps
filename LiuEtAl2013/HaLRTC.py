import logging
import numpy as np
from tqdm import tqdm

import common.tensor_utils as tu


# If tensor has n modes, introduce n aux variables: matrices M0,...,Mn-1
# Conditions
# Mi = unfold(X, i, "kolda") for each mode i
# Observed entries of tensX equal observed entries of tensT

# Inputs: Omega, tensT, rho, K
# Free variables: tensX, {M0,...,Mn-1}


def complete(incomplete_tensor, mask, rho=0.0001, alphas=None, K=200, epsilon=1e-2):
    logging.info(f"Running HaLRTC on tensor with shape {incomplete_tensor.shape}")
    # initialisation
    X = incomplete_tensor.copy()
    X[mask == 0] = 0.0
    dim = incomplete_tensor.shape
    M = np.zeros(np.insert(dim, 0, len(dim)))
    Y = np.zeros(np.insert(dim, 0, len(dim)))
    order = incomplete_tensor.ndim

    if alphas is None:
        alphas = np.array([1] * order) / order  # initialise weights

    patience = 0
    last_err = None

    # iterate
    for k in tqdm(range(K)):
        X_prev = X.copy()
        for i in range(order):
            M[i] = tu.fold(tu.SVT(tu.unfold(X + Y[i] / rho, i), alphas[i] / rho)[0], i, X.shape)
        X[mask == 0] = (np.sum(M - Y / rho, axis=0) / order)[mask == 0]
        Y = Y - rho * (M - np.broadcast_to(X, np.insert(dim, 0, len(dim))))

        err = np.linalg.norm(X - X_prev) / np.linalg.norm(X_prev)
        if last_err is not None and err < last_err:
            patience += 1
        else:
            patience = 0
        last_err = err * 1.0

        if err < epsilon and patience > 10:
            logging.info(f"Converged after just {k} of maximum {K} iterations")
            break

        rho *= 1.15
    return X
