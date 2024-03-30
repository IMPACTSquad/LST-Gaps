import numpy as np
from tqdm import tqdm

import common.tensor_utils as tu


# If tensor has n modes, introduce n aux variables: matrices M0,...,Mn-1
# Conditions
# Mi = unfold(X, i, "kolda") for each mode i
# Observed entries of tensX equal observed entries of tensT

# Inputs: Omega, tensT
# Free variables: tensX, {M0,...,Mn-1}


def complete(incomplete_tensor, mask, alphas=None, betas=None, K=500):
    if alphas is None:
        alphas = [1 / incomplete_tensor.ndim for _ in range(incomplete_tensor.ndim)]
    if betas is None:
        betas = [alpha / 10 for alpha in alphas]

    X = incomplete_tensor.copy()
    rank_prev = 0
    for k in tqdm(range(K)):
        X_fill = 0 * X
        for i in range(incomplete_tensor.ndim):
            X_unf = tu.unfold(X, i, unfolding="kolda")
            M, rank_prev = tu.SVT(X_unf, alphas[i] / betas[i], rank_prev)
            X_fill += betas[i] * tu.fold(M, i, X.shape, folding="kolda") / np.sum(betas)
        X[~(mask.astype(bool))] = X_fill[~(mask.astype(bool))]

    return X
