import numpy as np


def masked_rmse(uncentered, mu, mask, name):
    def rmse(truth, prediction):
        return np.mean(((uncentered - (prediction + mu))[mask]) ** 2) ** 0.5
    rmse.__name__ = name
    return rmse


def masked_mae(uncentered, mu, mask, name):
    def mae(truth, prediction):
        return np.mean(np.abs((uncentered - (prediction + mu))[mask]))
    mae.__name__ = name
    return mae
