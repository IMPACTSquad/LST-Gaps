import numpy as np


def rmse(lst_sim, lst_gt, lst_completed):
    if not lst_sim.ndim == lst_gt.ndim == lst_completed.ndim == 2:
        raise ValueError("LST arguments must be 2D LST arrays")
    contributing = np.isnan(lst_sim)
    truth, predicted = lst_gt[contributing], lst_completed[contributing]
    return np.mean((truth - predicted) ** 2) ** 0.5


def mse(lst_sim, lst_gt, lst_completed):
    if not lst_sim.ndim == lst_gt.ndim == lst_completed.ndim == 2:
        raise ValueError("LST arguments must be 2D LST arrays")
    contributing = np.isnan(lst_sim)
    truth, predicted = lst_gt[contributing], lst_completed[contributing]
    return np.mean((truth - predicted) ** 2)


def mae(lst_sim, lst_gt, lst_completed):
    if not lst_sim.ndim == lst_gt.ndim == lst_completed.ndim == 2:
        raise ValueError("LST arguments must be 2D LST arrays")
    contributing = np.isnan(lst_sim)
    truth, predicted = lst_gt[contributing], lst_completed[contributing]
    return np.mean(np.abs(truth - predicted))


def psnr(lst_sim, lst_gt, lst_completed):
    if not lst_sim.ndim == lst_gt.ndim == lst_completed.ndim == 2:
        raise ValueError("LST arguments must be 2D LST arrays")
    contributing = np.isnan(lst_sim)
    maximum = max(np.nanmax(lst_gt), np.nanmax(lst_completed))
    truth, predicted = lst_gt[contributing], lst_completed[contributing]
    return 20 * np.log10(maximum) - 10 * np.log10(np.mean((truth - predicted) ** 2))
