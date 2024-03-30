import os
from utils import metrics
import datetime
import numpy as np
import logging
import matplotlib.pyplot as plt

from scenarios import Scenario


def log_experiment(log_path, fn, scenario: Scenario):
    if not log_path.endswith(".log"):
        raise ValueError(f"Log path {log_path} must end with .log")
    if os.path.isfile(log_path):
        raise FileExistsError(f"Log file {log_path} already exists")

    # Clear all handlers
    log = logging.getLogger()
    for handler in log.handlers[:]:
        log.removeHandler(handler)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        filename=log_path,
        filemode="a",
        format="%(asctime)s.%(msecs)03d >> %(levelname)s >> %(message)s",
    )
    logging.Formatter.formatTime = (
        lambda self, record, datefmt=None: datetime.datetime.fromtimestamp(record.created, datetime.timezone.utc)
        .astimezone()
        .isoformat(sep="T", timespec="milliseconds")
    )
    # see output in console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)

    # Run experiment
    with open(log_path, "r") as f:
        logging.info(f"STARTING STOPWATCH")
        lst = fn(scenario)
        logging.info(f"STOPPING STOPWATCH")
    lst_w_gaps = scenario.compute_LST()
    lst_ground_truth = scenario.get_ground_truth_LST()
    # fig, ax = plt.subplots(1, 3, figsize=(20, 5), sharex=True, sharey=True)
    # ax[0].imshow(lst)
    # ax[0].set_title("LST")
    # ax[1].imshow(lst_w_gaps)
    # ax[1].set_title("LST with gaps")
    # ax[2].imshow(lst_ground_truth)
    # ax[2].set_title("Ground truth LST")
    # plt.show()
    # exit()
    logging.info(f"RMSE: {metrics.rmse(lst_w_gaps, lst_ground_truth, lst)}")
    logging.info(f"MSE: {metrics.mse(lst_w_gaps, lst_ground_truth, lst)}")
    logging.info(f"MAE: {metrics.mae(lst_w_gaps, lst_ground_truth, lst)}")
    logging.info(f"PSNR: {metrics.psnr(lst_w_gaps, lst_ground_truth, lst)}")
    logging.info(f"Missing {np.sum(np.isnan(lst_w_gaps))} out of {np.sum(~np.isnan(lst))} LST pixels")
    if scenario.aster_missing_only or scenario.landsat_and_aster_missing:
        logging.info(f"Missing {np.sum(scenario.aster_mask == 0)} out of {np.prod(scenario.shape)} ASTER pixels")
    if scenario.landsat_missing_only or scenario.landsat_and_aster_missing:
        logging.info(f"Missing {np.sum(scenario.landsat_mask == 0)} out of {np.prod(scenario.shape)} Landsat pixels")
    np.save(log_path.replace(".log", "_lst_output.npy"), lst.astype(np.float32))
