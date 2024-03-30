import numpy as np

from utils.SMW_coefficients import mapped_SMWcoef

TIR_BANDS = {
    "L4": "B6",
    "L5": "B6",
    "L7": "B6_VCID_1",
    "L8": 0,  # "B10" is first band
    "L9": "B10",
}


def get_LST(acq, landsat):
    A_img = mapped_SMWcoef(acq["TPWpos"], landsat, "A")
    B_img = mapped_SMWcoef(acq["TPWpos"], landsat, "B")
    C_img = mapped_SMWcoef(acq["TPWpos"], landsat, "C")

    return (
        A_img * acq["TOA"][..., TIR_BANDS[landsat]][..., np.newaxis] / acq["EM"]
        + B_img / acq["EM"]
        + C_img
    )
