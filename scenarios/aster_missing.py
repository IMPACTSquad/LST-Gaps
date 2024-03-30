import numpy as np
import matplotlib.pyplot as plt

from get_inputs.aster import load_dynamic_emissivity
from get_inputs import sr, toa, ncep, aster
from smw_algorithm import compute_lst
import get_emissivity_masks


def aster_missing_only(place_name, asset_id, landsat="L8"):
    landsat_fvc = sr.load_asset_fvc(place_name, asset_id)  # fully observed i.e. no clouds
    return aster.compute_dynamic_emissivity(
        landsat_fvc, partial_asterB13(place_name), partial_asterB14(place_name), partial_asterFVC(place_name), landsat
    )


def partial_asterB13(place_name, missing_emissivity, specific):
    b13 = aster.load_aster_b13(place_name)
    emissivity_observed = get_emissivity_masks.load_mask(b13.shape, missing_emissivity, specific)
    b13[emissivity_observed == 0] = np.nan
    return b13


def partial_asterB14(place_name, missing_emissivity, specific):
    b14 = aster.load_aster_b14(place_name)
    emissivity_observed = get_emissivity_masks.load_mask(b14.shape, missing_emissivity, specific)
    b14[emissivity_observed == 0] = np.nan
    return b14


def partial_asterFVC(place_name, missing_emissivity, specific):
    fvc = aster.load_aster_fvc(place_name)
    emissivity_observed = get_emissivity_masks.load_mask(fvc.shape, missing_emissivity, specific)
    fvc[emissivity_observed == 0] = np.nan
    return fvc
