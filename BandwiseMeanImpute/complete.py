import numpy as np

from smw_algorithm.compute_lst import compute_LST
from get_inputs import sr


def landsat_only(scenario):
    assert scenario.landsat_missing_only

    # only need SR_B4, SR_B5 & TOA_B10
    incomplete_landsat = np.stack([scenario.load_sr(4), scenario.load_sr(5), scenario.load_toa(10)], axis=-1)
    means = np.nanmean(incomplete_landsat, axis=(0, 1), keepdims=True)
    completed_landsat = np.nan_to_num(incomplete_landsat, nan=means)

    # [..., 0] indexes "SR_B4", [..., 1] indexes "SR_B5"
    asset_fvc = sr.compute_fvc(nir=completed_landsat[..., 1], red=completed_landsat[..., 0])
    dynamic_emissivity = scenario.load_dynamic_emissivity(asset_fvc=asset_fvc)

    # replace qa values using qa values from reference asset (i.e. gets rid of clouds in qa using values observed last time)
    qa = scenario.load_qa()
    qa[scenario.landsat_mask == 0] = sr.load_asset_qa(scenario.place_name, scenario.reference_asset_id)[
        scenario.landsat_mask == 0
    ]

    return compute_LST(
        dynamic_emissivity,
        qa,
        completed_landsat[..., 2],  # [..., 2] indexes "TOA_B10" after completion
        scenario.load_tpw_pos(),  # no gaps
        scenario.landsat,
    )
