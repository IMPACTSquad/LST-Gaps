import numpy as np

from smw_algorithm.compute_lst import compute_LST
from get_inputs import sr


def landsat_only(scenario):
    assert scenario.landsat_missing_only

    # only need SR_B4, SR_B5 & TOA_B10
    incomplete_landsat = np.stack([scenario.load_sr(4), scenario.load_sr(5), scenario.load_toa(10)], axis=-1)

    # [..., 0] indexes "SR_B4", [..., 1] indexes "SR_B5"
    asset_fvc = sr.compute_fvc(nir=incomplete_landsat[..., 1], red=incomplete_landsat[..., 0])
    dynamic_emissivity = scenario.load_dynamic_emissivity(asset_fvc=asset_fvc)

    # replace qa values using qa values from reference asset (i.e. gets rid of clouds in qa using values observed last time)
    qa = scenario.load_qa()
    qa[scenario.landsat_mask == 0] = sr.load_asset_qa(scenario.place_name, scenario.reference_asset_id)[
        scenario.landsat_mask == 0
    ]

    gappy_lst = compute_LST(
        dynamic_emissivity,
        qa,
        incomplete_landsat[..., 2],  # [..., 2] indexes "TOA_B10" after completion
        scenario.load_tpw_pos(),  # no gaps
        scenario.landsat,
    )

    mean_lst = np.nanmean(gappy_lst)
    return np.nan_to_num(gappy_lst, nan=mean_lst)
