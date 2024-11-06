import numpy as np
import logging

from scenarios import aster_missing
from smw_algorithm.compute_lst import compute_LST
from LiuEtAl2013 import HaLRTC
from get_inputs import sr, aster
from scenarios import Scenario
from get_inputs.sr import load_srB4, load_srB5
from get_inputs.toa import load_asset_b10


def aster_only(scenario: Scenario):
    assert scenario.aster_missing_only

    # ASTER bands B13, B14, FVC
    incomplete_aster = np.stack(
        [
            aster_missing.partial_asterB13(
                scenario.place_name,
                scenario.aster_scenario.percent_missing,
                scenario.aster_scenario.specific,
            ),
            aster_missing.partial_asterB14(
                scenario.place_name,
                scenario.aster_scenario.percent_missing,
                scenario.aster_scenario.specific,
            ),
            aster_missing.partial_asterFVC(
                scenario.place_name,
                scenario.aster_scenario.percent_missing,
                scenario.aster_scenario.specific,
            ),
        ],
        axis=-1,
    )
    mean, std = np.nanmean(incomplete_aster, axis=(0, 1)), np.nanstd(
        incomplete_aster, axis=(0, 1)
    )

    completed_aster = HaLRTC.complete(incomplete_aster, scenario.aster_mask)
    completed_aster = completed_aster * std + mean
    dynamic_emissivity = aster.compute_dynamic_emissivity(
        landsat_fvc=scenario.load_landsat_fvc(),
        aster_b13=completed_aster[..., 0],
        aster_b14=completed_aster[..., 1],
        aster_fvc=completed_aster[..., 2],
        landsat=scenario.landsat,
    )

    return compute_LST(
        dynamic_emissivity,
        scenario.load_qa(),
        scenario.load_toa(10),
        scenario.load_tpw_pos(),
        scenario.landsat,
    )


def landsat_only(scenario):
    assert scenario.landsat_missing_only

    # reference asset
    asset_features = [
        load_srB4(scenario.place_name, scenario.reference_asset_id),
        load_srB5(scenario.place_name, scenario.reference_asset_id),
        load_asset_b10(scenario.place_name, scenario.reference_asset_id),
    ]
    asset_features = np.stack(asset_features, axis=-1)

    # partially-observed asset (only need SR_B4, SR_B5 & TOA_B10)
    incomplete_landsat = np.stack(
        [scenario.load_sr(4), scenario.load_sr(5), scenario.load_toa(10)], axis=-1
    )

    both = np.stack([incomplete_landsat, asset_features], axis=0)
    mean, std = np.nanmean(both, axis=(1, 2), keepdims=True), np.nanstd(
        both, axis=(1, 2), keepdims=True
    )
    both = (both - mean) / std

    completed_landsat = HaLRTC.complete(
        both,
        np.stack([scenario.landsat_mask, np.ones_like(scenario.landsat_mask)], axis=0),
    )[
        0
    ]  # 0 to get the first (incomplete) asset
    completed_landsat = completed_landsat * std[0] + mean[0]

    # [..., 0] indexes "SR_B4", [..., 1] indexes "SR_B5"
    asset_fvc = sr.compute_fvc(
        nir=completed_landsat[..., 1], red=completed_landsat[..., 0]
    )
    dynamic_emissivity = scenario.load_dynamic_emissivity(asset_fvc=asset_fvc)

    # replace qa values using qa values from reference asset (i.e. gets rid of clouds in qa using values observed last time)
    qa = scenario.load_qa()
    qa[scenario.landsat_mask == 0] = sr.load_asset_qa(
        scenario.place_name, scenario.reference_asset_id
    )[scenario.landsat_mask == 0]

    return compute_LST(
        dynamic_emissivity,
        qa,
        completed_landsat[..., 2],  # [..., 2] indexes "TOA_B10" after completion
        scenario.load_tpw_pos(),  # no gaps
        scenario.landsat,
    )


def landsat_and_aster(scenario):
    assert scenario.landsat_and_aster_missing

    # ** Complete Landsat and ASTER as one tensor completion **
    logging.info("Completing Landsat AND ASTER")
    incomplete = np.stack(
        [
            scenario.load_sr(4),
            scenario.load_sr(5),
            scenario.load_toa(10),
            aster_missing.partial_asterB13(
                scenario.place_name,
                scenario.aster_scenario.percent_missing,
                scenario.aster_scenario.specific,
            ),
            aster_missing.partial_asterB14(
                scenario.place_name,
                scenario.aster_scenario.percent_missing,
                scenario.aster_scenario.specific,
            ),
            aster_missing.partial_asterFVC(
                scenario.place_name,
                scenario.aster_scenario.percent_missing,
                scenario.aster_scenario.specific,
            ),
        ],
        axis=-1,
    )
    mean, std = np.nanmean(incomplete, axis=(0, 1)), np.nanstd(incomplete, axis=(0, 1))
    incomplete = (incomplete - mean) / std
    completed = HaLRTC.complete(incomplete, ~np.isnan(incomplete))
    completed = completed * std + mean
    landsat_fvc = sr.compute_fvc(nir=completed[..., 1], red=completed[..., 0])
    dynamic_emissivity = aster.compute_dynamic_emissivity(
        landsat_fvc=landsat_fvc,
        aster_b13=completed[..., 3],
        aster_b14=completed[..., 4],
        aster_fvc=completed[..., 5],
        landsat=scenario.landsat,
    )

    # replace qa values using qa values from reference asset (i.e. gets rid of clouds in qa using values observed last time)
    qa = scenario.load_qa()
    qa[scenario.landsat_mask == 0] = sr.load_asset_qa(
        scenario.place_name, scenario.reference_asset_id
    )[scenario.landsat_mask == 0]

    return compute_LST(
        dynamic_emissivity,
        qa,
        completed[..., 2],  # [..., 2] indexes "TOA_B10" after completion
        scenario.load_tpw_pos(),  # no gaps
        scenario.landsat,
    )
