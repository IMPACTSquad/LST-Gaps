import numpy as np
import logging

from get_inputs import sr, aster
from scenarios import aster_missing
from smw_algorithm.compute_lst import compute_LST
import diffusion
from get_inputs.utils import graph_exists, save_graph, load_graph
from scenarios import Scenario


def aster_only(scenario: Scenario):
    assert scenario.aster_missing_only

    if not graph_exists(scenario.place_name, scenario.reference_asset_id):
        logging.info(f"Creating graph for {scenario.place_name} {scenario.reference_asset_id}")
        save_graph(scenario.place_name, scenario.reference_asset_id)
    else:
        logging.info(f"Loading pre-saved graph for {scenario.place_name} {scenario.reference_asset_id}")
    adj = load_graph(scenario.place_name, scenario.reference_asset_id)

    logging.info("Completing ASTER")
    # incomplete_em = scenario.load_dynamic_emissivity()
    # is_observed = ~np.isnan(incomplete_em)
    # completed_em = diffusion.graph_prop(adj, incomplete_em, is_observed)

    # ASTER bands B13, B14, FVC
    incomplete_aster = np.stack(
        [
            aster_missing.partial_asterB13(
                scenario.place_name, scenario.aster_scenario.percent_missing, scenario.aster_scenario.specific
            ),
            aster_missing.partial_asterB14(
                scenario.place_name, scenario.aster_scenario.percent_missing, scenario.aster_scenario.specific
            ),
            aster_missing.partial_asterFVC(
                scenario.place_name, scenario.aster_scenario.percent_missing, scenario.aster_scenario.specific
            ),
        ],
        axis=-1,
    )
    completed_aster = diffusion.graph_prop(adj, incomplete_aster, scenario.aster_mask)
    dynamic_emissivity = aster.compute_dynamic_emissivity(
        landsat_fvc=scenario.load_landsat_fvc(),
        aster_b13=completed_aster[..., 0],
        aster_b14=completed_aster[..., 1],
        aster_fvc=completed_aster[..., 2],
        landsat=scenario.landsat,
    )

    return compute_LST(
        dynamic_emissivity, scenario.load_qa(), scenario.load_toa(10), scenario.load_tpw_pos(), scenario.landsat
    )


def landsat_only(scenario: Scenario):
    assert scenario.landsat_missing_only

    # reference asset is the other acquisition (i.e. fully observed landsat from another day)
    if not graph_exists(scenario.place_name, scenario.reference_asset_id):
        logging.info(f"Creating graph for {scenario.place_name} {scenario.reference_asset_id}")
        save_graph(scenario.place_name, scenario.reference_asset_id)
    else:
        logging.info(f"Loading pre-saved graph for {scenario.place_name} {scenario.reference_asset_id}")
    adj = load_graph(scenario.place_name, scenario.reference_asset_id)
    logging.info("Completing Landsat")

    # only need SR_B4, SR_B5 & TOA_B10
    incomplete_landsat = np.stack([scenario.load_sr(4), scenario.load_sr(5), scenario.load_toa(10)], axis=-1)
    completed_landsat = diffusion.graph_prop(adj, incomplete_landsat, scenario.landsat_mask)
    # [..., 0] indexes "SR_B4", [..., 1] indexes "SR_B5"
    asset_fvc = sr.compute_fvc(nir=completed_landsat[..., 1], red=completed_landsat[..., 0])
    dynamic_emissivity = scenario.load_dynamic_emissivity(asset_fvc=asset_fvc)
    qa = scenario.load_qa()

    # replace qa values using qa values from reference asset (i.e. gets rid of clouds in qa using values observed last time)
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


def landsat_and_aster(scenario: Scenario):
    assert scenario.landsat_and_aster_missing

    # reference asset is the other acquisition (i.e. fully observed landsat from another day)
    if not graph_exists(scenario.place_name, scenario.reference_asset_id):
        logging.info(f"Creating graph for {scenario.place_name} {scenario.reference_asset_id}")
        save_graph(scenario.place_name, scenario.reference_asset_id)
    else:
        logging.info(f"Loading pre-saved graph for {scenario.place_name} {scenario.reference_asset_id}")
    adj = load_graph(scenario.place_name, scenario.reference_asset_id)

    # ** STEP 1: COMPLETE LANDSAT **

    # only need SR_B4, SR_B5 & TOA_B10
    logging.info("Completing Landsat")
    incomplete_landsat = np.stack([scenario.load_sr(4), scenario.load_sr(5), scenario.load_toa(10)], axis=-1)
    completed_landsat = diffusion.graph_prop(adj, incomplete_landsat, scenario.landsat_mask)
    # [..., 0] indexes "SR_B4", [..., 1] indexes "SR_B5"
    asset_fvc = sr.compute_fvc(nir=completed_landsat[..., 1], red=completed_landsat[..., 0])

    # ** STEP 2: COMPLETE ASTER (uses asset_fvc to compute dynamic emissivity) **
    logging.info("Completing ASTER")
    incomplete_aster = np.stack(  # ASTER bands B13, B14, FVC
        [
            aster_missing.partial_asterB13(
                scenario.place_name, scenario.aster_scenario.percent_missing, scenario.aster_scenario.specific
            ),
            aster_missing.partial_asterB14(
                scenario.place_name, scenario.aster_scenario.percent_missing, scenario.aster_scenario.specific
            ),
            aster_missing.partial_asterFVC(
                scenario.place_name, scenario.aster_scenario.percent_missing, scenario.aster_scenario.specific
            ),
        ],
        axis=-1,
    )
    completed_aster = diffusion.graph_prop(adj, incomplete_aster, scenario.aster_mask)
    dynamic_emissivity = aster.compute_dynamic_emissivity(
        landsat_fvc=asset_fvc,
        aster_b13=completed_aster[..., 0],
        aster_b14=completed_aster[..., 1],
        aster_fvc=completed_aster[..., 2],
        landsat=scenario.landsat,
    )

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
