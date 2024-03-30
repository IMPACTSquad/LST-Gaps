import os
import traceback
import logging
from glob import glob
from scenarios import Scenario, LandsatScenario
import graph_prop.complete
import HaLRTC.complete
import BandwiseMeanImpute.complete
import LSTMeanImpute.complete
from logs import log_experiment


GEO = {
    "jakarta": {"center": (-6.208588, 106.84569), "target_date": "2020-04-23"},
    "paris": {"center": (48.8588255, 2.2646355), "target_date": "2022-05-10"},
    "london": {"center": (51.5285262, -0.2664013), "target_date": "2020-09-14"},
}

METHODS = {
        # "HaLRTC": HaLRTC.complete.landsat_only,
        "GraphProp": graph_prop.complete.landsat_only,
        # "BandwiseInputMeanImputation": BandwiseMeanImpute.complete.landsat_only,
        # "LSTMeanImputation": LSTMeanImpute.complete.landsat_only,
}


def main():
    for place in GEO.keys():
        files = glob(f"tifs/{place}_L*.tif")
        # first asset is newest, last asset is oldest
        assets = sorted(
            list(set(["_".join(file.split("/")[-1].split("_")[1:4]) for file in files])),
            reverse=True,
        )
        if not len(assets) == 2:
            raise ValueError(f"Need exactly 2 assets for {place} (otherwise which asset is the reference?))")
        for method_name, method in METHODS.items():
            for missing_bin in range(10, 100, 10):
                for specific in range(10):
                    for asset, reference in zip(assets, assets[::-1]):
                        scenario = Scenario(
                            place_name=place,
                            asset_id=asset,
                            reference_asset_id=reference,
                            landsat_scenario=LandsatScenario(missing_bin, specific),
                            aster_scenario=None,  # aster fully observed
                        )
                        log_path = os.path.join(
                            "results", "landsat_only", f"{method_name}_{place}_{asset}_{missing_bin}_{specific}.log"
                        )
                        if os.path.exists(log_path):
                            continue
                        try:
                            try:
                                log_experiment(
                                    log_path=log_path,
                                    fn=method,
                                    scenario=scenario,
                                )
                            except Exception as e:
                                logging.error(f"Failed")
                                logging.error(traceback.format_exc())
                                continue
                            logging.info(f"Used method: {method_name}")
                            logging.info(f"Performed Landsat missing (ASTER observed) experiment")
                            logging.info(f"Place: {place}")
                            logging.info(f"Missing (%): {missing_bin}")
                            logging.info(f"Mask index: {specific}")
                            logging.info(f"Asset: {asset}")
                            logging.info(f"Reference asset (for graph): {reference}")
                            print()
                        except FileExistsError:
                            continue  # skip if already done


if __name__ == "__main__":
    main()

