import numpy as np

import get_emissivity_masks
import get_cloud_masks
from scenarios import landsat_missing
from scenarios import aster_missing
from smw_algorithm.compute_lst import compute_LST
from get_inputs import sr, aster, ncep, toa


class AsterScenario:
    def __init__(self, percent_missing, specific):
        self.percent_missing = percent_missing
        self.specific = specific

    def observed(self, shape):
        return get_emissivity_masks.load_mask(shape, self.percent_missing, self.specific)


class LandsatScenario:
    def __init__(self, percent_missing, specific):
        self.percent_missing = percent_missing
        self.specific = specific

    def observed(self, shape):
        return get_cloud_masks.load_mask(shape, self.percent_missing, self.specific)


class Scenario:
    def __init__(self, place_name, asset_id, landsat_scenario=None, aster_scenario=None, reference_asset_id=None):
        self.landsat = "L8"
        self.landsat_scenario = landsat_scenario
        self.aster_scenario = aster_scenario
        self.place_name = place_name
        self.asset_id = asset_id
        if not self.aster_missing_only:
            # if landsat or both missing, you will need a reference asset (a fully observed landsat asset to provide the graph)
            assert reference_asset_id is not None
            self.reference_asset_id = reference_asset_id
        else:
            # if aster missing only, you can use the landsat asset itself as the reference (to provide the graph)
            self.reference_asset_id = asset_id
        self.array_shape = None

    def _set_shape(self):
        self.array_shape = self.load_toa(10).shape

    @property
    def shape(self):
        if self.array_shape is None:
            self._set_shape()
        return self.array_shape

    @property
    def landsat_missing_only(self):
        if self.landsat_scenario is not None and self.aster_scenario is None:
            return True
        else:
            return False

    @property
    def aster_missing_only(self):
        if self.landsat_scenario is None and self.aster_scenario is not None:
            return True
        else:
            return False

    @property
    def landsat_and_aster_missing(self):
        if self.landsat_scenario is not None and self.aster_scenario is not None:
            return True
        else:
            return False

    @property
    def both_fully_observed(self):
        if self.landsat_scenario is None and self.aster_scenario is None:
            return True
        else:
            return False

    def load_toa(self, band, ground_truth=False):
        if self.both_fully_observed or self.aster_missing_only or ground_truth:
            function = {10: toa.load_asset_b10, 11: toa.load_asset_b11}
            return function[band](self.place_name, self.asset_id)
        else:
            function = {10: landsat_missing.partial_b10, 11: landsat_missing.partial_b11}
            return function[band](
                self.place_name,
                self.asset_id,
                self.landsat,
                self.landsat_scenario.percent_missing,
                self.landsat_scenario.specific,
            )

    @property
    def landsat_mask(self):
        # array with 1s where landsat is observed, 0s where it is missing
        if self.both_fully_observed or self.aster_missing_only:
            return np.ones_like(self.shape)  # if landsat observed it's an all ones array
        else:
            return self.landsat_scenario.observed(self.shape)

    @property
    def aster_mask(self):
        # array with 1s where aster is observed, 0s where it is missing
        if self.both_fully_observed or self.landsat_missing_only:
            return np.ones_like(self.shape)  # if landsat observed it's an all ones array
        else:
            return self.aster_scenario.observed(self.shape)

    def load_sr(self, band):
        if self.both_fully_observed or self.aster_missing_only:
            function = {i: sr.load_SR_band(i) for i in range(1, 8)}
            return function[band](self.place_name, self.asset_id)
        else:
            function = {i: landsat_missing.partial_SR_band(i) for i in range(1, 8)}
            return function[band](
                self.place_name,
                self.asset_id,
                self.landsat,
                self.landsat_scenario.percent_missing,
                self.landsat_scenario.specific,
            )

    def load_qa(self, ground_truth=False):
        if self.both_fully_observed or self.aster_missing_only or ground_truth:
            # if there are NOT landsat gaps
            return sr.load_asset_qa(self.place_name, self.asset_id)
        else:
            # if there ARE landsat gaps
            return landsat_missing.partial_qa(
                self.place_name,
                self.asset_id,
                self.landsat,
                self.landsat_scenario.percent_missing,
                self.landsat_scenario.specific,
            )

    def load_landsat_fvc(self, ground_truth=False):
        if self.both_fully_observed or self.aster_missing_only or ground_truth:
            # if there are NOT landsat gaps
            return sr.load_asset_fvc(self.place_name, self.asset_id)
        else:
            # if there ARE landsat gaps
            return landsat_missing.partial_asset_fvc(
                self.place_name,
                self.asset_id,
                self.landsat,
                self.landsat_scenario.percent_missing,
                self.landsat_scenario.specific,
            )

    def load_dynamic_emissivity(self, asset_fvc=None, ground_truth=False):
        # by default, asset_fvc not provided
        if asset_fvc is None:
            # the asset_fvc will have gaps or not depending on the scenario
            asset_fvc = self.load_landsat_fvc(ground_truth)

        if self.both_fully_observed or self.landsat_missing_only or ground_truth:
            # if there are not aster gaps
            return aster.compute_dynamic_emissivity(
                asset_fvc,
                aster.load_aster_b13(self.place_name),
                aster.load_aster_b14(self.place_name),
                aster.load_aster_fvc(self.place_name),
                self.landsat,
            )
        else:
            # if there are aster gaps
            return aster.compute_dynamic_emissivity(
                asset_fvc,
                aster_missing.partial_asterB13(
                    self.place_name, self.aster_scenario.percent_missing, self.aster_scenario.specific
                ),
                aster_missing.partial_asterB14(
                    self.place_name, self.aster_scenario.percent_missing, self.aster_scenario.specific
                ),
                aster_missing.partial_asterFVC(
                    self.place_name, self.aster_scenario.percent_missing, self.aster_scenario.specific
                ),
                self.landsat,
            )

    def load_tpw_pos(self, ground_truth=False):
        return ncep.load_asset_TPWpos(self.place_name, self.asset_id)

    def compute_LST(self):
        return compute_LST(
            self.load_dynamic_emissivity(), self.load_qa(), self.load_toa(10), self.load_tpw_pos(), self.landsat
        )

    def get_ground_truth_LST(self):
        return compute_LST(
            self.load_dynamic_emissivity(ground_truth=True),
            self.load_qa(ground_truth=True),
            self.load_toa(10, ground_truth=True),
            self.load_tpw_pos(ground_truth=True),
            self.landsat,
        )
