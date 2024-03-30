from get_inputs import aster, sr, toa, ncep
from utils import SMW_coefficients


def compute_GT_LST(place_name, asset_id, landsat="L8"):
    dynamic_emissivity = aster.load_dynamic_emissivity(place_name, asset_id, landsat)
    qa = sr.load_asset_qa(place_name, asset_id)
    tir = toa.load_asset_b10(place_name, asset_id)
    tpw_pos = ncep.load_asset_TPWpos(place_name, asset_id)

    return compute_LST(dynamic_emissivity, qa, tir, tpw_pos, landsat)


def compute_LST(em, qa, tir, tpw_pos, landsat="L8"):
    em[sr.is_snow(qa)] = 0.989  # prescribed snow emissivity
    em[sr.is_water_body(qa)] = 0.99  # prescribed water emissivity

    A = SMW_coefficients.mapped_SMWcoef(tpw_pos, landsat, "A")
    B = SMW_coefficients.mapped_SMWcoef(tpw_pos, landsat, "B")
    C = SMW_coefficients.mapped_SMWcoef(tpw_pos, landsat, "C")

    lst = A * tir / em + B / em + C

    return lst
