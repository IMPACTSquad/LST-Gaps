import matplotlib.pyplot as plt

from smw_algorithm.compute_lst import compute_LST
from get_inputs import aster, ncep
from aster_missing import partial_asterB13, partial_asterB14, partial_asterFVC
from landsat_missing import partial_asset_fvc, partial_qa, partial_b10


def get_dynamic_emissivity(
    place_name, asset_id, landsat_missing, aster_missing, landsat_specific, aster_specific, landsat="L8"
):
    landsat_fvc = partial_asset_fvc(place_name, asset_id, landsat, landsat_missing, landsat_specific)
    return aster.compute_dynamic_emissivity(
        landsat_fvc,
        partial_asterB13(place_name, aster_missing, aster_specific),
        partial_asterB14(place_name, aster_missing, aster_specific),
        partial_asterFVC(place_name, aster_missing, aster_specific),
        landsat,
    )


def demo_landsat_and_aster_missing():
    place_name = "jakarta"
    asset_id = "LC08_122064_20200422"
    landsat_missing = 10
    aster_missing = 10
    landsat_specific = 0
    aster_specific = 0

    tir = partial_b10(place_name, asset_id, "L8", landsat_missing, landsat_specific)
    qa = partial_qa(place_name, asset_id, "L8", landsat_missing, landsat_specific)
    em = get_dynamic_emissivity(place_name, asset_id, landsat_missing, aster_missing, landsat_specific, aster_specific)

    # fully observed
    tpw_pos = ncep.load_asset_TPWpos(place_name, asset_id)

    lst = compute_LST(em, qa, tir, tpw_pos, "L8")

    fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)
    ax[0].matshow(tir)
    ax[0].set_title("Landsat gaps")
    ax[1].matshow(em)
    ax[1].set_title("EM gaps")
    ax[2].matshow(lst)
    ax[2].set_title("LST output")
    plt.show()
