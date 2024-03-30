import ee
import rasterio
import matplotlib.pyplot as plt

import get_inputs.sr as sr


def get_c13(landsat="L8"):
    if landsat == "L4":
        return 0.3222
    elif landsat == "L5":
        return -0.0723
    elif landsat == "L7":
        return 0.2147
    else:
        return 0.6820


def get_c14(landsat="L8"):
    if landsat == "L4":
        return 0.6498
    elif landsat == "L5":
        return 1.0521
    elif landsat == "L7":
        return 0.7789
    else:
        return 0.2578


def get_c(landsat="L8"):
    if landsat == "L4":
        return 0.0272
    elif landsat == "L5":
        return 0.0195
    elif landsat == "L7":
        return 0.0059
    else:
        return 0.0584


def get_EM0(aoi, landsat="L8"):
    c13 = ee.Number(get_c13(landsat))
    c14 = ee.Number(get_c14(landsat))
    c = ee.Number(get_c(landsat))
    aster = ee.Image("NASA/ASTER_GED/AG100_003")
    EM0 = aster.expression(
        "c13 * EM13 + c14 * EM14 + c",
        {
            "EM13": get_band13(aoi, landsat),
            "EM14": get_band14(aoi, landsat),
            "c13": ee.Image(c13),
            "c14": ee.Image(c14),
            "c": ee.Image(c),
        },
    )

    return EM0


def get_band13(aoi, landsat):
    return ee.Image("NASA/ASTER_GED/AG100_003").select("emissivity_band13").multiply(0.001)


def get_band14(aoi, landsat):
    return ee.Image("NASA/ASTER_GED/AG100_003").select("emissivity_band14").multiply(0.001)


def get_bandNDVI(aoi, landsat):
    return ee.Image("NASA/ASTER_GED/AG100_003").select("ndvi").multiply(0.01)


def get_bandFVC(aoi, landsat):
    return get_bandNDVI(aoi, landsat).expression(
        "((ndvi-ndvi_bg)/(ndvi_vg - ndvi_bg))**2",
        {
            "ndvi": get_bandNDVI(aoi, landsat),
            "ndvi_bg": 0.2,
            "ndvi_vg": 0.86,
        },
    )


def load_static_emissivity(place_name):
    return rasterio.open(f"tifs/{place_name}_asterEM0.tif").read(1)


def load_aster_fvc(place_name):
    return rasterio.open(f"tifs/{place_name}_asterFVC.tif").read(1)


def load_aster_b13(place_name):
    return rasterio.open(f"tifs/{place_name}_asterB13.tif").read(1)


def load_aster_b14(place_name):
    return rasterio.open(f"tifs/{place_name}_asterB14.tif").read(1)


def load_dynamic_emissivity(place_name, asset_id, landsat="L8"):
    aster_fvc = load_aster_fvc(place_name)
    aster_b13 = load_aster_b13(place_name)
    aster_b14 = load_aster_b14(place_name)
    asset_fvc = sr.load_asset_fvc(place_name, asset_id)
    return compute_dynamic_emissivity(asset_fvc, aster_b13, aster_b14, aster_fvc, landsat)


def compute_dynamic_emissivity(landsat_fvc, aster_b13, aster_b14, aster_fvc, landsat="L8"):
    emiss_bare = (get_c13(landsat) * (aster_b13 - 0.99 * aster_fvc)) / (1 - aster_fvc)
    emiss_bare += (get_c14(landsat) * (aster_b14 - 0.99 * aster_fvc)) / (1 - aster_fvc)
    emiss_bare += get_c(landsat)

    return landsat_fvc * 0.99 + (1 - landsat_fvc) * emiss_bare


if __name__ == "__main__":
    plt.matshow(load_dynamic_emissivity("jakarta", "LC08_122064_20230906"))
    plt.show()
