import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_SR4(image, aoi):
    return get_SR_band(4)(image, aoi)


def get_SR_band(band):
    return lambda image, aoi: image.select(f"SR_B{band}")


def get_SR5(image, aoi):
    return get_SR_band(5)(image, aoi)


def get_QA(image, aoi):
    return image.select("QA_PIXEL")


def load_asset_qa(place_name, asset_id):
    with rasterio.open(f"tifs/{place_name}_{asset_id}_srQA.tif", "r") as src:
        qa = src.read(1)
    return qa


def is_snow(qa):
    # bit 5 is snow (https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2#bands)
    return np.bitwise_and(qa, 1 << 5) != 0


def is_cloud(qa):
    # bit 3 is cloud, bit 4 is cloud shadow (https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2#bands)
    return np.logical_or(np.bitwise_and(qa, 1 << 3) != 0, np.bitwise_and(qa, 1 << 4) != 0)


def is_water_body(qa):
    # bit 7 is water (https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2#bands)
    return np.bitwise_and(qa, 1 << 7) != 0


def apply_cloud(qa, is_cloud_mask):
    # where is_cloud mask says there should be cloud, if qa indicates water, we lose this information
    qa[np.logical_and(is_cloud_mask, is_water_body(qa))] ^= 1 << 7
    # where is_cloud mask says there should be cloud, if qa indicates snow, we lose this information
    qa[np.logical_and(is_cloud_mask, is_snow(qa))] ^= 1 << 5
    # where is_cloud mask says there should be cloud, if not already indicated as so in qa, we add it
    qa[np.logical_and(is_cloud_mask, ~is_cloud(qa))] ^= 1 << 3
    return qa


def load_srB4(place_name, asset_id):
    # Landsat 8 red uses SR_B4
    return load_SR_band(4)(place_name, asset_id)


def load_srB5(place_name, asset_id):
    # Landsat 8 NIR uses SR_B5
    return load_SR_band(5)(place_name, asset_id)


def load_SR_band(band):
    def wrap(place_name, asset_id):
        with rasterio.open(f"tifs/{place_name}_{asset_id}_srB{band}.tif", "r") as src:
            return src.read(1) * 0.0000275 - 0.2

    return wrap


def load_asset_fvc(place_name, asset_id):
    nir, red = load_srB5(place_name, asset_id), load_srB4(place_name, asset_id)
    return compute_fvc(nir, red)


def compute_fvc(nir, red):
    ndvi = (nir - red) / (nir + red)

    fvc = ((ndvi - 0.2) / (0.86 - 0.2)) ** 2
    fvc = np.clip(fvc, 0, 1)

    return fvc


if __name__ == "__main__":
    qa = load_asset_qa("jakarta", "LC08_122064_20200422")
    simulated_cloud = np.load("cloud_masks/45to55/LC08_199026_20160508.npy").astype(np.uint8).squeeze()
    simulated_cloud = np.array(Image.fromarray(simulated_cloud).resize(reversed(qa.shape), Image.NEAREST))
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
    ax[0, 0].matshow(is_cloud(qa), vmin=0, vmax=1)
    ax[0, 1].matshow(is_snow(qa), vmin=0, vmax=1)
    ax[0, 2].matshow(is_water_body(qa), vmin=0, vmax=1)
    qa = apply_cloud(qa, simulated_cloud)
    ax[1, 0].matshow(is_cloud(qa), vmin=0, vmax=1)
    ax[1, 1].matshow(is_snow(qa), vmin=0, vmax=1)
    ax[1, 2].matshow(is_water_body(qa), vmin=0, vmax=1)
    plt.show()
