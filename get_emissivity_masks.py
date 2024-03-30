import logging
import numpy as np
import ee
import matplotlib.pyplot as plt
from PIL import Image
import requests
import io
from glob import glob
import os

# import my_py.plotting as myplt
from get_cloud_masks import get_extent

ee.Initialize()

from modules_gee.Landsat_LST import COLLECTION


def randomly_sample():
    points = np.random.normal(0, 1, (int(1e5), 3))
    # uniform random sample of points on unit sphere
    points = points / (np.sum(points**2, axis=1, keepdims=True) ** 0.5)
    lats = 180 * np.arctan(points[:, -1] / points[:, 0]) / np.pi
    lons = 180 * np.arctan2(points[:, 0], points[:, 1]) / np.pi
    lons = lons[(lats < 66.5) & (lats > -66.5)]  # remove polar regions
    lats = lats[(lats < 66.5) & (lats > -66.5)]  # remove polar regions

    em = ee.Image("NASA/ASTER_GED/AG100_003")
    land = ee.ImageCollection("MODIS/006/MOD44W").first()
    # lats, lons = [3.163363447717939], [101.68062560223133]

    for lat, lon in zip(lats, lons):
        try:
            region = get_extent(lat, lon, 20)
            aoi = ee.Geometry.Rectangle(region)
            collection_dict = ee.Dictionary(COLLECTION.get("L8"))

            images = (
                ee.ImageCollection(collection_dict.get("SR"))
                .filterDate("2013-01-01", "2023-01-01")
                .filterBounds(aoi)
                .filter(ee.Filter.contains(".geo", aoi))  # Filter images that completely cover the AOI
            )
            if images.size().getInfo() == 0:
                logging.info("No images found")
                continue

            use_projection = images.first().projection().getInfo()
            masked_em = em.clip(ee.Geometry.Rectangle(region)).reproject(use_projection["crs"], None, 30)
            quicker_maybe_em_missing = (
                masked_em.select("emissivity_band10")
                .mask()
                .reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=30)
                .getInfo()
            )

            # download_url = masked_em.getDownloadUrl(
            #     {"bands": ["emissivity_band10"], "region": region, "scale": 30, "format": "NPY"}
            # )
            # response = requests.get(download_url)
            # em_observed = np.array(np.load(io.BytesIO(response.content)).tolist()) != -9999
            masked_land = land.clip(ee.Geometry.Rectangle(region)).reproject(use_projection["crs"], None, 30)
            quicker_maybe_is_water = (
                masked_land.select("water_mask")
                .eq(1)
                .reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=30)
                .getInfo()
            )

            if quicker_maybe_is_water["water_mask"] >= 0.15:
                logging.info(f"Too much water: {quicker_maybe_is_water['water_mask']}")
                continue  # skip if more than 15% water

            emissivity_bin = int(np.round((1 - quicker_maybe_em_missing["emissivity_band10"]) * 10, 0) * 10)
            if emissivity_bin in [0, 100]:
                continue
            if len(glob(f"emissivity_masks/{emissivity_bin-5}to{emissivity_bin+5}/*.npy")) >= 10:
                logging.info(f"Sufficient masks for bin {emissivity_bin}")
                continue  # save only 10 masks per bin

            download_url = masked_em.getDownloadUrl(
                {"bands": ["emissivity_band10"], "region": region, "scale": 30, "format": "NPY"}
            )
            response = requests.get(download_url)
            em_observed = np.array(np.load(io.BytesIO(response.content)).tolist()) != -9999

            path = os.path.join(
                "emissivity_masks",
                f"{emissivity_bin - 5}to{emissivity_bin + 5}",
                f"{'_'.join([str(f).replace('.', 'p') for f in region])}.npy",
            )
            np.save(path, em_observed)
        except ee.EEException:
            continue


def load_mask(shape, missing_emissivity=50, specific=None):
    files = sorted(glob(f"emissivity_masks/{missing_emissivity - 5}to{missing_emissivity + 5}/*.npy"))

    if specific is None:
        specific = np.random.choice(np.arange(len(files)))

    mask = np.load(files[specific]).astype(np.uint8).squeeze()
    return np.array(Image.fromarray(mask).resize(reversed(shape), Image.NEAREST))


def vis_masks():
    fig, ax = plt.subplots(9, 10, figsize=(9, 10), sharex=True, sharey=True)
    for i, missing_emissivity in enumerate(range(10, 100, 10)):
        for j in range(15):
            try:
                ax[i, j].matshow(
                    load_mask((500, 500), missing_emissivity=missing_emissivity, specific=j), vmin=0, vmax=1
                )
            except IndexError:
                continue
    # myplt.axes_off(ax)
    plt.show()


if __name__ == "__main__":
    # np.random.seed(42)
    # randomly_sample()
    vis_masks()
