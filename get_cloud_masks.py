import logging
import ee
import io
import os
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm
from glob import glob
import my_py.plotting as myplt

ee.Initialize()

from modules_gee.Landsat_LST import COLLECTION


def cloud_band_sr(image):
    # 0 if cloud, 1 if clear
    qa = image.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 3).Or(qa.bitwiseAnd(1 << 4))
    mask = mask.Not()
    return image.addBands(mask.rename("CLOUD"))


def get_cloud_masks(region, target_date="2022-05-15", plot=False):
    # Define the area of interest (AOI)
    aoi = ee.Geometry.Rectangle(region)

    collection_dict = ee.Dictionary(COLLECTION.get("L8"))

    # Filter Landsat 8 images for the target date and AOI
    images = (
        ee.ImageCollection(collection_dict.get("SR"))
        .filterDate("2013-01-01", target_date)
        .filterBounds(aoi)
        .filter(ee.Filter.contains(".geo", aoi))  # Filter images that completely cover the AOI
        .sort("system:time_start", False)  # False = descending, True = ascending
        .map(cloud_band_sr)
    )

    # Check if any images are found for the target date
    if images.size().getInfo() == 0:
        logging.info("No images found")
        return

    for i in tqdm(range(images.size().getInfo())):
        image = ee.Image(images.toList(images.size()).get(i))

        cloud_url = image.getDownloadUrl(
            {
                "bands": ["CLOUD"],
                "region": aoi,
                "scale": 30,
                "format": "NPY",
            }
        )
        response = requests.get(cloud_url)
        observed = np.array(np.load(io.BytesIO(response.content)).tolist())

        cloud_bin = int(np.round((1 - np.mean(observed)) * 10, 0) * 10)
        if cloud_bin in [0, 100]:
            continue
        if len(glob(f"cloud_masks/{cloud_bin-5}to{cloud_bin+5}/*.npy")) > 10:
            continue  # save only 10 masks per bin

        path = os.path.join(
            "cloud_masks", f"{cloud_bin - 5}to{cloud_bin + 5}", f"{image.get('system:index').getInfo()}.npy"
        )
        np.save(path, observed)

        if plot:
            url = image.getDownloadUrl(
                {
                    "bands": ["SR_B4", "SR_B3", "SR_B2"],
                    "region": aoi,
                    "scale": 30,
                    "format": "NPY",
                }
            )
            response = requests.get(url)
            rgb = np.array(np.load(io.BytesIO(response.content)).tolist())

            rgb = np.clip((rgb * 0.0000275 - 0.2) / 0.3, 0, 1)

            fig, ax = plt.subplots(1, 2, figsize=(10, 10))
            ax[0].imshow(rgb, vmin=0, vmax=0.3)
            ax[0].set_title(f"{cloud_bin}")
            ax[1].matshow(observed)
            ax[1].set_title(f"{1 - np.mean(observed)}")
            ax[0].axis("off")
            ax[1].axis("off")
            plt.show()

    return


def get_extent(lat_center, lon_center, side_length_km):
    R = 6371  # earth radius in km
    half_diagonal = side_length_km / (2**0.5)
    lat_center = lat_center * np.pi / 180
    lon_center = lon_center * np.pi / 180

    lat_max = np.arcsin(
        np.sin(lat_center) * np.cos(half_diagonal / R)
        + np.cos(lat_center) * np.sin(half_diagonal / R) * np.cos(np.pi / 4)
    )
    lat_max = 180 / np.pi * lat_max
    lon_max = lon_center + np.arctan2(
        np.sin(np.pi / 4) * np.sin(half_diagonal / R) * np.cos(lat_center),
        np.cos(half_diagonal / R) - np.sin(lat_center) * np.sin(lat_max * np.pi / 180),
    )
    lon_max = ((180 / np.pi * lon_max) + 540) % 360 - 180

    lat_min = np.arcsin(
        np.sin(lat_center) * np.cos(half_diagonal / R)
        + np.cos(lat_center) * np.sin(half_diagonal / R) * np.cos(5 * np.pi / 4)
    )
    lat_min = 180 / np.pi * lat_min
    lon_min = lon_center + np.arctan2(
        np.sin(5 * np.pi / 4) * np.sin(half_diagonal / R) * np.cos(lat_center),
        np.cos(half_diagonal / R) - np.sin(lat_center) * np.sin(lat_max * np.pi / 180),
    )
    lon_min = ((180 / np.pi * lon_min) + 540) % 360 - 180
    return [lon_min, lat_min, lon_max, lat_max]


def vis_masks():
    fig, ax = myplt.dense_fig(9, 10, figsize=(9, 10), sharex=True, sharey=True, keep_square=True)
    for i, cloud_cover in enumerate(range(10, 100, 10)):
        for j in range(10):
            ax[i, j].matshow(load_mask((500, 500), cloud_cover=cloud_cover, specific=j))
    myplt.axes_off(ax)
    plt.show()


def load_mask(shape, cloud_cover=50, specific=None):
    """mask is True/1 if observed (i.e. NOT cloudy) and False/0 if not observed (i.e. cloudy)"""
    # files = sorted(glob(f"cloud_masks/{cloud_cover - 5}to{cloud_cover + 5}/*.npy"))

    if specific is None:
        specific = np.random.choice(np.arange(10))

    # mask = np.load(files[specific]).astype(np.uint8).squeeze()
    mask = np.load(f"cloud_masks/cloudmask_{cloud_cover}_{specific}.npy").astype(np.uint8).squeeze()
    return np.array(Image.fromarray(mask).resize(reversed(shape), Image.NEAREST))


if __name__ == "__main__":
    # get_cloud_masks(get_extent(52.396, -0.497, 20), plot=False)
    vis_masks()
