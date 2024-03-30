import rasterio


def get_B10(image, aoi):
    return image.select("B10")


def get_B11(image, aoi):
    return image.select("B11")


def partial_asset_fvc(place_name, asset_id, landsat, missing_landsat, specific):
    b10 = landsat_observed(place_name, asset_id)
    landsat_observed = get_cloud_masks.load_mask(srB4.shape, missing_landsat, specific)
    srB4[landsat_observed == 0] = np.nan
    srB5[landsat_observed == 0] = np.nan
    return sr.compute_fvc


def load_asset_b10(place_name, asset_id):
    with rasterio.open(f"tifs/{place_name}_{asset_id}_toaB10.tif", "r") as src:
        b10 = src.read(1)
    return b10


def load_asset_b11(place_name, asset_id):
    with rasterio.open(f"tifs/{place_name}_{asset_id}_toaB11.tif", "r") as src:
        b11 = src.read(1)
    return b11
