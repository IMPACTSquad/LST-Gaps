import os
import rasterio
import json
import scipy.sparse as sp
import numpy as np

import graphs


def is_crs_saved(place_name):
    return os.path.isfile(f"tifs/{place_name}_crs.json")


def save_crs(place_name, path):
    with rasterio.open(path, "r") as src:
        crs = str(src.crs)
        transform = src.transform
    info = {"crs": crs, "transform": transform[:6]}
    with open(f"tifs/{place_name}_crs.json", "w") as f:
        json.dump(info, f)


def load_crs(place_name):
    with open(f"tifs/{place_name}_crs.json", "r") as f:
        info = json.load(f)
    return info["crs"], info["transform"]


def use_crs_if_saved(place_name, params):
    if is_crs_saved(place_name):
        crs, transform = load_crs(place_name)
        params["crs"] = crs
        params["transform"] = transform
    return params


def graph_exists(place_name, asset_id, k=10):
    return os.path.isfile(f"tifs/{place_name}_{asset_id}_k{k}_graph.npz")


def save_graph(place_name, asset_id, k=10):
    bands = [f"srB{i}" for i in range(1, 8)] + ["srQA"] + ["toaB10", "toaB11"]
    asset_features = []
    for band in bands:
        with rasterio.open(f"tifs/{place_name}_{asset_id}_{band}.tif", "r") as src:
            asset_features.append(src.read(1))
    asset_features = np.stack(asset_features, axis=-1)
    graph = graphs.single_graph(asset_features, k=k)
    sp.save_npz(f"tifs/{place_name}_{asset_id}_k{k}_graph.npz", graph)


def load_graph(place_name, asset_id, k=10):
    return sp.load_npz(f"tifs/{place_name}_{asset_id}_k{k}_graph.npz")


if __name__ == "__main__":
    # print(load_crs("jakarta"))
    save_graph("jakarta", "LC08_122064_20200422")
