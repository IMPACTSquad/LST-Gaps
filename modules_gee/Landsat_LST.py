import ee

import modules_gee.cloudmask as cloudmask
import modules_gee.compute_NVDI as NDVI
import modules_gee.compute_FVC as FVC
import modules_gee.NCEP_TPW as NCEP_TPW
import modules_gee.compute_emissivity as EM
import modules_gee.SMWalgorithm as LST


COLLECTION = ee.Dictionary(
    {
        "L4": {
            "TOA": ee.ImageCollection("LANDSAT/LT04/C02/T1_TOA"),
            "SR": ee.ImageCollection("LANDSAT/LT04/C02/T1_L2"),
            "TIR": [
                "B6",
            ],
            "VISW": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "QA_PIXEL"],
        },
        "L5": {
            "TOA": ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA"),
            "SR": ee.ImageCollection("LANDSAT/LT05/C02/T1_L2"),
            "TIR": [
                "B6",
            ],
            "VISW": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "QA_PIXEL"],
        },
        "L7": {
            "TOA": ee.ImageCollection("LANDSAT/LE07/C02/T1_TOA"),
            "SR": ee.ImageCollection("LANDSAT/LE07/C02/T1_L2"),
            "TIR": ["B6_VCID_1", "B6_VCID_2"],
            "VISW": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "QA_PIXEL"],
        },
        "L8": {
            "TOA": ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA"),
            "SR": ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"),
            "TIR": ["B10", "B11"],
            "VISW": [
                "SR_B1",
                "SR_B2",
                "SR_B3",
                "SR_B4",
                "SR_B5",
                "SR_B6",
                "SR_B7",
                "QA_PIXEL",
            ],
        },
        "L9": {
            "TOA": ee.ImageCollection("LANDSAT/LC09/C02/T1_TOA"),
            "SR": ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"),
            "TIR": ["B10", "B11"],
            "VISW": [
                "SR_B1",
                "SR_B2",
                "SR_B3",
                "SR_B4",
                "SR_B5",
                "SR_B6",
                "SR_B7",
                "QA_PIXEL",
            ],
        },
    }
)


# Define a function to create a Landsat collection
def get_collection(landsat, date_start, date_end, geometry, use_ndvi):
    # Load TOA Radiance/Reflectance
    collection_dict = ee.Dictionary(COLLECTION.get(landsat))

    landsatTOA = (
        ee.ImageCollection(collection_dict.get("TOA"))
        .filter(ee.Filter.date(date_start, date_end))
        .filterBounds(geometry)
    )

    # Load Surface Reflectance collection for NDVI
    landsatSR = (
        ee.ImageCollection(collection_dict.get("SR"))
        .filter(ee.Filter.date(date_start, date_end))
        .filterBounds(geometry)
        .map(cloudmask.sr)
        .map(NDVI.add_band(landsat))
        .map(FVC.add_band(landsat))
        .map(NCEP_TPW.add_band)
        .map(EM.add_band(landsat, use_ndvi))
    )

    # Combine collections
    # All channels from surface reflectance collection
    # except TIR channels: from TOA collection
    # Select TIR bands
    tir = ee.List(collection_dict.get("TIR"))
    visw = (
        ee.List(collection_dict.get("VISW"))
        .add("NDVI")
        .add("FVC")
        .add("TPW")
        .add("TPWpos")
        .add("EM")
    )

    landsatALL = landsatSR.select(visw).combine(landsatTOA.select(tir), True)

    # Compute the LST
    landsatLST = landsatALL.map(LST.add_band(landsat))

    return landsatLST
