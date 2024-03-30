import ee

aster = ee.Image("NASA/ASTER_GED/AG100_003")
aster_ndvi = aster.select("ndvi").multiply(0.01)
aster_fvc = aster_ndvi.expression(
    "((ndvi - ndvi_bg) / (ndvi_vg - ndvi_bg))**2",
    {"ndvi": aster_ndvi, "ndvi_bg": 0.2, "ndvi_vg": 0.86},
)
# Clip FVC values to the range [0, 1]
aster_fvc = aster_fvc.where(aster_fvc.lt(0.0), 0.0)
aster_fvc = aster_fvc.where(aster_fvc.gt(1.0), 1.0)


def emiss_bare_band(image, band):
    return image.expression(
        "(EM - 0.99 * fvc) / (1.0 - fvc)",
        {
            "EM": aster.select(f"emissivity_band{band}").multiply(0.001),
            "fvc": aster_fvc,
        },
    ).clip(image.geometry())


def emiss_bare_band10(image):
    return emiss_bare_band(image, 10)


def emiss_bare_band11(image):
    return emiss_bare_band(image, 11)


def emiss_bare_band12(image):
    return emiss_bare_band(image, 12)


def emiss_bare_band13(image):
    return emiss_bare_band(image, 13)


def emiss_bare_band14(image):
    return emiss_bare_band(image, 14)
