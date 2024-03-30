import rasterio
import ee


def get_two_closest(date, aoi):
    year = date.get("year")
    month = date.get("month")
    day = date.get("day")
    date1 = ee.Date.fromYMD(year, month, day)
    date2 = date1.advance(1, "days")

    # Compute the time difference from Landsat image
    datedist = lambda img: img.set(
        "DateDist",
        ee.Number(img.get("system:time_start")).subtract(date.millis()).abs(),
    )
    TPWcollection = (
        ee.ImageCollection("NCEP_RE/surface_wv")
        .filter(ee.Filter.date(date1.format("yyyy-MM-dd"), date2.format("yyyy-MM-dd")))
        .map(datedist)
    )

    # Select the two closest model times
    return TPWcollection.sort("DateDist").toList(2)


def get_TPW(date, aoi):
    closest = get_two_closest(date, aoi)
    # Check if there is atmospheric data for the wanted day
    tpw1 = ee.Image(
        ee.Algorithms.If(
            closest.size().eq(0),
            ee.Image.constant(-999.0),
            ee.Image(closest.get(0)).select("pr_wtr"),
        )
    )
    tpw2 = ee.Image(
        ee.Algorithms.If(
            closest.size().eq(0),
            ee.Image.constant(-999.0),
            ee.Algorithms.If(closest.size().eq(1), tpw1, ee.Image(closest.get(1)).select("pr_wtr")),
        )
    )

    time1 = ee.Number(ee.Algorithms.If(closest.size().eq(0), 1.0, ee.Number(tpw1.get("DateDist")).divide(21600000)))
    time2 = ee.Number(ee.Algorithms.If(closest.size().lt(2), 0.0, ee.Number(tpw2.get("DateDist")).divide(21600000)))

    tpw = tpw1.expression(
        "tpw1*time2+tpw2*time1",
        {"tpw1": tpw1, "time1": time1, "tpw2": tpw2, "time2": time2},
    )

    return tpw


def get_TPWpos(date, aoi):
    tpw = get_TPW(date, aoi)

    tpw_pos = tpw.expression(
        "value = (TPW>0 && TPW<=6) ? 0"
        + ": (TPW>6 && TPW<=12) ? 1"
        + ": (TPW>12 && TPW<=18) ? 2"
        + ": (TPW>18 && TPW<=24) ? 3"
        + ": (TPW>24 && TPW<=30) ? 4"
        + ": (TPW>30 && TPW<=36) ? 5"
        + ": (TPW>36 && TPW<=42) ? 6"
        + ": (TPW>42 && TPW<=48) ? 7"
        + ": (TPW>48 && TPW<=54) ? 8"
        + ": 9",
        {"TPW": tpw},
    )

    return tpw_pos


def load_asset_TPWpos(place_name, asset_id):
    with rasterio.open(f"tifs/{place_name}_{asset_id}_ncepTPWpos.tif", "r") as src:
        tpw_pos = src.read(1)
    return tpw_pos
