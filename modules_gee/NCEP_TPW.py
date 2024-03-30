import ee


# Define a function to add TPW band to Landsat imagery
def add_band(image):
    # Extract the date of the Landsat image
    date = ee.Date(image.get("system:time_start"))
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
    closest = TPWcollection.sort("DateDist").toList(2)

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
            ee.Algorithms.If(
                closest.size().eq(1), tpw1, ee.Image(closest.get(1)).select("pr_wtr")
            ),
        )
    )

    time1 = ee.Number(
        ee.Algorithms.If(
            closest.size().eq(0), 1.0, ee.Number(tpw1.get("DateDist")).divide(21600000)
        )
    )
    time2 = ee.Number(
        ee.Algorithms.If(
            closest.size().lt(2), 0.0, ee.Number(tpw2.get("DateDist")).divide(21600000)
        )
    )

    tpw = tpw1.expression(
        "tpw1*time2+tpw2*time1",
        {"tpw1": tpw1, "time1": time1, "tpw2": tpw2, "time2": time2},
    ).clip(image.geometry())

    # Define TPW bins
    tpw_bins = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 999]

    # Find the bin of each TPW value
    tpwpos = tpw.expression(
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
    tpwpos = tpwpos.clip(image.geometry())

    # Add TPW and TPWpos as bands to the image
    image_with_tpw = image.addBands([tpw.rename("TPW"), tpwpos.rename("TPWpos")])

    return image_with_tpw
