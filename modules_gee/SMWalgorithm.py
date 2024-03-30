import ee

# coefficients for the Statistical Mono-Window Algorithm
import modules_gee.SMW_coefficients as SMWcoef


# Function to create a lookup between two columns in a feature collection
def get_lookup_table(fc, prop_1, prop_2):
    reducer = ee.Reducer.toList().repeat(2)
    lookup = fc.reduceColumns(reducer, [prop_1, prop_2])
    return ee.List(lookup.get("list"))


def add_band(landsat):
    def wrap(image):
        # Select algorithm coefficients
        coeff_SMW = ee.FeatureCollection(
            ee.Algorithms.If(
                landsat == "L4",
                SMWcoef.coeff_SMW_L4,
                ee.Algorithms.If(
                    landsat == "L5",
                    SMWcoef.coeff_SMW_L5,
                    ee.Algorithms.If(
                        landsat == "L7",
                        SMWcoef.coeff_SMW_L7,
                        ee.Algorithms.If(
                            landsat == "L8", SMWcoef.coeff_SMW_L8, SMWcoef.coeff_SMW_L9
                        ),
                    ),
                ),
            )
        )

        # Create lookups for the algorithm coefficients
        A_lookup = get_lookup_table(coeff_SMW, "TPWpos", "A")
        B_lookup = get_lookup_table(coeff_SMW, "TPWpos", "B")
        C_lookup = get_lookup_table(coeff_SMW, "TPWpos", "C")

        # Map coefficients to the image using the TPW bin position
        A_img = image.remap(A_lookup.get(0), A_lookup.get(1), 0.0, "TPWpos").resample(
            "bilinear"
        )
        B_img = image.remap(B_lookup.get(0), B_lookup.get(1), 0.0, "TPWpos").resample(
            "bilinear"
        )
        C_img = image.remap(C_lookup.get(0), C_lookup.get(1), 0.0, "TPWpos").resample(
            "bilinear"
        )

        # select TIR band
        tir = (
            "B10"
            if landsat == "L9"
            else (
                "B10" if landsat == "L8" else ("B6_VCID_1" if landsat == "L7" else "B6")
            )
        )

        # compute the LST
        lst = image.expression(
            "A * Tb1 / em1 + B / em1 + C",
            {
                "A": A_img,
                "B": B_img,
                "C": C_img,
                "em1": image.select("EM"),
                "Tb1": image.select(tir),
            },
        ).updateMask(image.select("TPW").lt(0).Not())

        return image.addBands(lst.rename("LST"))

    return wrap
