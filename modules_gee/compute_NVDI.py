import ee


# Define a function to compute NDVI for Landsat imagery
def add_band(landsat):
    def wrap(image):
        # Choose bands based on Landsat version
        if landsat == "L8":
            nir = "SR_B5"
            red = "SR_B4"
        elif landsat == "L9":
            nir = "SR_B5"
            red = "SR_B4"
        else:
            raise ValueError("Invalid Landsat version. Valid inputs: 'L8', 'L9'")

        # Compute NDVI
        ndvi = image.expression(
            "(nir - red) / (nir + red)",
            {
                "nir": image.select(nir).multiply(0.0000275).add(-0.2),
                "red": image.select(red).multiply(0.0000275).add(-0.2),
            },
        ).rename("NDVI")

        return image.addBands(ndvi)

    return wrap
