# Define the cloudmask functions for TOA data
def toa(image):
    qa = image.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 3).eq(0)
    return image.updateMask(mask.Not())


# Define the cloudmask functions for SR data
def sr(image):
    qa = image.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 3).Or(qa.bitwiseAnd(1 << 4))
    return image.updateMask(mask.Not())
