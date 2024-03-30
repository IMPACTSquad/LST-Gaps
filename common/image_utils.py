import numpy as np
from PIL import Image


def make_diag_mask(shape, unmasked_box_width, mask_box_width):
    """mask = 1 if observed (i.e. unmasked) 0 otherwise (masked)"""
    # if shape with more than two dimensions given, assume first two are masking dims
    if len(shape) > 2:
        shape = shape[:2]
    n = -int(np.ceil((shape[0] - unmasked_box_width) / (2 * (unmasked_box_width + mask_box_width - 1)))) * 2 * (unmasked_box_width + mask_box_width - 1)
    n -= (unmasked_box_width - 1)
    ones = np.ones(shape)
    op = 0 * ones
    while n < shape[1]:
        a = 1 - np.tril(ones, n - 1)
        b = np.tril(ones, (n + 2 * unmasked_box_width - 1) - 1)
        op += a * b
        n += (2 * unmasked_box_width - 1)

        # # if you want to have inverse output, the below = 1 if not observed
        # a = 1 - np.tril(foo, n - 1)
        # b = np.tril(foo, (n + 2 * mask_box_width - 1) - 1)
        # print(a * b)
        n += (2 * mask_box_width - 1)
    return op


def make_random_mask(shape, masked_ratio=.6):
    if len(shape) > 2:
        shape = shape[:2]
    indices = np.random.choice(np.arange(shape[0] * shape[1]),
                               int(masked_ratio * shape[0] * shape[1]),
                               replace=False)
    indices_ii, indices_jj = np.divmod(indices, shape[1])
    op = np.ones(shape)
    op[indices_ii, indices_jj] = 0.
    return op


def load_img(path):
    return np.array(Image.open(path))


def apply_mask(ip, mask):
    op = ip.copy()
    op[~(mask.astype(bool))] = 0
    return op


def dataset_split_masks(shape, train_ratio, val_ratio, slide_ratio):
    train_mask, val_mask, test_mask = np.zeros(shape, dtype="bool"), np.zeros(shape, dtype="bool"), np.zeros(shape, dtype="bool")
    slide_rows = int(shape[0] * slide_ratio)
    val_rows = np.arange(int(val_ratio * shape[0]))
    test_rows = np.arange(int(val_ratio * shape[0]), int((1 - train_ratio) * shape[0]))
    train_rows = np.arange(int((1 - train_ratio) * shape[0]), shape[0])
    for t in range(shape[-1]):
        train_mask[train_rows, :, t] = True
        val_mask[val_rows, :, t] = True
        test_mask[test_rows, :, t] = True
        train_rows = (train_rows + slide_rows) % shape[0]
        val_rows = (val_rows + slide_rows) % shape[0]
        test_rows = (test_rows + slide_rows) % shape[0]
    return train_mask, val_mask, test_mask
