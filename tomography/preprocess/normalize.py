import numpy as np


def robust_normalize(image, percentile=99.9):
    """
    HAADF-safe normalization.
    """
    scale = np.percentile(image, percentile)
    if scale <= 0:
        raise ValueError("Invalid normalization scale")
    return image / scale
