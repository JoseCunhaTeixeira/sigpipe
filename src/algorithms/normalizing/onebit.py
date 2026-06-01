import numpy as np


def normalize_onebit(xt: np.ndarray) -> np.ndarray:
    return np.sign(xt).astype(np.float32)
