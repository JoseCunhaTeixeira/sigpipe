import numpy as np


def array_repr(array: np.ndarray | None) -> str:
    """Compact repr for an ndarray field: shape and dtype instead of full contents."""
    if array is None:
        return "None"
    return f"array(shape={array.shape}, dtype={array.dtype})"
