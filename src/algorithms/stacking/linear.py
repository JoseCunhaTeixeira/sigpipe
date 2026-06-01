import numpy as np


def stack_linear(
    data: np.ndarray,
) -> np.ndarray:
    """Linear stack.

    Args:
        data (np.ndarray): traces to stack [ntraces, nt]

    Returns:
        np.ndarray: resulting stacked trace [nt]
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError("data must be a 2D numpy array: [ntraces, nt]")
    return np.mean(data, axis=0).astype(np.float32)
