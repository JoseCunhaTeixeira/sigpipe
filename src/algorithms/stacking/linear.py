import numpy as np


def stack_linear(
    xt: np.ndarray,
) -> np.ndarray:
    """Linear stack.

    Args:
        xt (np.ndarray): traces to stack [ntraces, nt]

    Returns:
        np.ndarray: resulting stacked trace [nt]
    """
    if not isinstance(xt, np.ndarray) or xt.ndim != 2:
        raise TypeError("xt must be a 2D numpy array: [ntraces, nt]")
    return np.mean(xt, axis=0).astype(np.float32)
