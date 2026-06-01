import numpy as np


def stack_root(
    xt: np.ndarray,
    *,
    n: int = 2,
) -> np.ndarray:
    """nth-root stack according to Schimmel and Paulssen (1997).

    Args:
        xt (np.ndarray): traces to stack [ntraces, nt]
        n (int, optional): stack order (n=1 forlinear stack). Defaults to 2.

    Returns:
        np.ndarray: resulting stacked trace [nt]
    """
    if not isinstance(xt, np.ndarray) or xt.ndim != 2:
        raise TypeError("xt must be a 2D numpy array: [ntraces, nt]")
    if n < 1:
        msg = f"n {n} must be greater or equal to 1"
        raise ValueError(msg)
    r = np.mean(np.sign(xt) * np.abs(xt) ** (1 / n), axis=0)
    trace_stacked = np.sign(r) * np.abs(r) ** n
    return trace_stacked.astype(np.float32)
