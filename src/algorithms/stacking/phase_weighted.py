import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import hilbert


def stack_phase_weighted(
    data: np.ndarray,
    *,
    nu: int = 2,
) -> np.ndarray:
    """Phase-weighted stack according to Schimmel and Paulssen (1997).

    Args:
        data (np.ndarray): traces to stack [ntraces, nt]
        nu (int): stack order (nu=0 for linear stack), defaults to 2.

    Returns:
        np.ndarray: resulting stacked trace [nt]
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError("data must be a 2D numpy array: [ntraces, nt]")
    if nu < 0:
        msg = f"nu {nu} must be greater or equal to 0"
        raise ValueError(msg)
    npts = np.shape(data)[1]
    nfft = next_fast_len(npts)
    anal_sig = np.array(hilbert(data, N=nfft, axis=-1), np.complex64)[:, :npts]
    instant_phase = anal_sig / (np.abs(anal_sig) + 1e-12)
    phase_stack = np.abs(np.mean(instant_phase, axis=0)) ** nu
    trace_stacked = np.mean(data, axis=0) * phase_stack
    return trace_stacked.astype(np.float32)
