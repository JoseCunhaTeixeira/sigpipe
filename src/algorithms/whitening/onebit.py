import numpy as np
from scipy.fft import fft, ifft


def whiten_onebit(
    data: np.ndarray,
) -> np.ndarray:
    """One-bit spectral whitening.

    Args:
        data (np.ndarray): raw data [ntraces, nt]

    Returns:
        np.ndarray: whitened data [ntraces, nt]
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError("data must be a 2D numpy array: [ntraces, nt]")
    _, nt = data.shape
    if nt < 2:
        raise ValueError("Signal length nt too small for FFT processing")
    data_fft = np.array(fft(data, axis=-1), dtype=np.complex64)
    if data_fft.size == 0:
        raise ValueError("FFT failed: empty result")
    data_whitened_fft = np.exp(1j * np.angle(data_fft))
    data_whitened = np.array(ifft(data_whitened_fft, axis=-1), dtype=np.complex64)
    if data_whitened.size == 0:
        raise ValueError("IFFT failed: empty result")
    return data_whitened.real.astype(np.float32)
