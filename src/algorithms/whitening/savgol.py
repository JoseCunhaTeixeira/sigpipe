import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import savgol_filter


def whiten_savgol(
    data: np.ndarray,
    *,
    savgol_window_size_pts: int = 11,
    savgol_order: int = 1,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Spectral whitening using Savitzky-Golay smoothing.

    Args:
        data (np.ndarray): raw data [ntraces, nt]
        savgol_window_size_pts (int): savgol smoothing window size [in points]
        savgol_order (int): sagvol order
        epsilon (float): epsilon to avoid division by 0. Defaults to 1e-10

    Returns:
        np.ndarray: whitened data [ntraces, nt]
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError("data must be a 2D numpy array: [ntraces, nt]")
    _, nt = data.shape
    if nt < 3:
        raise ValueError("Signal length nt too small for FFT processing")
    if savgol_window_size_pts < 3:
        raise ValueError("savgol_window_size_pts must be >= 3")
    if savgol_window_size_pts % 2 == 0:
        raise ValueError("savgol_window_size_pts must be odd")
    if savgol_window_size_pts > nt:
        raise ValueError(
            f"savgol_window_size_pts ({savgol_window_size_pts}) "
            f"cannot exceed signal length ({nt})"
        )
    if savgol_order < 0:
        raise ValueError("savgol_order must be >= 0")
    if savgol_order >= savgol_window_size_pts:
        raise ValueError(
            "savgol_order must be strictly less than savgol_window_size_pts"
        )
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    data_fft = np.array(fft(data, axis=-1), dtype=np.complex64)
    if data_fft.size == 0:
        raise ValueError("FFT failed: empty result")
    magnitude = np.abs(data_fft)
    if np.all(magnitude == 0):
        raise ValueError("Input signal has zero energy (FFT magnitude is all zeros)")
    magnitude_smooth = np.array(
        savgol_filter(
            magnitude,
            window_length=savgol_window_size_pts,
            polyorder=savgol_order,
            axis=-1,
        ),
        dtype=np.float32,
    )
    data_whitened_fft = data_fft / (magnitude_smooth + epsilon)
    data_whitened = np.array(ifft(data_whitened_fft, axis=-1), dtype=np.complex64)
    if data_whitened.size == 0:
        raise ValueError("IFFT failed: empty result")
    return data_whitened.real.astype(np.float32)
