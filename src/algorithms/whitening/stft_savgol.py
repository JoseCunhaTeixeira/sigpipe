import numpy as np
from scipy.signal import istft, savgol_filter, stft


def whiten_stft_savgol(
    data: np.ndarray,
    *,
    sampling_freq: float,
    sfft_window_duration_sec: float,
    savgol_window_size_pts: int = 11,
    savgol_order: int = 1,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Short-time Fourier transform-based spectral whitening using Savitzky-Golay smoothing.

    Args:
        data (np.ndarray): raw data [ntraces, nt]
        sampling_freq (float): sampling frequency [in Hz]
        sfft_window_duration_sec (float): short time Fourier transform window length [in s]
        savgol_window_size_pts (int): savgol smoothing window size [in points]
        savgol_order (int): sagvol order
        epsilon (float): epsilon to avoid division by 0. Defaults to 1e-10

    Returns:
        np.ndarray: whitened data [ntraces, nt]
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError("data must be a 2D numpy array: [ntraces, nt]")
    if sampling_freq <= 0:
        raise ValueError("sampling_freq must be > 0")
    if sfft_window_duration_sec <= 0:
        raise ValueError("sfft_window_duration_sec must be > 0")
    fft_size = int(sfft_window_duration_sec * sampling_freq)
    if fft_size < 2:
        raise ValueError(
            "FFT size too small; increase window duration or sampling_freq"
        )
    _, _, data_fft = stft(data, nperseg=fft_size)
    _, n_freqs, _ = data_fft.shape
    if savgol_window_size_pts < 3:
        raise ValueError("savgol_window_size_pts must be >= 3")
    if savgol_window_size_pts % 2 == 0:
        raise ValueError("savgol_window_size_pts must be odd")
    if savgol_window_size_pts > n_freqs:
        raise ValueError(
            f"savgol_window_size_pts ({savgol_window_size_pts}) "
            f"cannot exceed number of frequency bins ({n_freqs})"
        )
    if savgol_order < 0:
        raise ValueError("savgol_order must be >= 0")
    if savgol_order >= savgol_window_size_pts:
        raise ValueError(
            "savgol_order must be strictly less than savgol_window_size_pts"
        )
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    data_whitened_fft = np.empty_like(data_fft, dtype=np.complex128)
    for t in range(data_fft.shape[2]):
        magnitude = np.abs(data_fft[:, :, t])
        magnitude_smooth = savgol_filter(
            magnitude,
            window_length=savgol_window_size_pts,
            polyorder=savgol_order,
        )
        data_whitened_fft[:, :, t] = data_fft[:, :, t] / (magnitude_smooth + epsilon)
    _, data_whitened = istft(data_whitened_fft, nperseg=fft_size)
    return data_whitened.real.astype(np.float32)
