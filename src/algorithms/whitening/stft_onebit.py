import numpy as np
from scipy.signal import istft, stft


def whiten_stft_onebit(
    data: np.ndarray,
    *,
    sampling_freq: float,
    sfft_window_duration_sec: float,
) -> np.ndarray:
    """Short-time Fourier transform-based one bit spectral whitening.

    Args:
        data (np.ndarray): raw data [ntraces, nt]
        sampling_freq (float): sampling frequency [in Hz]
        sfft_window_duration_sec (float): short time Fourier transform window length [in s]

    Returns:
        np.ndarray: whitened data [ntraces, nt]
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError("data must be a 2D numpy array: [ntraces, nt]")
    if sampling_freq <= 0:
        raise ValueError("sampling_freq must be > 0")
    if sfft_window_duration_sec <= 0:
        raise ValueError("sfft_window_duration_sec must be > 0")
    _, nt = data.shape
    fft_size = int(sfft_window_duration_sec * sampling_freq)
    if fft_size < 2:
        raise ValueError(
            f"fft_size too small ({fft_size}). "
            "Increase window duration or sampling frequency"
        )
    if fft_size > nt:
        raise ValueError(
            f"fft_size ({fft_size}) is larger than signal length nt ({nt})"
        )
    _, _, data_fft = stft(data, nperseg=fft_size)
    if data_fft.size == 0:
        raise ValueError(
            "STFT returned empty result. Check input signal and parameters"
        )
    data_whitened_fft = np.exp(1j * np.angle(data_fft))
    _, data_whitened = istft(data_whitened_fft, nperseg=fft_size)
    if data_whitened is None or data_whitened.size == 0:
        raise ValueError("ISTFT failed or returned empty output")
    return data_whitened.real.astype(np.float32)
