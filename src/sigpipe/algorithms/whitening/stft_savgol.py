import numpy as np
from scipy.signal import istft, savgol_filter, stft

from sigpipe.base.stream import Stream


def whiten_stft_savgol(
    stream: Stream,
    *,
    sfft_window_duration_sec: float,
    savgol_window_size_pts: int = 11,
    savgol_order: int = 1,
    epsilon: float = 1e-10,
) -> Stream:
    """Short-time Fourier transform-based spectral whitening using Savitzky-Golay smoothing."""
    if sfft_window_duration_sec <= 0:
        raise ValueError("sfft_window_duration_sec must be > 0")
    fft_size = int(sfft_window_duration_sec * stream.sampling_freq)
    if fft_size < 2:
        raise ValueError("FFT size too small; increase window duration or sampling_freq")
    _, _, data_fft = stft(stream.xt, nperseg=fft_size)
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
        raise ValueError("savgol_order must be strictly less than savgol_window_size_pts")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    data_fft_whitened = np.empty_like(data_fft, dtype=np.complex128)
    for t in range(data_fft.shape[2]):
        magnitude = np.abs(data_fft[:, :, t])
        magnitude_smooth = savgol_filter(
            magnitude,
            window_length=savgol_window_size_pts,
            polyorder=savgol_order,
        )
        data_fft_whitened[:, :, t] = data_fft[:, :, t] / (magnitude_smooth + epsilon)
    _, data_whitened = istft(data_fft_whitened, nperseg=fft_size)
    data_whitened = np.array(data_whitened, dtype=np.complex64).real
    return Stream(
        xt=data_whitened,
        ts=stream.ts,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
    )
