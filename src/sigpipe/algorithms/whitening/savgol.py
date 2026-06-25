import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import savgol_filter

from sigpipe.base.stream import Stream


def whiten_savgol(
    stream: Stream,
    *,
    savgol_window_size_pts: int = 11,
    savgol_order: int = 1,
    epsilon: float = 1e-10,
) -> Stream:
    """Spectral whitening using Savitzky-Golay smoothing."""
    if stream.nt < 3:
        raise ValueError("Signal length nt too small for FFT processing")
    if savgol_window_size_pts < 3:
        raise ValueError("savgol_window_size_pts must be >= 3")
    if savgol_window_size_pts % 2 == 0:
        raise ValueError("savgol_window_size_pts must be odd")
    if savgol_window_size_pts > stream.nt:
        raise ValueError(
            f"savgol_window_size_pts ({savgol_window_size_pts}) "
            f"cannot exceed signal length ({stream.nt})"
        )
    if savgol_order < 0:
        raise ValueError("savgol_order must be >= 0")
    if savgol_order >= savgol_window_size_pts:
        raise ValueError("savgol_order must be strictly less than savgol_window_size_pts")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    data_fft = np.array(fft(stream.xt, axis=-1), dtype=np.complex64)
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
    data_fft_whitened = data_fft / (magnitude_smooth + epsilon)
    data_whitened = np.array(ifft(data_fft_whitened, axis=-1), dtype=np.complex64).real
    if data_whitened.size == 0:
        raise ValueError("IFFT failed: empty result")
    return Stream(
        xt=data_whitened,
        ts=stream.ts,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
    )
