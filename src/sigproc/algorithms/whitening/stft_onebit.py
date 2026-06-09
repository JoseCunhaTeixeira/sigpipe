import numpy as np
from scipy.signal import istft, stft

from sigproc.base.stream import Stream


def whiten_stft_onebit(
    stream: Stream,
    *,
    sfft_window_duration_sec: float,
) -> Stream:
    """Short-time Fourier transform-based one bit spectral whitening."""
    if sfft_window_duration_sec <= 0:
        raise ValueError("sfft_window_duration_sec must be > 0")
    fft_size = int(sfft_window_duration_sec * stream.sampling_freq)
    if fft_size < 2:
        raise ValueError(
            f"fft_size too small ({fft_size}). "
            "Increase window duration or sampling frequency"
        )
    if fft_size > stream.nt:
        raise ValueError(
            f"fft_size ({fft_size}) is larger than signal length nt ({stream.nt})"
        )
    _, _, data_fft = stft(stream.xt, nperseg=fft_size)
    if data_fft.size == 0:
        raise ValueError(
            "STFT returned empty result. Check input signal and parameters"
        )
    data_fft_whitened = np.exp(1j * np.angle(data_fft))
    _, data_whitened = istft(data_fft_whitened, nperseg=fft_size)
    data_whitened = np.array(data_whitened, dtype=np.complex64).real
    if data_whitened is None or data_whitened.size == 0:
        raise ValueError("ISTFT failed or returned empty output")
    return Stream(
        xt=data_whitened,
        ts=stream.ts,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
    )
