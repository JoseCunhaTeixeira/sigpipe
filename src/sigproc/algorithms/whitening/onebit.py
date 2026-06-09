import numpy as np
from scipy.fft import fft, ifft

from sigproc.base.stream import Stream


def whiten_onebit(
    stream: Stream,
) -> Stream:
    """One-bit spectral whitening."""
    if stream.nt < 2:
        raise ValueError("Signal length nt too small for FFT processing")
    data_fft = np.array(fft(stream.xt, axis=-1), dtype=np.complex64)
    if data_fft.size == 0:
        raise ValueError("FFT failed: empty result")
    data_fft_whitened = np.exp(1j * np.angle(data_fft))
    data_whitened = np.array(ifft(data_fft_whitened, axis=-1), dtype=np.complex64).real
    if data_whitened.size == 0:
        raise ValueError("IFFT failed: empty result")
    return Stream(
        xt=data_whitened,
        ts=stream.ts,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
    )
