import numpy as np
from scipy.fft import fft, fftfreq, ifft

from src.sigproc.base.stream import Stream


def whiten_onebit_apod(
    stream: Stream,
    *,
    fmin: float,
    fmax: float,
    taper_width_Hz: float = 1_000,
) -> Stream:
    """One-bit spectral whitening with appodisation over a frequency range."""
    if not (0 <= fmin < fmax):
        raise ValueError(f"require 0 Hz <= fmin ({fmin} Hz) < fmax ({fmax} Hz)")
    if fmax - fmin <= taper_width_Hz:
        raise ValueError(
            f"taper_width_Hz ({taper_width_Hz} Hz) must be smaller than "
            f"the frequency band width ({fmax - fmin} Hz)"
        )
    if stream.nt < 2:
        raise ValueError(
            f"signal length nt ({stream.nt} samples) too small for FFT processing"
        )
    df = stream.sampling_freq / stream.nt
    if int((fmax - fmin) / df) < 2:
        raise ValueError(
            f"frequency range ({int((fmax - fmin) / df)} samples) is too small, "
            f"for fmin ({fmin} Hz) and fmax ({fmax} Hz)"
        )
    fs = fftfreq(stream.nt, d=1 / stream.sampling_freq)
    nsmo = max(int(taper_width_Hz / df), 1)
    if nsmo > int((fmax - fmin) / df):
        raise ValueError(
            f"taper width ({nsmo} samples) is too small, "
            f"increase taper_width_Hz ({taper_width_Hz} Hz)"
        )
    absf = np.abs(fs)
    band = (absf >= fmin) & (absf <= fmax)
    taper = np.zeros(stream.nt, dtype=np.float32)
    taper[band] = 1.0
    idx = np.argsort(absf)
    b_sorted = band[idx]
    band_idx = np.where(b_sorted)[0]
    if len(band_idx) > 2 * nsmo:
        # left edge (low freq side)
        left = band_idx[:nsmo]
        taper[idx[left]] = 0.5 * (1 - np.cos(np.linspace(0, np.pi, nsmo)))
        # right edge (high freq side)
        right = band_idx[-nsmo:]
        taper[idx[right]] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, nsmo)))
    else:
        w = len(band_idx)
        taper[idx[band_idx]] = 0.5 * (1 - np.cos(np.linspace(0, np.pi, w)))
    data_fft = np.array(fft(stream.xt, axis=-1), dtype=np.complex64)
    phases = np.exp(1j * np.angle(data_fft))
    data_fft_whitened = phases * taper
    data_whitened = np.array(ifft(data_fft_whitened, axis=-1), dtype=np.complex64).real
    if data_whitened.size == 0:
        raise ValueError("IFFT failed: empty result")
    return Stream(
        xt=data_whitened,
        ts=stream.ts,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
    )
