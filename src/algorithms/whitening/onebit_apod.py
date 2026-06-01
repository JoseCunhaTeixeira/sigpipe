import numpy as np
from scipy.fft import fft, fftfreq, ifft


def whiten_onebit_apod(
    data: np.ndarray,
    sampling_freq: float,
    *,
    fmin: float,
    fmax: float,
    taper_width_Hz: float = 1_000,
) -> np.ndarray:
    """One-bit spectral whitening with appodisation over a frequency range.

    Args:
        data (np.ndarray): raw data [ntraces, nt]
        sampling_freq (float): sampling frequency [in Hz]
        fmin (float): minumum frequency to whiten [in Hz]
        fmax (float): maximum frequency to whiten [in Hz]
        taper_width_Hz (float, optional): taper width [in Hz]. Defaults to 10.

    Returns:
        np.ndarray: whitened data [ntraces, nt]
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError("data must be a 2D numpy array: [ntraces, nt]")
    if sampling_freq <= 0:
        raise ValueError(f"sampling_freq ({sampling_freq} Hz) must be > 0 Hz")
    if not (0 <= fmin < fmax):
        raise ValueError(f"require 0 Hz <= fmin ({fmin} Hz) < fmax ({fmax} Hz)")
    if fmax - fmin <= taper_width_Hz:
        raise ValueError(
            f"taper_width_Hz ({taper_width_Hz} Hz) must be smaller than "
            f"the frequency band width ({fmax - fmin} Hz)"
        )
    _, nt = data.shape
    if nt < 2:
        raise ValueError(
            f"signal length nt ({nt} samples) too small for FFT processing"
        )
    df = sampling_freq / nt
    if int((fmax - fmin) / df) < 2:
        raise ValueError(
            f"frequency range ({int((fmax - fmin) / df)} samples) is too small, "
            f"for fmin ({fmin} Hz) and fmax ({fmax} Hz)"
        )
    fs = fftfreq(nt, d=1 / sampling_freq)
    nsmo = max(int(taper_width_Hz / df), 1)
    if nsmo > int((fmax - fmin) / df):
        raise ValueError(
            f"taper width ({nsmo} samples) is too small, "
            f"increase taper_width_Hz ({taper_width_Hz} Hz)"
        )
    absf = np.abs(fs)
    band = (absf >= fmin) & (absf <= fmax)
    taper = np.zeros(nt, dtype=np.float32)
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
    data_fft = np.array(fft(data, axis=-1), dtype=np.complex64)
    phases = np.exp(1j * np.angle(data_fft))
    data_fft_whitened = phases * taper
    return np.array(ifft(data_fft_whitened, axis=-1), dtype=np.complex64).real.astype(
        np.float32
    )
