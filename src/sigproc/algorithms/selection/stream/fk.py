import numpy as np
from numpy.fft import fft, fftfreq
from scipy.fft import rfft, rfftfreq
from scipy.fftpack import next_fast_len

from sigproc.algorithms.flipping.flipping import FlipAxis, flip
from sigproc.base.stream import Stream


def selection_fk(
    stream: Stream,
    threshold: float,
    vmin: float,
    vmax: float,
    flip_negatives: bool = False,
) -> Stream | None:

    if not 0 <= threshold <= 1:
        raise ValueError(f"requires 0 <= threshold <= 1, got {threshold}")
    if stream.nx < 2:
        raise ValueError(
            f"At least 2 receivers are required to define a wavenumber axis, got {stream.nx}"
        )

    # Time FFT (real signal -> non-negative frequencies only)
    nf_fft = next_fast_len(stream.nt)
    fs = rfftfreq(nf_fft, d=1.0 / stream.sampling_freq).astype(np.float32)
    xf = np.asarray(rfft(stream.xt, n=nf_fft, axis=1), dtype=np.complex64)

    # Space FFT -> f-k spectrum, shape (nk_fft, n_freq)
    dx = np.diff(stream.acquisition.offsets)
    if not np.allclose(dx, dx[0]):
        raise ValueError("Non-uniform receiver spacing")
    nk_fft = next_fast_len(stream.nx)
    ks = fftfreq(nk_fft, d=dx[0]).astype(np.float32)
    kf = np.abs(fft(xf, n=nk_fft, axis=0)).astype(np.float32)

    # Keep, per wavenumber row, only frequencies inside the velocity band:
    #   vmin * |k| <= f <= vmax * |k|
    # band_mask has shape (nk_fft, n_freq); True = inside the band (kept).
    abs_k = np.abs(ks)[:, None]  # (nk_fft, 1)
    f_row = fs[None, :]  # (1, n_freq)
    band_mask = (f_row >= vmin * abs_k) & (f_row <= vmax * abs_k)
    kf_band = kf * band_mask  # zeros everything out of band

    # Split into positive / negative wavenumbers
    energy_pos = kf_band[ks > 0].sum()
    energy_neg = kf_band[ks < 0].sum()

    eps = 1e-12
    fk_ratio = (energy_pos - energy_neg) / (energy_pos + energy_neg + eps)

    if np.abs(fk_ratio) <= threshold:
        return None

    if flip_negatives and fk_ratio > 0:
        return flip(stream=stream, axis=FlipAxis.SPACE, flip_acquisition=False)

    return stream
