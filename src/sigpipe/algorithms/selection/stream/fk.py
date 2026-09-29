import numpy as np
from numpy.fft import fft, fftfreq
from scipy.fft import rfft, rfftfreq
from scipy.fftpack import next_fast_len

from sigpipe.algorithms.flipping.flipping import FlipAxis, flip
from sigpipe.base.stream import Stream


def fk_ratio(stream: Stream, vmin: float | None = None, vmax: float | None = None) -> float:
    """How lopsided `stream`'s f-k energy is between positive and negative wavenumbers, in the
    velocity band between `vmin` and `vmax` (each when given): from -1 (all negative) to 1 (all
    positive), 0 when balanced. What `selection_fk` keeps a segment by."""
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

    # Keep, per wavenumber row, only frequencies inside the velocity band, each bound when given:
    #   vmin * |k| <= f <= vmax * |k|
    # band_mask has shape (nk_fft, n_freq); True = inside the band (kept).
    abs_k = np.abs(ks)[:, None]  # (nk_fft, 1)
    f_row = fs[None, :]  # (1, n_freq)
    band_mask = np.ones((abs_k.shape[0], f_row.shape[1]), dtype=bool)
    if vmin is not None:
        band_mask &= f_row >= vmin * abs_k
    if vmax is not None:
        band_mask &= f_row <= vmax * abs_k
    kf_band = kf * band_mask  # zeros everything out of band

    # Split into positive / negative wavenumbers
    energy_pos = kf_band[ks > 0].sum()
    energy_neg = kf_band[ks < 0].sum()

    eps = 1e-12
    return float((energy_pos - energy_neg) / (energy_pos + energy_neg + eps))


def selection_fk(
    stream: Stream,
    threshold: float,
    vmin: float | None = None,
    vmax: float | None = None,
    flip_negatives: bool = False,
) -> Stream | None:
    """Keep `stream` only if its in-band f-k energy is asymmetric enough
    between positive/negative wavenumbers (|fk_ratio| > threshold). The band
    is between `vmin` and `vmax`, each when given.

    If `flip_negatives` is set, the stream is space-flipped when
    `fk_ratio > 0`, i.e. when positive-wavenumber energy dominates -- despite
    the parameter name, a stream dominated by negative wavenumbers is
    returned unflipped.
    """

    return select_by_ratio(stream, fk_ratio(stream, vmin, vmax), threshold, flip_negatives)


def select_by_ratio(
    stream: Stream, ratio: float, threshold: float, flip_negatives: bool = False
) -> Stream | None:
    """`selection_fk`'s decision on `stream`, its fk_ratio `ratio` measured: kept when
    |ratio| > `threshold`, space-flipped when `flip_negatives` and `ratio` > 0."""
    if not 0 <= threshold <= 1:
        raise ValueError(f"requires 0 <= threshold <= 1, got {threshold}")

    if np.abs(ratio) <= threshold:
        return None

    if flip_negatives and ratio > 0:
        return flip(stream=stream, axis=FlipAxis.SPACE, flip_acquisition=False)

    return stream
