"""Measures of a dispersion image before picking (PACo's image QC, G2, judges them): the noise
floor, the coherent columns, peaks on the grid's edges, competing ridges, and aliased ones."""

import math

import numpy as np
from scipy.signal import find_peaks

from sigpipe.algorithms.picking.dispersion.curve import (
    max_resolvable_wavelength,
    min_resolvable_wavelength,
)
from sigpipe.base.dispersion_image import DispersionImage


def noise_floor(image: DispersionImage) -> float:
    """Random phases sum to about 1 / sqrt(N) on a phase-shift image (the picker's floor)."""
    return 1 / math.sqrt(len(image.acquisition.receivers))


def coherent_columns(image: DispersionImage, level: float) -> np.ndarray:
    """The frequencies (rows of fv_map) whose peak lies `level` of the way from the noise floor
    to 1."""
    floor = noise_floor(image)
    return image.fv_map.max(axis=1) > floor + level * (1 - floor)


def edge_peaks(
    image: DispersionImage, coherent: np.ndarray, edge_share: float, floor: float = 0.0
) -> tuple[int, int]:
    """How many coherent columns peak within `edge_share` of the grid's lowest velocity, and,
    looking above `floor` only (below it lie artifacts), within `edge_share` of its highest,
    among the columns where the window tells that velocity from an infinite one: a window of
    aperture L resolves slowness to about 1 / (f L), so below f = vmax / L a peak at vmax is
    energy with no moveout (noise common to every trace), not a ridge beyond the grid."""
    vs = np.asarray(image.vs, dtype=float)
    fs = np.asarray(image.fs, dtype=float)
    span = vs.max() - vs.min()
    peaks = vs[image.fv_map[coherent].argmax(axis=1)] if coherent.any() else np.array([])
    above = _above(image, floor)[coherent]
    peaks_above = vs[above.argmax(axis=1)] if coherent.any() else np.array([])
    aperture = max_resolvable_wavelength(image.acquisition) or 0.0
    resolved = (fs * aperture > vs.max())[coherent]
    low = int(np.sum(peaks <= vs.min() + edge_share * span))
    high = int(np.sum((peaks_above >= vs.max() - edge_share * span) & resolved))
    return low, high


def competing_ridges(
    image: DispersionImage,
    coherent: np.ndarray,
    ratio: float,
    separation: float,
    floor: float = 0.0,
) -> np.ndarray:
    """For each coherent column, whether a second local maximum reaches `ratio` of the highest
    while lying at least `separation` (relative) away from it in velocity; peaks below `floor`
    are artifacts, not ridges."""
    vs = np.asarray(image.vs, dtype=float)
    above = _above(image, floor)
    result = np.zeros(image.fv_map.shape[0], dtype=bool)
    for row in np.flatnonzero(coherent):
        column = above[row]
        best = int(column.argmax())
        peaks, _ = find_peaks(column, height=ratio * column[best])
        others = [peak for peak in peaks if abs(vs[peak] - vs[best]) >= separation * vs[best]]
        result[row] = bool(others)
    return result


def aliased(image: DispersionImage, coherent: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Coherent columns whose peak (above `floor`) lies below the aliasing limit v = 2 dx f (the
    shortest wavelength the receivers resolve, times f): a second ridge there is the alias of
    the first, not a mode. None of them when the receivers' positions are unknown."""
    shortest = min_resolvable_wavelength(image.acquisition)
    if shortest is None:
        return np.zeros_like(coherent)
    vs = np.asarray(image.vs, dtype=float)
    fs = np.asarray(image.fs, dtype=float)
    peaks = vs[_above(image, floor).argmax(axis=1)]
    return coherent & (peaks < shortest * fs)


def _above(image: DispersionImage, floor: float) -> np.ndarray:
    """The image with the velocities below `floor` zeroed: where a correlation's zero lag or a
    mute's edge puts energy no surface wave has."""
    vs = np.asarray(image.vs, dtype=float)
    return np.where(vs[None, :] >= floor, image.fv_map, 0.0)
