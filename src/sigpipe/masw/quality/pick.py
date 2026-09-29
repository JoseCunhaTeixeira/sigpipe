"""Measures of a dispersion image's M0 pick (PACo's curve QC, G3, judges them), each a median
over the pick's kept points:

- sharpness: peak width at half its height above the noise floor, over the same width for a
  perfect plane wave at that frequency and velocity, through the window's receivers and on the
  same grid. A propagating wave cannot be much narrower: well below 1, the peak is a fringe or
  an edge, not a ridge.
- prominence: peak height above the floor over the column's median height above the floor,
  over the same ratio for the perfect plane wave: 1 is as prominent as the window allows.
- on_data: share of points on their column's brightest value (within 10 %), so the data and not
  the tracker's smoothing drew the curve.
- constant_wavelength: share of points where velocity grows like frequency, the signature of the
  edge of what the window resolves.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict

from sigpipe.algorithms.picking.dispersion.tracking import PickedMode
from sigpipe.algorithms.picking.dispersion.tracking.plane_waves import (
    plane_wave_columns,
    prominences,
)
from sigpipe.base.dispersion_image import DispersionImage

# A point is on the data when its velocity is within this fraction of its column's brightest one.
_ON_DATA_TOLERANCE = 0.1
# Velocity grows like frequency when the local log-log slope of the pick is above this.
_CONSTANT_WAVELENGTH_SLOPE = 0.7
# The local slope is fitted over this many neighbouring points.
_SLOPE_POINTS = 7
# Prominence is counted up to this: beyond, any ridge stands out enough. A long array's perfect
# plane wave has its sidelobes under the noise floor and would rise far above it, making any
# real image look weak.
_PROMINENT_ENOUGH = 4.0


class PickMeasures(BaseModel):
    """The measures of one image's M0 pick; with no ridge (no pick, or fewer than 2 kept
    points), only how many points it kept."""

    model_config = ConfigDict(frozen=True)

    n_points: int
    band_hz: tuple[float, float] | None = None
    sharpness: float | None = None
    prominence: float | None = None
    on_data: float | None = None
    constant_wavelength: float | None = None


def measure_pick(image: DispersionImage, m0: PickedMode | None) -> PickMeasures:
    """The measures of `image`'s M0 pick `m0`, on its kept points."""
    if m0 is None:
        return PickMeasures(n_points=0)

    # Frequency 0 has no wavelength: it cannot be measured.
    kept = m0.kept & (m0.frequencies > 0)
    frequencies, velocities = m0.frequencies[kept], m0.velocities[kept]
    if frequencies.size < 2:
        return PickMeasures(n_points=int(frequencies.size))

    grid_v = image.vs.astype(float)
    rows = _nearest_rows(np.asarray(image.fs, dtype=float), frequencies)
    peaks = np.searchsorted(grid_v, velocities)
    above = np.clip(image.fv_map[rows].astype(float) - m0.noise_floor, 0, None)
    perfect = plane_wave_columns(image, frequencies, velocities, m0.noise_floor)
    brightest = grid_v[np.argmax(above, axis=1)]
    return PickMeasures(
        n_points=int(frequencies.size),
        band_hz=(float(frequencies.min()), float(frequencies.max())),
        sharpness=_sharpness(above, perfect, peaks, grid_v),
        prominence=_prominence(above, perfect, peaks),
        on_data=float(np.mean(np.abs(brightest - velocities) / velocities < _ON_DATA_TOLERANCE)),
        constant_wavelength=_constant_wavelength(frequencies, velocities),
    )


def _nearest_rows(fs: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """The image's row nearest each frequency: a pick's lie on its rows, a saved curve's,
    resampled over wavelength, between them."""
    above = np.clip(np.searchsorted(fs, frequencies), 1, fs.size - 1)
    below = above - 1
    return np.where(frequencies - fs[below] < fs[above] - frequencies, below, above)


def _half_width(column: np.ndarray, peak: int, grid_v: np.ndarray) -> float:
    """The width of the peak at `peak`, at half its height."""
    level = column[peak] / 2
    low = high = peak
    while low > 0 and column[low - 1] >= level:
        low -= 1
    while high < column.size - 1 and column[high + 1] >= level:
        high += 1
    return float(grid_v[high] - grid_v[low])


def _sharpness(
    above: np.ndarray, perfect: np.ndarray, peaks: np.ndarray, grid_v: np.ndarray
) -> float:
    """Median peak width at half its height above the floor, over a perfect plane wave's."""
    ratios = [
        _half_width(above[k], int(peak), grid_v) / width
        for k, peak in enumerate(peaks)
        if (width := _half_width(perfect[k], int(peak), grid_v)) > 0
    ]
    return float(np.median(ratios)) if ratios else 1.0


def _prominence(above: np.ndarray, perfect: np.ndarray, peaks: np.ndarray) -> float:
    """Median prominence (peak height above the floor over its column's median height above
    the floor, counted up to _PROMINENT_ENOUGH), over a perfect plane wave's."""
    measured = np.minimum(prominences(above, peaks), _PROMINENT_ENOUGH)
    reference = np.minimum(prominences(perfect, peaks), _PROMINENT_ENOUGH)
    return float(np.median(measured / np.maximum(reference, 1e-3)))


def _constant_wavelength(frequencies: np.ndarray, velocities: np.ndarray) -> float:
    """Share of points where the local log-log slope of the pick is close to 1."""
    return float(np.mean(local_slopes(frequencies, velocities) > _CONSTANT_WAVELENGTH_SLOPE))


def constant_wavelength_start(frequencies: np.ndarray, velocities: np.ndarray) -> float | None:
    """The shortest wavelength (m) of the stretch at the pick's long-wavelength end where
    velocity grows like frequency (the edge of what the window resolves): where to cut it.
    None when the longest wavelengths do not follow it."""
    wavelengths = velocities / frequencies
    along = local_slopes(frequencies, velocities) > _CONSTANT_WAVELENGTH_SLOPE
    start: float | None = None
    for index in np.argsort(wavelengths)[::-1]:
        if not along[index]:
            break
        start = float(wavelengths[index])
    return start


def local_slopes(frequencies: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """The pick's local log-log slope at each point, fitted over its neighbours."""
    log_f, log_v = np.log(frequencies), np.log(velocities)
    half = _SLOPE_POINTS // 2
    return np.array(
        [
            np.polyfit(
                log_f[max(0, k - half) : k + half + 1], log_v[max(0, k - half) : k + half + 1], 1
            )[0]
            for k in range(frequencies.size)
        ]
    )
