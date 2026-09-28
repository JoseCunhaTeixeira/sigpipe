"""Picking the modes of one dispersion image: M0, then each higher mode above the previous one."""

import math
from dataclasses import dataclass, replace

import numpy as np

from sigpipe.algorithms.picking.dispersion.curve import (
    lorentzian_uncertainty,
    resample_wavelength,
    shortest_picked_wavelength,
)
from sigpipe.algorithms.picking.dispersion.tracking.models import PickedMode, PickingParameters
from sigpipe.algorithms.picking.dispersion.tracking.plane_waves import (
    plane_wave_columns,
    prominences,
)
from sigpipe.algorithms.picking.dispersion.tracking.ridges import (
    corridor,
    followed_ridge,
    lowest_ridge,
    track,
)
from sigpipe.base.dispersion_curve import DispersionCurve, DispersionCurvesImage, Mode
from sigpipe.base.dispersion_image import DispersionImage


def pick_tracking(dispersion_image: DispersionImage, **parameters: object) -> DispersionImage:
    """`dispersion_image` with the curves of the modes the picker finds, PickingParameters'
    fields given as keyword arguments: the "tracking" method of the picking registry, as Pick
    runs it. A mode that already has a curve gets the new one."""
    modes = pick_modes(dispersion_image, PickingParameters.model_validate(parameters))
    picked = [mode.curve for mode in modes if mode.curve is not None]
    repicked = {curve.mode for curve in picked}
    kept = [
        curve for curve in dispersion_image.dispersion_curves or () if curve.mode not in repicked
    ]
    curves = (*kept, *picked)
    return replace(
        dispersion_image,
        dispersion_curves=DispersionCurvesImage(dispersion_curves=curves) if curves else None,
    )


def pick_modes(
    image: DispersionImage, parameters: PickingParameters | None = None
) -> list[PickedMode]:
    """The modes of `image`, from M0 up; empty when not even M0 stands above the noise floor.

    Each mode is searched above the corridor of the one below it, then its pick goes on past
    its run's ends as far as its ridge does. The search stops at the first mode with fewer than
    `min_frequencies` kept points, or whose kept points have a median coherence under
    `mode_min_ratio` times the noise floor.
    """
    parameters = parameters or PickingParameters()
    frequencies = image.fs.astype(float)
    velocities = image.vs.astype(float)
    values = image.fv_map.astype(float)
    n_v = velocities.size

    # Phase-shift images are coherences: random phases sum to about 1/sqrt(N), not to 0.
    noise_floor = 1 / math.sqrt(len(image.acquisition.receivers))
    # The search starts at min_wavelength spacings: one by default, a ridge followed into the
    # aliasing zone (under two spacings), where its points are flagged by the checks, not cut.
    shortest = shortest_picked_wavelength(image.acquisition, parameters.min_wavelength) or 0.0
    start = np.searchsorted(velocities, shortest * frequencies, side="left")
    # Above the longest wavelength the window resolves, if limited, the search stops.
    stop = np.full(frequencies.size, n_v - 1)
    if parameters.max_wavelength is not None:
        receivers = image.acquisition.receivers
        longest = parameters.max_wavelength * abs(receivers[-1].x - receivers[0].x)
        stop = np.searchsorted(velocities, longest * frequencies, side="right") - 1

    # A band cut (G3's first fix for a mode jump): the columns outside are not searched.
    outside = np.zeros(frequencies.size, dtype=bool)
    if parameters.fmin is not None:
        outside |= frequencies < parameters.fmin
    if parameters.fmax is not None:
        outside |= frequencies > parameters.fmax
    start = np.where(outside, n_v, start)

    modes: list[PickedMode] = []
    for number in range(parameters.max_modes):
        span = _longest_run(start < stop)
        if span is None or span.stop - span.start < parameters.min_frequencies:
            break

        guide = parameters.guide if number == 0 else None
        if guide:
            ridge = _guided_ridge(guide, frequencies[span], velocities, start[span], stop[span])
        else:
            ridge = lowest_ridge(values[span], start[span], stop[span], parameters.threshold)
        tracked = _tracked(image, span, ridge, start, stop, parameters, noise_floor)
        # The pick found, it goes on past its run's ends as far as its ridge does, where the
        # lowest ridge drops under it onto a dimmer sidelobe or alias: further to the low and
        # high frequencies, on the same ridge (the user, 2026-09-28).
        if not guide and tracked.kept.any():
            run = np.flatnonzero(tracked.kept)
            followed = followed_ridge(
                values[span],
                velocities,
                start[span],
                stop[span],
                parameters.threshold,
                parameters.corridor,
                (int(run[0]), int(run[-1])),
            )
            if not np.array_equal(followed, ridge):
                tracked = _tracked(image, span, followed, start, stop, parameters, noise_floor)
        if tracked.kept.sum() < parameters.min_frequencies:
            break
        if float(np.median(tracked.ratio[tracked.kept])) < parameters.mode_min_ratio:
            break

        kept = tracked.kept
        modes.append(
            PickedMode(
                number=number,
                frequencies=frequencies[span],
                velocities=velocities[tracked.path],
                coherence=tracked.coherence,
                pinned=tracked.pinned,
                kept=kept,
                noise_floor=noise_floor,
                curve=_curve(
                    image,
                    frequencies[span][kept],
                    velocities[tracked.path][kept],
                    number,
                    parameters.wavelength_step,
                ),
            )
        )

        # The next mode lies above this one's corridor, where this one was tracked.
        next_start = np.full_like(start, n_v)
        next_start[span] = tracked.high + 1
        start = next_start

    return modes


@dataclass(frozen=True, slots=True)
class _Tracked:
    """A ridge tracked through its corridor, over the columns searched."""

    high: np.ndarray  # the corridor's top: the next mode is searched above it
    path: np.ndarray  # velocity index of the pick at each frequency
    pinned: np.ndarray
    coherence: np.ndarray
    ratio: np.ndarray  # the coherence over the noise floor
    kept: np.ndarray


def _tracked(
    image: DispersionImage,
    span: slice,
    ridge: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    parameters: PickingParameters,
    noise_floor: float,
) -> _Tracked:
    """`ridge` (over the columns of `span`) fenced in by its corridor and tracked, and the points
    of the track kept."""
    frequencies = image.fs.astype(float)[span]
    velocities = image.vs.astype(float)
    values = image.fv_map.astype(float)[span]
    low, high = corridor(velocities, ridge, start[span], stop[span], parameters.corridor)
    path, on_edge = track(values, velocities, frequencies, low, high, parameters.smoothness)
    # A ridge cut by a search bound stays pinned, wherever the smoothing moved the pick.
    pinned = on_edge | (ridge == start[span]) | (ridge == stop[span])
    coherence = values[np.arange(path.size), path]
    ratio = coherence / noise_floor
    # Judged on its kept points only: pinned points are where a bound, not the data, decided,
    # and 0 Hz has no wavelength.
    kept = ~pinned & (ratio >= parameters.point_min_ratio) & (frequencies > 0)
    # Points far below the mode's typical coherence are sidelobes or noise, not its ridge.
    if kept.any():
        kept &= coherence >= parameters.min_relative_coherence * np.median(coherence[kept])
    # Where even a perfect plane wave barely varies over the grid, the window resolves no
    # velocity: the pick there is the tracker's, not the data's.
    if parameters.min_contrast is not None and kept.any():
        kept &= _resolved(
            image, frequencies, velocities[path], kept, noise_floor, parameters.min_contrast
        )
    # Where the ridge breaks, at either end, the pick stops: its longest continuous run.
    if parameters.max_gap_hz is not None:
        kept = _continuous_run(
            frequencies, velocities[path], kept, parameters.max_gap_hz, parameters.break_slope
        )
    return _Tracked(high, path, pinned, coherence, ratio, kept)


def _guided_ridge(
    guide: tuple[tuple[float, float], ...],
    frequencies: np.ndarray,
    velocities: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
) -> np.ndarray:
    """The guide's velocity at each frequency (interpolated, held flat beyond its ends), as
    indices on the velocity grid within the search bounds: where the corridor is centred."""
    points = np.array(sorted(guide), dtype=float)
    centre = np.interp(frequencies, points[:, 0], points[:, 1])
    ridge = np.searchsorted(velocities, centre, side="left")
    return np.clip(ridge, start, stop)


def _resolved(
    image: DispersionImage,
    frequencies: np.ndarray,
    velocities: np.ndarray,
    kept: np.ndarray,
    floor: float,
    contrast: float,
) -> np.ndarray:
    """Where a perfect plane wave at the pick's frequency and velocity stands at least
    `contrast` above its column's median, through the window's receivers (only the kept points
    are computed)."""
    resolved = np.zeros_like(kept)
    indices = np.flatnonzero(kept)
    columns = plane_wave_columns(image, frequencies[indices], velocities[indices], floor)
    peaks = np.minimum(np.searchsorted(image.vs, velocities[indices]), image.vs.size - 1)
    resolved[indices] = prominences(columns, peaks) >= 1 + contrast
    return resolved


def _continuous_run(
    frequencies: np.ndarray,
    velocities: np.ndarray,
    kept: np.ndarray,
    max_gap_hz: float,
    break_slope: float,
) -> np.ndarray:
    """`kept` reduced to its longest continuous run (the widest band; the lowest on a tie):
    consecutive kept points stay in one run while the columns between them that are not kept
    span at most `max_gap_hz`, and the step between them is at most `break_slope` in
    |d ln v / d ln f|. A pick goes down as far as the ridge holds."""
    indices = np.flatnonzero(kept)
    if indices.size < 2:
        return kept
    column = float(np.min(np.diff(frequencies)))
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    for before, after in zip(indices[:-1].tolist(), indices[1:].tolist(), strict=True):
        gap = frequencies[after] - frequencies[before] - column
        slope = abs(math.log(velocities[after] / velocities[before])) / math.log(
            frequencies[after] / frequencies[before]
        )
        if gap > max_gap_hz + 1e-9 or slope > break_slope:
            runs.append((start, before))
            start = after
    runs.append((start, int(indices[-1])))
    first, last = max(runs, key=lambda run: frequencies[run[1]] - frequencies[run[0]])
    within = np.zeros_like(kept)
    within[first : last + 1] = kept[first : last + 1]
    return within


def _longest_run(mask: np.ndarray) -> slice | None:
    """The longest run of consecutive True values in `mask`, as a slice."""
    best: slice | None = None
    run_start: int | None = None
    for index, value in enumerate([*mask.tolist(), False]):
        if value and run_start is None:
            run_start = index
        elif not value and run_start is not None:
            if best is None or index - run_start > best.stop - best.start:
                best = slice(run_start, index)
            run_start = None
    return best


def _curve(
    image: DispersionImage,
    frequencies: np.ndarray,
    velocities: np.ndarray,
    number: int,
    step: float,
) -> DispersionCurve | None:
    """The kept points as a sigpipe curve, like PAC's box picks: labelled M<n>, with Lorentzian
    uncertainties, resampled over wavelength every `step` metres."""
    if frequencies.size < 2:
        return None
    fs = frequencies.astype(np.float32)
    vs = velocities.astype(np.float32)
    curve = DispersionCurve(
        fs=fs,
        vs=vs,
        mode=Mode("M", number),
        acquisition=image.acquisition,
        vs_err=lorentzian_uncertainty(fs, vs, image.acquisition),
        type=image.type,
    )
    return resample_wavelength(curve, step=step)
