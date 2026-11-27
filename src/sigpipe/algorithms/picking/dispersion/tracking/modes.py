"""Picking the modes of one dispersion image: M0, then each higher mode above the previous one."""

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import median_filter

from sigpipe.algorithms.picking.dispersion.curve import (
    brightest_peaks,
    clean_picks,
    longest_reached_wavelength,
    lorentzian_uncertainty,
    min_resolvable_wavelength,
    resample_wavelength,
    shortest_picked_wavelength,
)
from sigpipe.algorithms.picking.dispersion.lasso import lasso_cells, lasso_curve, lasso_picks
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
    its run's ends as far as its ridge does, its kept points picked as a lasso drawn along its
    corridor picks them. The search stops at the first mode with fewer than `min_frequencies`
    kept points, or whose kept points have a median coherence under `mode_min_ratio` times the
    noise floor. With a lasso, M0 is picked inside it as PAC's lasso picks (none when it holds
    fewer than 2 peaks), each higher mode searched above it.
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
    # A guide steers within its own band: beyond its ends it says nothing (the neighbours give no
    # curve there), and a corridor held on its last velocity would follow the noise.
    if parameters.guide:
        guided = [f for f, _ in parameters.guide]
        outside |= (frequencies < min(guided)) | (frequencies > max(guided))
    start = np.where(outside, n_v, start)

    modes: list[PickedMode] = []
    for number in range(parameters.max_modes):
        if number == 0 and parameters.lasso is not None:
            lassoed = _lassoed(image, parameters, start, stop, noise_floor)
            if lassoed is None:
                break
            mode, start = lassoed
            modes.append(mode)
            continue

        span = _longest_run(start < stop)
        if span is None or span.stop - span.start < parameters.min_frequencies:
            break

        guide = parameters.guide if number == 0 else None
        if guide:
            ridge = _guided_ridge(guide, frequencies[span], velocities, start[span], stop[span])
            tracked = _mode(
                _tracked(image, span, ridge, start, stop, parameters, noise_floor), parameters
            )
        else:
            # The lowest maxima that reach the noise floor, as the points are judged: random
            # phases sum to about it, so a maximum under it is noise, where a column holds a
            # stronger one.
            floor = parameters.point_min_ratio * noise_floor
            tracked = _searched(image, span, start, stop, parameters, noise_floor, floor)
            # No mode along the lowest maxima, or one mostly off its columns' brightest values:
            # searched again, the maxima fainter than a mode skipped where a column holds a
            # stronger one (noise or an alias under the ridge, not the fundamental mode).
            on_data = _on_data(values[span], velocities, tracked)
            if on_data < parameters.min_on_data:
                floor = parameters.mode_min_ratio * noise_floor
                above = _searched(image, span, start, stop, parameters, noise_floor, floor)
                if _on_data(values[span], velocities, above) > on_data:
                    tracked = above
        if tracked is None:
            break

        kept = tracked.kept
        # What the search tracked, the kept points as a lasso along its ridge picks them.
        reported = velocities[tracked.path]
        reported[kept] = _picked(image, span, tracked)
        step = np.gradient(velocities)
        modes.append(
            PickedMode(
                number=number,
                frequencies=frequencies[span],
                velocities=reported,
                coherence=tracked.coherence,
                pinned=tracked.pinned,
                kept=kept,
                noise_floor=noise_floor,
                curve=_curve(
                    image,
                    frequencies[span][kept],
                    reported[kept],
                    number,
                    parameters.wavelength_step,
                ),
                low=velocities[tracked.low] - step[tracked.low] / 2,
                high=velocities[tracked.high] + step[tracked.high] / 2,
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
    low: np.ndarray  # the corridor's bottom
    path: np.ndarray  # velocity index of the pick at each frequency
    pinned: np.ndarray
    coherence: np.ndarray
    ratio: np.ndarray  # the coherence over the noise floor
    kept: np.ndarray


def _searched(
    image: DispersionImage,
    span: slice,
    start: np.ndarray,
    stop: np.ndarray,
    parameters: PickingParameters,
    noise_floor: float,
    floor: float,
) -> _Tracked | None:
    """The mode along the lowest ridge of `span`'s columns (their maxima under `floor` skipped
    where one reaches it), or None. The pick found, it goes on past its run's ends as far as its
    ridge does, where the lowest ridge drops under it onto a dimmer sidelobe or alias: further to
    the low and high frequencies, on the same ridge."""
    values = image.fv_map.astype(float)[span]
    velocities = image.vs.astype(float)
    ridge = lowest_ridge(values, start[span], stop[span], parameters.threshold, floor)
    tracked = _tracked(image, span, ridge, start, stop, parameters, noise_floor)
    if tracked.kept.any():
        run = np.flatnonzero(tracked.kept)
        followed = followed_ridge(
            values,
            velocities,
            start[span],
            stop[span],
            parameters.threshold,
            parameters.corridor,
            (int(run[0]), int(run[-1])),
            floor,
        )
        if not np.array_equal(followed, ridge):
            tracked = _tracked(image, span, followed, start, stop, parameters, noise_floor)
    return _mode(tracked, parameters)


def _mode(tracked: _Tracked, parameters: PickingParameters) -> _Tracked | None:
    """`tracked` when it is a mode: enough kept points, their median coherence at the level a
    mode must reach."""
    if tracked.kept.sum() < parameters.min_frequencies:
        return None
    if float(np.median(tracked.ratio[tracked.kept])) < parameters.mode_min_ratio:
        return None
    return tracked


def _on_data(values: np.ndarray, velocities: np.ndarray, tracked: _Tracked | None) -> float:
    """The share of `tracked`'s kept points within 10 % of their column's brightest velocity (as
    the curve's checks count it), over the whole column: a pick held where the search starts, on
    the flank of an alias under it, lies off it; 0 without a mode."""
    if tracked is None:
        return 0.0
    rows = np.flatnonzero(tracked.kept)
    brightest = velocities[np.argmax(values[rows], axis=1)]
    picked = velocities[tracked.path[rows]]
    return float(np.mean(np.abs(brightest - picked) / picked < 0.1))


def _tracked(
    image: DispersionImage,
    span: slice,
    ridge: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    parameters: PickingParameters,
    noise_floor: float,
    bridged: np.ndarray | None = None,
) -> _Tracked:
    """`ridge` (over the columns of `span`) fenced in by its corridor and tracked, and the points
    of the track kept. Where the ridge leaves its level for a faster ridge and comes back to it
    (_faster_ridge: a higher mode brighter there, M0 a shoulder under it), it is tracked again
    along its level there (_levelled), the columns `bridged` holding no point of the mode's: left
    out, the run going on across them."""
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
    if bridged is None:
        strength = values[np.arange(ridge.size), ridge]
        bridged = _faster_ridge(
            frequencies, velocities[ridge], strength, kept, parameters.min_frequencies
        )
        if bridged.any():
            levelled = _levelled(frequencies, velocities, ridge, bridged)
            return _tracked(image, span, levelled, start, stop, parameters, noise_floor, bridged)
    kept &= ~bridged
    # Where the ridge breaks, at either end, the pick stops: its longest continuous run.
    if parameters.max_gap_hz is not None:
        kept = _continuous_run(
            frequencies,
            velocities[path],
            kept,
            parameters.max_gap_hz,
            parameters.break_slope,
            parameters.reach_slope,
            (
                min_resolvable_wavelength(image.acquisition),
                longest_reached_wavelength(image.acquisition),
            ),
            parameters.min_frequencies,
            _slower_below(values, velocities, path, parameters.point_min_ratio * noise_floor),
            bridged,
        )
    # Where it slows down toward its lowest frequencies, at wavelengths longer than the window,
    # its low end is dropped.
    if parameters.max_low_fall is not None:
        receivers = image.acquisition.receivers
        length = abs(receivers[-1].x - receivers[0].x)
        kept = _low_end(frequencies, velocities[path], kept, parameters.max_low_fall, length)
    return _Tracked(high, low, path, pinned, coherence, ratio, kept)


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
    reach_slope: float,
    resolved: tuple[float | None, float | None],
    min_points: int,
    slower: np.ndarray,
    bridged: np.ndarray | None = None,
) -> np.ndarray:
    """`kept` reduced to its longest continuous run (the widest in octaves; the lowest on a tie):
    consecutive kept points stay in one run while the columns between them that are not kept
    span at most `max_gap_hz` (a faster ridge's columns, `bridged`, not counted: the ridge left
    its level there and came back, _faster_ridge), and the step between them is at most
    `break_slope` in
    |d ln v / d ln f|, `reach_slope` where either lies beyond the window's reach (`resolved`:
    the shortest and longest wavelengths the window resolves, m; None: unknown, `break_slope`
    throughout). A pick goes down as far as the ridge holds: a ridge turning fast steps less
    than a jump onto another ridge, while beyond the reach a pick climbing steeply is on the
    smear of a ridge the window no longer resolves.

    A run starting more than HIGHER_RISE faster than the end of a run of the mode's below it (at
    least `min_points`, its wavelengths mostly within what the window resolves, its velocity
    falling with frequency), over a slower ridge in most of its columns (`slower`: M0, fainter
    there), is a higher mode's, brighter there: M0 is the slowest mode. A run
    rising with frequency as a constant wavelength does, or slower than any surface wave, is an
    artefact. Where another run is left, neither is kept; of the others with at least
    `min_points`, the widest in octaves within what the window resolves (the wavelengths it
    informs: a band in Hz favours the highest frequencies, the shortest wavelengths), or the
    lowest of those nearly as wide."""
    indices = np.flatnonzero(kept)
    if indices.size < 2:
        return kept
    shortest, longest = resolved
    column = float(np.min(np.diff(frequencies)))
    wavelengths = velocities / np.where(frequencies > 0, frequencies, np.inf)
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    for before, after in zip(indices[:-1].tolist(), indices[1:].tolist(), strict=True):
        gap = frequencies[after] - frequencies[before] - column
        if bridged is not None:
            gap -= column * int(bridged[before + 1 : after].sum())
        slope = abs(math.log(velocities[after] / velocities[before])) / math.log(
            frequencies[after] / frequencies[before]
        )
        beyond = longest is not None and max(wavelengths[before], wavelengths[after]) > longest
        if gap > max_gap_hz + 1e-9 or slope > (reach_slope if beyond else break_slope):
            runs.append((start, before))
            start = after
    runs.append((start, int(indices[-1])))

    def substantial(run: tuple[int, int]) -> bool:
        """A run of the mode's: enough points, mostly within what the window resolves, its
        velocity not rising with frequency as a constant wavelength does (the edge of what the
        window resolves, or an artefact near the grid's floor)."""
        within = kept[run[0] : run[1] + 1]
        if int(within.sum()) < min_points:
            return False
        lengths = wavelengths[run[0] : run[1] + 1][within]
        resolved_here = (lengths >= (shortest or 0.0)) & (lengths <= (longest or np.inf))
        log_f = np.log(frequencies[run[0] : run[1] + 1][within])
        trend = float(np.polyfit(log_f, np.log(velocities[run[0] : run[1] + 1][within]), 1)[0])
        return float(np.mean(resolved_here)) > 0.5 and trend < MODE_RISE

    def trend(run: tuple[int, int]) -> float:
        """The run's log-log slope of velocity against frequency; 0 under 3 kept points."""
        within = kept[run[0] : run[1] + 1]
        if int(within.sum()) < 3:
            return 0.0
        log_f = np.log(frequencies[run[0] : run[1] + 1][within])
        return float(np.polyfit(log_f, np.log(velocities[run[0] : run[1] + 1][within]), 1)[0])

    candidates: list[tuple[int, int]] = []
    below: tuple[int, int] | None = None  # the last run of the mode's, at lower frequencies
    for run in runs:
        higher = (
            below is not None
            and velocities[run[0]] > (1 + HIGHER_RISE) * velocities[below[1]]
            and float(np.mean(slower[run[0] : run[1] + 1][kept[run[0] : run[1] + 1]])) > 0.5
        )
        # Rising with frequency as a constant wavelength does, or slower than any surface wave:
        # an artefact, no mode's.
        within = kept[run[0] : run[1] + 1]
        slow = float(np.median(velocities[run[0] : run[1] + 1][within])) < ARTEFACT_SPEED
        if not higher and not slow and trend(run) < CONSTANT_RISE:
            candidates.append(run)
            if substantial(run):
                below = run
    # A run of fewer than `min_points` is no mode's, as a pick of fewer is none.
    long_enough = [run for run in candidates if int(kept[run[0] : run[1] + 1].sum()) >= min_points]
    chosen = long_enough or candidates or runs

    def octaves(run: tuple[int, int], reach: bool) -> float:
        """The run's width in octaves (log of its highest frequency over its lowest: the
        wavelengths it spans at a steady velocity, without the fall of a smear's), over its kept
        points within what the window resolves if `reach`; 0 under 2 such points."""
        within = kept[run[0] : run[1] + 1]
        band = frequencies[run[0] : run[1] + 1][within]
        if reach:
            lengths = wavelengths[run[0] : run[1] + 1][within]
            band = band[(lengths >= (shortest or 0.0)) & (lengths <= (longest or np.inf))]
        return math.log(band.max() / band.min()) if band.size >= 2 else 0.0

    widths = [octaves(run, reach=True) for run in chosen]
    if max(widths) <= 0.0:  # none within what the window resolves: their whole bands
        widths = [octaves(run, reach=False) for run in chosen]
    # The widest; the lowest of those nearly as wide: its longer wavelengths inform more ground.
    first, last = next(
        run for run, width in zip(chosen, widths, strict=True) if width >= TIE * max(widths)
    )
    within = np.zeros_like(kept)
    within[first : last + 1] = kept[first : last + 1]
    return within


# How much faster than the run of the mode's below it a run starts, at most, before it is taken
# for a higher mode's: G3's largest step between neighbouring points.
HIGHER_RISE = 0.3
# A run whose velocity rises with frequency at this log-log slope or more is no mode's: a mode
# slows down with frequency, or rises mildly over a bump.
MODE_RISE = 0.5
# A run rising with frequency at this log-log slope or more, nearly a constant wavelength (1), is
# an artefact (an alias at one receiver spacing, a correlation's zero lag), never the pick.
CONSTANT_RISE = 0.8
# Runs at least this share as wide as the widest, in octaves, tie with it: the lowest is kept.
TIE = 0.9
# m/s: under it a ridge is an artefact (a correlation's zero lag, a mute's edge), as G2 counts.
ARTEFACT_SPEED = 30.0
# A step steeper than this |d ln v / d ln f| is a jump onto another ridge (G3's mode jump,
# sigpipe.masw.quality.curve.JUMP_SLOPE): a ridge turns less steeply.
JUMP_SLOPE = 2.0
# How much faster than the level the ridge climbs, at least, to be bridged: a higher mode's,
# 50 to 70 % faster on active_p2's 21-receiver windows; a ridge's own wiggle in noise (33 % on
# PACo's 7-shot test line) stays.
BRIDGED_RISE = 0.4
# How close to the level it held a ridge comes back, as a share of it, to be on it again: at most
# LEVEL_BAND faster (the share that also bounds the columns holding the level) and LEVEL_FALL
# slower (M0 slowing with frequency across the faster ridge; a ridge far slower is an artefact's).
LEVEL_BAND = 0.12
LEVEL_FALL = 0.2
# How bright the level's ridge is, at least, as a share of the faster ridge's: M0 holding its
# level, not a faint ridge under the mode (an alias, a smear's edge) the mode would seem to leave.
LEVEL_STRENGTH = 0.75
# The widest faster ridge bridged, as the log of its frequencies' ratio (0.4: a third wider at its
# top than at its foot, about half an octave), and the share of its columns kept at least: a
# higher mode brighter over a short band, a ridge of its own, not the noise or stripes of a wide
# stretch the ridge would seem to come back from.
BRIDGED_WIDTH = 0.4
BRIDGED_KEPT = 0.5


def _faster_ridge(
    frequencies: np.ndarray,
    velocities: np.ndarray,
    strength: np.ndarray,
    kept: np.ndarray,
    min_points: int,
) -> np.ndarray:
    """The columns where the ridge followed (`velocities` and its value `strength`, at each
    column) has left its level for a faster ridge it comes back to the level from: those between
    the last kept column within LEVEL_BAND of the level before the rise and the first kept one
    back within it, its median over 5 kept columns too and the next `min_points` kept columns'.
    The ridge rises more than BRIDGED_RISE above the level, by a step steeper than JUMP_SLOPE
    from that last column. The level is the one the ridge holds just before: the slowest of its medians
    over 5 kept columns at the `min_points` kept columns before the rise, all within LEVEL_BAND of
    it (a ridge falling fast, or a stray column far below, holds none), the ridge there at least
    LEVEL_STRENGTH as bright as the faster one. M0 is the slowest mode: the faster ridge is a
    higher mode's, brighter there, M0 under it a shoulder whose columns hold no peak of it. A ridge
    that rises gently (a bump of its own) or does not come back stays."""
    faster = np.zeros_like(kept)
    indices = np.flatnonzero(kept)
    if indices.size <= 2 * min_points:
        return faster
    f = frequencies[indices]
    v = velocities[indices]
    s = strength[indices]
    medians = np.asarray(median_filter(v, size=5, mode="nearest"), dtype=np.float64)
    position = min_points
    while position < v.size:
        before = medians[position - min_points : position]
        level = float(before.min())
        band = (1 + LEVEL_BAND) * level
        if before.max() > band or v[position] <= (1 + BRIDGED_RISE) * level:
            position += 1
            continue
        floor = (1 - LEVEL_FALL) * level
        foot = position - 1
        while foot >= 0 and not floor <= v[foot] <= band:
            foot -= 1
        back = position + 1
        while back < v.size and not (floor <= v[back] <= band and floor <= medians[back] <= band):
            back += 1
        after = medians[back : back + min_points]
        steep = foot >= 0 and (
            math.log(v[position] / v[foot]) / math.log(f[position] / f[foot]) > JUMP_SLOPE
        )
        held = (
            after.size == min_points and float(after.max()) <= band and float(after.min()) >= floor
        )
        bright = float(np.median(s[position - min_points : position])) >= LEVEL_STRENGTH * float(
            np.median(s[foot + 1 : back])
        )
        narrow = (
            steep
            and back < v.size
            and (
                math.log(f[back] / f[foot]) <= BRIDGED_WIDTH
                and (back - foot - 1) >= BRIDGED_KEPT * (indices[back] - indices[foot] - 1)
            )
        )
        if steep and held and bright and narrow:
            faster[indices[foot] + 1 : indices[back]] = True
            position = back
        else:
            position += 1
    return faster


def _levelled(
    frequencies: np.ndarray, velocities: np.ndarray, ridge: np.ndarray, bridged: np.ndarray
) -> np.ndarray:
    """`ridge` (velocity indices) along its level across each stretch of `bridged` columns: its
    velocity there interpolated, in log-log, between the columns on either side."""
    levelled = ridge.copy()
    columns = np.flatnonzero(bridged)
    for stretch in np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1):
        first, last = int(stretch[0]) - 1, int(stretch[-1]) + 1
        ends = np.log(frequencies[[first, last]])
        level = np.exp(
            np.interp(np.log(frequencies[stretch]), ends, np.log(velocities[ridge[[first, last]]]))
        )
        levelled[stretch] = np.searchsorted(velocities, level, side="left")
    return levelled


def _slower_below(
    values: np.ndarray, velocities: np.ndarray, path: np.ndarray, floor: float
) -> np.ndarray:
    """Per column, whether a slower ridge stands under the pick: a local maximum reaching `floor`
    between SLOWER_BAND times the pick's velocity, where M0 lies under a higher mode."""
    low, high = SLOWER_BAND
    inner = (values[:, 1:-1] >= values[:, :-2]) & (values[:, 1:-1] >= values[:, 2:])
    peaks = np.zeros(values.shape, dtype=bool)
    peaks[:, 1:-1] = inner & (values[:, 1:-1] >= floor)
    picked = velocities[path][:, None]
    within = (velocities[None, :] >= low * picked) & (velocities[None, :] <= high * picked)
    return (peaks & within).any(axis=1)


# Where a slower ridge under a pick lies, as shares of its velocity: M0 under a higher mode (at
# 1.3 to 2.5 times M0), beyond its own corridor.
SLOWER_BAND = (0.4, 0.8)


def _low_end(
    frequencies: np.ndarray,
    velocities: np.ndarray,
    kept: np.ndarray,
    fall: float,
    length: float,
) -> np.ndarray:
    """`kept` without its low-frequency end where the pick slows down toward low frequencies,
    where its wavelengths are longer than the window (`length`, m): from the lowest kept point
    up, each more than `fall` slower than the fastest kept point above it there is dropped, until
    one is not. A mode's velocity rises toward its lowest frequencies; one that falls there is the
    near field, or a ridge the window no longer resolves. At shorter wavelengths nothing is cut:
    a bump or a rise stays."""
    indices = np.flatnonzero(kept & (velocities >= length * frequencies))
    trimmed = kept.copy()
    for position, index in enumerate(indices.tolist()):
        above = indices[position + 1 :]
        if not above.size or velocities[index] >= (1 - fall) * velocities[above].max():
            break
        trimmed[index] = False
    return trimmed


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


def _picked(image: DispersionImage, span: slice, tracked: _Tracked) -> np.ndarray:
    """The velocities of `tracked`'s kept points, as a lasso drawn along its corridor picks them
    (PAC's lasso): each the brightest peak of its column within the corridor (brightest_peaks;
    where the column has none there, the tracked velocity), cleaned (clean_picks): a peak
    standing apart from its neighbours' (noise, a ridge too broad to place it) gives way to their
    median, where an interpolation over a run of such peaks, at a broad end, would carry them on.

    The search finds which ridge to follow and where it holds; its smoothness, which keeps it on
    one ridge through noise, holds it off the crest where the ridge turns fast (a steep branch,
    its knee, a bump) and where a column's maxima pull its corridor away. Each column's peak is
    that frequency's velocity."""
    kept = tracked.kept
    rows = np.arange(image.fs.size)[span][kept]
    grid = image.vs.astype(np.float64)
    if rows.size == 0:
        return np.empty(0, dtype=np.float64)
    cells = np.arange(grid.size)
    within = (cells[None, :] >= tracked.low[kept, None]) & (
        cells[None, :] <= tracked.high[kept, None]
    )
    values = image.fv_map[rows]
    found, peaks = brightest_peaks(values, grid, within)
    velocities = grid[tracked.path[kept]]
    velocities[found] = peaks
    neighbours = np.asarray(median_filter(velocities, size=5, mode="nearest"), dtype=np.float64)
    cleaned = clean_picks(image.fs[rows], velocities, fallback=neighbours)
    return np.asarray(cleaned, dtype=np.float64)


def _lassoed(
    image: DispersionImage,
    parameters: PickingParameters,
    start: np.ndarray,
    stop: np.ndarray,
    noise_floor: float,
) -> tuple[PickedMode, np.ndarray] | None:
    """M0 picked inside the picking's lasso as PAC's lasso picks (pick_lasso), within the search
    bounds `start` to `stop`: each column's brightest peak inside, cleaned, kept where it reaches
    the noise floor as the tracker's points must (point_min_ratio); pinned where the lasso's edge
    cuts a ridge still rising there (no peak inside: its brightest cell inside reported); its
    curve resampled every wavelength_step metres of wavelength. With where the next mode is
    searched from, above the lasso's top; None when the lasso holds fewer than 2 such peaks."""
    polygon = parameters.lasso or ()
    grid = image.vs.astype(np.float64)
    cells = np.arange(grid.size)
    inside = (
        lasso_cells(image, polygon)
        & (cells[None, :] >= start[:, None])
        & (cells[None, :] <= stop[:, None])
    )
    rows, picked = lasso_picks(image, inside, parameters.point_min_ratio * noise_floor)
    if rows.size < 2:
        return None
    peaked, _ = brightest_peaks(image.fv_map, image.vs, inside)
    held = np.flatnonzero(inside.any(axis=1))
    span = slice(int(held[0]), int(held[-1]) + 1)
    frequencies = image.fs[span].astype(np.float64)
    values = image.fv_map[span].astype(np.float64)
    within = inside[span]
    some = within.any(axis=1)
    found = rows - span.start

    velocities = grid[np.argmax(np.where(within, values, -np.inf), axis=1)]
    velocities[found] = picked
    # Pinned where the column has no peak inside (the edge decided); kept where its peak reaches
    # the noise floor; a peak under it, neither.
    pinned = np.ones(frequencies.size, dtype=bool)
    pinned[peaked - span.start] = False
    kept = np.zeros(frequencies.size, dtype=bool)
    kept[found] = True
    # A column the lasso holds no cell of (a bound, a concave lasso): between its neighbours.
    velocities[~some] = np.interp(frequencies[~some], frequencies[found], picked)
    nearest = np.clip(np.searchsorted(grid, velocities), 0, grid.size - 1)
    coherence = values[np.arange(frequencies.size), nearest]

    step = np.gradient(grid)
    first = np.argmax(within, axis=1)
    last = grid.size - 1 - np.argmax(within[:, ::-1], axis=1)
    low = np.interp(frequencies, frequencies[some], (grid[first] - step[first] / 2)[some])
    high = np.interp(frequencies, frequencies[some], (grid[last] + step[last] / 2)[some])

    mode = PickedMode(
        number=0,
        frequencies=frequencies,
        velocities=velocities,
        coherence=coherence,
        pinned=pinned,
        kept=kept,
        noise_floor=noise_floor,
        curve=lasso_curve(image, rows, picked, polygon, Mode("M", 0), parameters.wavelength_step),
        low=low,
        high=high,
    )
    above = np.full_like(start, grid.size)
    above[span] = np.where(some, last + 1, grid.size)
    return mode, above


def _curve(
    image: DispersionImage,
    frequencies: np.ndarray,
    velocities: np.ndarray,
    number: int,
    step: float,
) -> DispersionCurve | None:
    """The kept points as a sigpipe curve, like PAC's hand picks: labelled M<n>, with Lorentzian
    uncertainties, resampled over wavelength every `step` metres, along the curve where its
    wavelength turns back."""
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
    return resample_wavelength(curve, step=step, along=True)
