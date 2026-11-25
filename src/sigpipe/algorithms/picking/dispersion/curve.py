import math
from itertools import pairwise

import numpy as np
from scipy.interpolate import interp1d, make_smoothing_spline
from scipy.ndimage import median_filter

from sigpipe.base.acquisition import Acquisition
from sigpipe.base.dispersion_curve import (
    DispersionCurve,
    DispersionCurvesImage,
    Mode,
)
from sigpipe.base.dispersion_image import DispersionImage


def lorentzian_uncertainty(
    fs: np.ndarray,
    vs: np.ndarray,
    acquisition: Acquisition,
    a: float = 0.5,
) -> np.ndarray | None:
    """Per-point phase-velocity uncertainty from the receiver array's resolving power.

    Lorentzian resolution formula: a tighter array (more receivers, smaller
    spacing) resolves velocity more precisely, so its curve gets a smaller
    uncertainty. Receiver count and spacing are taken from the curve's own
    acquisition geometry (assumed uniform, from the first two receivers).

    Returns None when the acquisition's geometry isn't known (e.g. a stacked
    dispersion image whose shots don't share a single geometry) — there is
    no array to derive a resolving power from.
    """

    if acquisition.is_unknown or len(acquisition.receivers) < 2:
        return None

    receivers = acquisition.receivers
    n_receivers = len(receivers)
    dx = abs(receivers[1].x - receivers[0].x)

    fs = np.asarray(fs, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)

    fac = 10 ** (1 / np.sqrt(n_receivers * dx))
    dc_left = 1 / (1 / vs + 1e-12 - 1 / (2 * fs * n_receivers * fac * dx + 1e-12))
    dc_right = 1 / (1 / vs + 1e-12 + 1 / (2 * fs * n_receivers * fac * dx + 1e-12))
    resolution = np.abs(dc_left - dc_right)

    raw = (10**-a) * resolution
    uncertainty = np.where(raw > 0.4 * vs, 0.4 * vs, raw)
    uncertainty = np.where(raw < 5, 5, uncertainty)

    return uncertainty.astype(np.float32)


def receiver_spacings(acquisition: Acquisition) -> list[float] | None:
    """Distances between consecutive receivers of a line (ordered by x),
    along the ground (x, z), or None when the array geometry isn't known.
    """
    if acquisition.is_unknown or len(acquisition.receivers) < 2:
        return None

    receivers = sorted(acquisition.receivers, key=lambda receiver: receiver.x)
    return [math.hypot(b.x - a.x, b.z - a.z) for a, b in pairwise(receivers)]


def min_resolvable_wavelength(acquisition: Acquisition) -> float | None:
    """Smallest wavelength a receiver line can reliably resolve: twice its
    smallest spacing (the spatial Nyquist limit), or None when the array
    geometry isn't known. Below it, picks are spatially aliased.
    """
    spacings = receiver_spacings(acquisition)
    return 2 * min(spacings) if spacings else None


# The shortest wavelength the picker searches, in receiver spacings: one, a clear ridge
# followed into the aliasing zone under two (min_resolvable_wavelength), where its points are
# flagged, not cut.
PICKED_SPACINGS = 1.0


def shortest_picked_wavelength(
    acquisition: Acquisition, spacings: float = PICKED_SPACINGS
) -> float | None:
    """The shortest wavelength the picker searches: `spacings` of the line's smallest receiver
    spacing (the picker's min_wavelength), or None when the array geometry isn't known."""
    receivers = receiver_spacings(acquisition)
    return spacings * min(receivers) if receivers else None


def max_resolvable_wavelength(acquisition: Acquisition) -> float | None:
    """Longest wavelength a receiver line resolves: its length along the
    ground (the aperture), or None when the array geometry isn't known.
    """
    spacings = receiver_spacings(acquisition)
    return sum(spacings) if spacings else None


# The longest wavelength the checks trust, in window lengths: three. The picker follows a ridge
# beyond it, where its points are flagged, not cut.
REACHED_LENGTHS = 3.0


def longest_reached_wavelength(
    acquisition: Acquisition, lengths: float = REACHED_LENGTHS
) -> float | None:
    """The longest wavelength a window reaches: `lengths` of its length along the ground, or None
    when the array geometry isn't known. Beyond it, a pick's points are flagged as beyond the
    window's reach."""
    aperture = max_resolvable_wavelength(acquisition)
    return lengths * aperture if aperture else None


def resample_wavelength(
    curve: DispersionCurve,
    step: float = 1.0,
    wmax: float | None = None,
    along: bool = False,
) -> DispersionCurve:
    """Resample a picked curve onto a uniform wavelength grid.

    Interpolates velocity (and uncertainty, if present) over wavelength
    w = v/f at uniform steps, then converts back to frequency and re-sorts
    by frequency. Where the curve's wavelength turns back (its velocity rising
    faster than its frequency, a stretch of inverse dispersion), the curve meets
    a wavelength more than once: `along`, each meeting is kept, along the curve
    (_crossings: a hand pick's), where interpolating over the wavelengths sorted
    mixes points of both stretches (the tracker's, as PACo's checks were set on).
    """
    fs = np.asarray(curve.fs, dtype=np.float64)
    vs = np.asarray(curve.vs, dtype=np.float64)
    w = vs / fs

    w_min = float(np.ceil(np.min(w) / step) * step)
    w_max = float(np.floor(np.max(w) / step) * step)

    if w_min >= w_max:
        return DispersionCurve(
            fs=fs[:1],
            vs=vs[:1],
            mode=curve.mode,
            acquisition=curve.acquisition,
            vs_err=curve.vs_err[:1] if curve.vs_err is not None else None,
            type=curve.type,
        )

    n_steps = round((w_max - w_min) / step) + 1
    w_resamp = np.linspace(w_min, w_max, n_steps)

    by_frequency = np.argsort(fs, kind="stable")
    turns = np.diff(np.sign(np.diff(w[by_frequency])))
    if along and (np.any(turns != 0) or np.any(np.diff(w[by_frequency]) == 0)):
        w_resamp, vs_resamp, vs_err_resamp = _crossings(
            w[by_frequency],
            vs[by_frequency],
            np.asarray(curve.vs_err, dtype=np.float64)[by_frequency]
            if curve.vs_err is not None
            else None,
            w_resamp,
        )
    else:
        vs_resamp = interp1d(w, vs, kind="linear")(w_resamp)
        vs_err_resamp = (
            interp1d(
                w,
                np.asarray(curve.vs_err, dtype=np.float64),
                kind="linear",
                fill_value="extrapolate",  # pyright: ignore[reportArgumentType] -- scipy's stub omits this valid literal
            )(w_resamp)
            if curve.vs_err is not None
            else None
        )

    if wmax is not None and w_resamp.max() > wmax:
        keep = w_resamp <= wmax
        if not np.any(keep):
            keep[0] = True
        w_resamp = w_resamp[keep]
        vs_resamp = vs_resamp[keep]
        if vs_err_resamp is not None:
            vs_err_resamp = vs_err_resamp[keep]

    fs_resamp = vs_resamp / w_resamp
    order = np.argsort(fs_resamp)
    fs_resamp = fs_resamp[order]
    vs_resamp = vs_resamp[order]
    if vs_err_resamp is not None:
        vs_err_resamp = vs_err_resamp[order]

    return DispersionCurve(
        fs=fs_resamp,
        vs=vs_resamp,
        mode=curve.mode,
        acquisition=curve.acquisition,
        vs_err=vs_err_resamp,
        type=curve.type,
    )


def _crossings(
    w: np.ndarray, vs: np.ndarray, errors: np.ndarray | None, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Where a curve, its points in frequency order at wavelengths `w`, meets each wavelength of
    `grid`, every time: the wavelengths met, the velocities there (and uncertainties, if given),
    interpolated linearly in wavelength between the two points around. A wavelength met at a
    point is taken once there."""
    met: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    uncertainties: list[np.ndarray] = []
    last = w.size - 2
    for i in range(w.size - 1):
        a, b = w[i], w[i + 1]
        if a == b:
            continue
        meets = (grid >= min(a, b)) & (grid <= max(a, b))
        if i < last:
            meets &= grid != b  # the next stretch's, from its first point
        t = grid[meets]
        share = (t - a) / (b - a)
        met.append(t)
        velocities.append(vs[i] + share * (vs[i + 1] - vs[i]))
        if errors is not None:
            uncertainties.append(errors[i] + share * (errors[i + 1] - errors[i]))
    if not met:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty if errors is not None else None
    return (
        np.concatenate(met),
        np.concatenate(velocities),
        np.concatenate(uncertainties) if errors is not None else None,
    )


# The most columns between two picks that a hand pick's resampling bridges: farther apart, the
# image has no peak between them where the pick was drawn, and nothing is made up there.
BRIDGED_COLUMNS = 2


def resample_runs(
    curve: DispersionCurve, rows: np.ndarray, step: float = 1.0, bridged: int = BRIDGED_COLUMNS
) -> DispersionCurve:
    """`curve`, its points picked on the image's `rows`, resampled over wavelength along it
    (resample_wavelength) run by run: picks at most `bridged` columns apart in one run, a run of
    a single pick dropped. Across a wider gap a single resampling would draw a chord, off the
    ridge; each run keeps to its picks."""
    rows = np.asarray(rows)
    breaks = np.flatnonzero(np.diff(rows) > bridged) + 1
    runs = [run for run in np.split(np.arange(rows.size), breaks) if run.size >= 2]
    pieces = [
        resample_wavelength(
            DispersionCurve(
                fs=curve.fs[run],
                vs=curve.vs[run],
                mode=curve.mode,
                acquisition=curve.acquisition,
                vs_err=curve.vs_err[run] if curve.vs_err is not None else None,
                type=curve.type,
            ),
            step=step,
            along=True,
        )
        for run in runs
    ]
    if not pieces:
        return resample_wavelength(curve, step=step, along=True)
    fs = np.concatenate([piece.fs for piece in pieces])
    order = np.argsort(fs, kind="stable")
    errors = [piece.vs_err for piece in pieces]
    return DispersionCurve(
        fs=fs[order],
        vs=np.concatenate([piece.vs for piece in pieces])[order],
        mode=curve.mode,
        acquisition=curve.acquisition,
        vs_err=np.concatenate([e for e in errors if e is not None])[order]
        if all(e is not None for e in errors)
        else None,
        type=curve.type,
    )


def pick_curves(
    dispersion_image: DispersionImage,
    fmins: list[float | None] | None = None,
    fmaxs: list[float | None] | None = None,
    vmins: list[float | None] | None = None,
    vmaxs: list[float | None] | None = None,
    lbdmins: list[float | None] | None = None,
    lbdmaxs: list[float | None] | None = None,
    modeled_curves: list[DispersionCurve | None] | None = None,
    modeled_dvs: list[float | None] | None = None,
    labels: list[str] | None = None,
    modes: list[int] | None = None,
    resample_over_wavelength: bool = False,
) -> DispersionImage:
    """Pick one curve per label as the per-frequency energy maximum of
    `fv_map`, within whatever bounds that label is given: the brightest peak of each column
    within them (brightest_peaks), so that a column whose ridge a bound cuts gives no pick.

    `modeled_curves`/`modeled_dvs` restrict a label's search to a band
    around a known reference curve -- at each frequency, only velocities
    within `+/- modeled_dv` of the modeled curve's own velocity there are
    eligible. Use it when the global maximum lands on the wrong mode (or on
    noise) but the curve's rough shape is known in advance. Outside the
    modeled curve's own frequency range there is no band to search, so
    nothing is picked there.

    A mode that already has a curve on `dispersion_image` gets the new one.
    """

    if labels is None:
        labels = [""]
    if fmins is None:
        fmins = [None for _ in labels]
    if fmaxs is None:
        fmaxs = [None for _ in labels]
    if vmins is None:
        vmins = [None for _ in labels]
    if vmaxs is None:
        vmaxs = [None for _ in labels]
    if lbdmins is None:
        lbdmins = [None for _ in labels]
    if lbdmaxs is None:
        lbdmaxs = [None for _ in labels]
    if modeled_curves is None:
        modeled_curves = [None for _ in labels]
    if modeled_dvs is None:
        modeled_dvs = [None for _ in labels]
    if modes is None:
        modes = list(range(len(labels)))

    dispersion_curves: list[DispersionCurve] = (
        list(dispersion_image.dispersion_curves)
        if dispersion_image.dispersion_curves is not None
        else []
    )
    picked_curves: list[DispersionCurve] = []

    lengths = {
        len(fmins),
        len(fmaxs),
        len(vmins),
        len(vmaxs),
        len(lbdmins),
        len(lbdmaxs),
        len(modeled_curves),
        len(modeled_dvs),
        len(labels),
        len(modes),
    }
    if len(lengths) > 1:
        raise ValueError(
            "requires same length for fmins, fmaxs, vmins, vmaxs, lbdmins, lbdmaxs, "
            "modeled_curves, modeled_dvs, labels, modes"
        )

    for fmin, fmax, vmin, vmax, lbdmin, lbdmax, modeled_curve, modeled_dv, label, mode in zip(
        fmins,
        fmaxs,
        vmins,
        vmaxs,
        lbdmins,
        lbdmaxs,
        modeled_curves,
        modeled_dvs,
        labels,
        modes,
        strict=False,
    ):
        if (modeled_curve is None) != (modeled_dv is None):
            raise ValueError(
                f"modeled_curves and modeled_dvs must both be set or both be None per label, "
                f"got modeled_curve={modeled_curve!r}, modeled_dv={modeled_dv!r} "
                f"for label '{label}'"
            )
        if modeled_dv is not None and modeled_dv <= 0:
            raise ValueError(f"modeled_dvs must be > 0, got {modeled_dv} for label '{label}'")
        fs = dispersion_image.fs
        vs = dispersion_image.vs

        F, V = np.meshgrid(fs, vs, indexing="ij")
        within = np.isfinite(dispersion_image.fv_map)
        wavelength = V / (F + 1e-12)
        if lbdmin is not None:
            within &= wavelength >= lbdmin
        if lbdmax is not None:
            within &= wavelength <= lbdmax

        if modeled_curve is not None and modeled_dv is not None:
            # np.interp would clamp to the modeled curve's endpoint
            # velocities outside its own frequency range, turning "no model
            # here" into a flat band at an arbitrary velocity -- mask those
            # frequencies out entirely instead, so nothing is picked where
            # the modeled curve says nothing.
            modeled_vs = np.interp(fs, modeled_curve.fs, modeled_curve.vs)
            in_modeled_range = (fs >= modeled_curve.fs.min()) & (fs <= modeled_curve.fs.max())
            v_lo = (modeled_vs - modeled_dv)[:, None]
            v_hi = (modeled_vs + modeled_dv)[:, None]
            within &= (v_lo <= V) & (v_hi >= V) & in_modeled_range[:, None]

        if fmin is not None:
            within &= fmin <= F
        if fmax is not None:
            within &= fmax >= F
        if vmin is not None:
            within &= vmin <= V
        if vmax is not None:
            within &= vmax >= V

        n_rows = int(np.count_nonzero(within.any(axis=1)))
        if n_rows < 2:
            raise ValueError(
                f"label '{label}': bounds leave {n_rows} frequencies with any energy, "
                "too few to pick a curve -- widen fmin/fmax, vmin/vmax, lbdmin/lbdmax, "
                "or modeled_dv"
            )
        rows, peaks = brightest_peaks(dispersion_image.fv_map, vs, within)
        if rows.size < 2:
            raise ValueError(
                f"label '{label}': the image peaks within the bounds at {rows.size} "
                "frequencies, too few to pick a curve (where a bound cuts a ridge, its "
                "column has no peak within them) -- widen fmin/fmax, vmin/vmax, "
                "lbdmin/lbdmax, or modeled_dv"
            )
        picked_fs = fs[rows]
        picked_vs = clean_picks(picked_fs, peaks)

        picked_curve = DispersionCurve(
            fs=picked_fs,
            vs=picked_vs,
            mode=Mode(label if label else "M", mode),
            acquisition=dispersion_image.acquisition,
            type=dispersion_image.type,
            vs_err=lorentzian_uncertainty(picked_fs, picked_vs, dispersion_image.acquisition),
        )

        if resample_over_wavelength:
            picked_curve = resample_runs(picked_curve, rows)

        picked_curves.append(picked_curve)

    repicked = {curve.mode for curve in picked_curves}
    kept = [curve for curve in dispersion_curves if curve.mode not in repicked]

    return DispersionImage(
        fv_map=dispersion_image.fv_map,
        fs=dispersion_image.fs,
        vs=dispersion_image.vs,
        type=dispersion_image.type,
        acquisition=dispersion_image.acquisition,
        dispersion_curves=DispersionCurvesImage(dispersion_curves=(*kept, *picked_curves)),
    )


# The faintest peak a pick takes, as a share of the brightest value it may take in its column:
# fainter, a bump beside the ridge (a sidelobe, noise) where the bounds cut the ridge itself.
PEAK_SHARE = 0.5


def brightest_peaks(
    values: np.ndarray, velocities: np.ndarray, within: np.ndarray, share: float = PEAK_SHARE
) -> tuple[np.ndarray, np.ndarray]:
    """Per frequency (row) of `values` [n_f, n_v]: the brightest of the column's peaks `within`
    the bounds (a [n_f, n_v] mask), when it reaches `share` of the brightest value within them,
    its velocity refined between cells (refined_peaks). The rows picked, and their velocities.

    A peak is where the whole column rises into a cell and falls after it (column_peaks), so
    that where a bound cuts a ridge still rising at its edge, the column gives no pick: the
    ridge's maximum lies beyond, and the edge is not where the image peaks. Each column of a
    phase-shift image is a frequency's own delay and sum, so its peak is that frequency's
    velocity, however steep the curve there."""
    column = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    peaks = column_peaks(column)
    held = np.where(within, column, -np.inf)
    brightest = held.max(axis=1, initial=-np.inf)
    candidates = peaks & within & (column >= share * brightest[:, None])
    rows = np.flatnonzero(candidates.any(axis=1))
    cells = np.argmax(np.where(candidates[rows], column[rows], -np.inf), axis=1)
    return rows, refined_peaks(column, np.asarray(velocities, dtype=np.float64), rows, cells)


def column_peaks(values: np.ndarray) -> np.ndarray:
    """[n_f, n_v] mask: the cells each column of `values` rises into and falls after, past any
    flat stretch (each cell of a flat top). A column that only rises, or only falls, has none,
    whatever steps rounding leaves on its slope; nor has a flat one, nor its edges."""
    steps = np.sign(np.diff(values, axis=1))  # step j: from cell j to j + 1
    n_steps = steps.shape[1]
    positions = np.arange(n_steps)
    # The last step that changes the value before each cell, and the first from it on.
    before = np.maximum.accumulate(np.where(steps != 0, positions, -1), axis=1)
    after = np.minimum.accumulate(np.where(steps != 0, positions, n_steps)[:, ::-1], axis=1)[
        :, ::-1
    ]
    rises = np.take_along_axis(steps, np.clip(before, 0, None), axis=1) > 0
    rises &= before >= 0
    falls = np.take_along_axis(steps, np.clip(after, None, n_steps - 1), axis=1) < 0
    falls &= after < n_steps
    peaks = np.zeros(values.shape, dtype=bool)
    # Cell j (inner): risen into by the steps before it (to j - 1), falling from step j on.
    peaks[:, 1:-1] = rises[:, :-1] & falls[:, 1:]
    return peaks


def refined_peaks(
    values: np.ndarray, velocities: np.ndarray, rows: np.ndarray, cells: np.ndarray
) -> np.ndarray:
    """The velocity of each peak (`rows`, `cells`: inner cells of `values`), between cells: the
    vertex of the parabola through the peak and its two neighbours, at most half a cell away."""
    if rows.size == 0:
        return np.empty(0, dtype=np.float64)
    left, peak, right = (values[rows, cells + k] for k in (-1, 0, 1))
    curvature = left - 2 * peak + right
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(curvature < 0, 0.5 * (left - right) / curvature, 0.0)
    shift = np.clip(np.nan_to_num(shift), -0.5, 0.5)
    step = np.where(
        shift >= 0,
        velocities[cells + 1] - velocities[cells],
        velocities[cells] - velocities[cells - 1],
    )
    return velocities[cells] + shift * step


def clean_picks(fs: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """Picked velocities `vs` at frequencies `fs`, cleaned: each point far from the median of its
    5 neighbours (beyond 2.5 times the median distance, and 1 % of its velocity) replaced by the
    others' interpolation, then the curve smoothed (smoothed)."""
    fs = np.asarray(fs, dtype=np.float64)
    vs = np.array(vs, dtype=np.float64)
    # The ends padded with their own values: zeros would make them outliers.
    median_vs = median_filter(vs, size=5, mode="nearest")
    residual = np.abs(vs - median_vs) / median_vs
    # On a monotonic stretch each point is its neighbours' median: with more than half the curve
    # such, the median distance is 0, and without a floor any point off it would be an outlier.
    outliers = residual > max(2.5 * float(np.median(residual)), 0.01)
    if np.any(outliers) and np.count_nonzero(~outliers) >= 2:
        valid = ~outliers
        vs[outliers] = np.interp(fs[outliers], fs[valid], vs[valid])
    return smoothed(fs, vs).astype(np.float32)


def smoothed(fs: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """`vs` smoothed: the smoothing spline of ln v over ln f whose smoothness generalized
    cross-validation chooses (scipy's make_smoothing_spline), as little as the picks' own scatter
    calls for. Where they follow a steep branch and its knee, so does the curve, which one
    polynomial over much of it would cut across; where they scatter about a flat stretch, it is
    smoothed as much. Under 5 points, or frequencies not increasing, kept as picked."""
    fs = np.asarray(fs, dtype=np.float64)
    vs = np.array(vs, dtype=np.float64)
    if fs.size < 5 or np.any(fs <= 0) or np.any(vs <= 0) or np.any(np.diff(fs) <= 0):
        return vs
    x = np.log(fs)
    spline = make_smoothing_spline(x, np.log(vs))
    return np.exp(np.asarray(spline(x), dtype=np.float64))
