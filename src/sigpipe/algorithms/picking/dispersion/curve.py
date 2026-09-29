import math
from itertools import pairwise

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

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
) -> DispersionCurve:
    """Resample a picked curve onto a uniform wavelength grid.

    Interpolates velocity (and uncertainty, if present) over wavelength
    w = v/f at uniform steps, then converts back to frequency and re-sorts
    by frequency.
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
    `fv_map`, within whatever bounds that label is given.

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
        fs = dispersion_image.fs.copy()
        vs = dispersion_image.vs.copy()
        fv_map = dispersion_image.fv_map.copy()

        F, V = np.meshgrid(fs, vs, indexing="ij")
        wavelength = V / (F + 1e-12)
        wavelength_mask = np.ones_like(fv_map, dtype=bool)
        if lbdmin is not None:
            wavelength_mask &= wavelength >= lbdmin
        if lbdmax is not None:
            wavelength_mask &= wavelength <= lbdmax
        fv_map[~wavelength_mask] = np.nan

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
            band_mask = (v_lo <= V) & (v_hi >= V) & in_modeled_range[:, None]
            fv_map[~band_mask] = np.nan

        mask_f = np.ones_like(fs, dtype=bool)
        if fmin is not None:
            mask_f &= fs >= fmin
        if fmax is not None:
            mask_f &= fs <= fmax
        mask_v = np.ones_like(vs, dtype=bool)
        if vmin is not None:
            mask_v &= vs >= vmin
        if vmax is not None:
            mask_v &= vs <= vmax
        fs = fs[mask_f]
        vs = vs[mask_v]
        fv_map = fv_map[mask_f][:, mask_v]

        valid_rows = ~np.all(np.isnan(fv_map), axis=1)
        fs = fs[valid_rows]
        fv_map = fv_map[valid_rows]

        if len(fs) < 2:
            raise ValueError(
                f"label '{label}': bounds leave {len(fs)} frequencies with any energy, "
                "too few to pick a curve -- widen fmin/fmax, vmin/vmax, lbdmin/lbdmax, "
                "or modeled_dv"
            )

        idx = np.array([np.where(row == np.nanmax(row))[0][-1] for row in fv_map])
        picked_vs = clean_picks(fs, vs[idx])

        picked_curve = DispersionCurve(
            fs=fs,
            vs=picked_vs,
            mode=Mode(label if label else "M", mode),
            acquisition=dispersion_image.acquisition,
            type=dispersion_image.type,
            vs_err=lorentzian_uncertainty(fs, picked_vs, dispersion_image.acquisition),
        )

        if resample_over_wavelength:
            picked_curve = resample_wavelength(picked_curve)

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


def clean_picks(fs: np.ndarray, vs: np.ndarray, polyorder: int = 2) -> np.ndarray:
    """Picked velocities `vs` at frequencies `fs`, cleaned: each point far from
    the median of its 5 neighbours (beyond 2.5 times the median distance)
    replaced by the others' interpolation, then the curve smoothed."""
    vs = np.array(vs, dtype=np.float64)
    # The ends padded with their own values: zeros would make them outliers.
    median_vs = median_filter(vs, size=5, mode="nearest")
    residual = np.abs(vs - median_vs)
    outliers = residual > 2.5 * np.median(residual)
    if np.any(outliers):
        valid = ~outliers
        vs[outliers] = np.interp(fs[outliers], fs[valid], vs[valid])
    return _smooth(vs, polyorder)


def _smooth(vs: np.ndarray, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay over about half the curve (an odd window). A curve too
    short for the polynomial's order is kept as picked."""
    window = len(vs) // 2
    if window % 2 == 0:
        window += 1
    if window <= polyorder:
        return np.asarray(vs, dtype=np.float32)
    return np.asarray(
        savgol_filter(vs, window_length=window, polyorder=polyorder),
        dtype=np.float32,
    )
