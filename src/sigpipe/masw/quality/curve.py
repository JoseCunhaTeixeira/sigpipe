"""The measures of a picked curve (PACo's curve QC, G3, judges them; PAC measures its own runs'
alike): the pick on its image (measure_pick: sharpness, prominence, on the data, constant
wavelength), the points the window cannot resolve at either end, the points of the resampled
curve, the wavelengths it spans, its largest step, the air wave's share, the dispersion trend,
the uncertainty, and the nearest shot against the longest wavelength. Each measure says what it
describes (`of`: the curve) and what it covers."""

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import rankdata

from sigpipe.algorithms.picking.dispersion.curve import (
    longest_reached_wavelength,
    min_resolvable_wavelength,
)
from sigpipe.algorithms.picking.dispersion.tracking import PickedMode
from sigpipe.base.dispersion_curve import DispersionCurve
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.masw.quality.image import noise_floor
from sigpipe.masw.quality.measures import Measure, figure
from sigpipe.masw.quality.pick import PickMeasures, measure_pick


class PickLimits(BaseModel):
    """Limits past which a measure of the pick on its image fails.

    Tuned on the demo windows only, so recalibrate them on a reference set of windows judged by
    hand before trusting them. Sharpness and prominence are measured against a perfect plane
    wave for the same window, frequency and velocity: the demo's windows score 1.00 on both at
    every length from 5 to 24 receivers, where fixed limits would measure the array, not the
    data (a 5-receiver window cannot be as prominent as a 24-receiver one).
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    min_sharpness: float = Field(
        default=0.8,
        gt=0,
        description="Flag when the median peak width is below this multiple of a perfect plane "
        "wave's width for the same window (a plane wave scores 1).",
    )
    min_prominence: float = Field(
        default=0.5,
        gt=0,
        description="Flag when the median peak prominence (height above the noise floor over "
        "its column's median height) is below this multiple of a perfect plane wave's for the "
        "same window (a plane wave scores 1).",
    )
    min_on_data: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Flag when a smaller share of the points sit on their column's brightest "
        "value (within 10 %).",
    )
    max_constant_wavelength: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="Flag when a larger share of the points have a velocity growing like "
        "frequency: the edge of what the window resolves, not a dispersion curve.",
    )


class CurveLimits(BaseModel):
    """How a picked curve is measured, and its limits (PACo's G3 judges against them, PAC
    measures its own runs' alike): provisional, measured on the demo profiles."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    metrics: PickLimits = Field(
        default_factory=PickLimits,
        description="Sharpness, prominence, on_data and constant wavelength.",
    )
    max_jump: float = Field(
        default=0.3,
        gt=0,
        description="Relative velocity step between consecutive points of the resampled "
        "curve, at most: beyond it the pick jumped onto another mode.",
    )
    air_wave_band: tuple[float, float] = Field(
        default=(330.0, 345.0), description="m/s, where the air wave sits"
    )
    max_air_share: float = Field(
        default=0.5, gt=0, le=1, description="Share of the points in the air wave's band, at most."
    )
    min_points: int = Field(
        default=4,
        ge=2,
        description="Points of the resampled curve, at least: fewer, the pick is resampled finer "
        "or keeps more of the ridge.",
    )
    min_wavelength_ratio: float = Field(
        default=1.34,
        gt=1,
        description="The curve's longest wavelength over its shortest, at least: the inversion "
        "needs over 4/3 for its 3 layers, each at least a third of the shortest wavelength, "
        "down to half the longest. More points on a narrow span do not help.",
    )
    near_offset_wavelengths: float = Field(
        default=0.5,
        gt=0,
        description="The nearest shot's offset, as a share of the longest wavelength kept, at "
        "least: closer, the long wavelengths may read slow (near field). Reported, not applied.",
    )


@dataclass(frozen=True)
class CurveReport:
    """A picked curve's measures, and what its checks act on: its points by wavelength."""

    measures: tuple[Measure, ...]
    pick: PickMeasures  # the pick on its image
    fs: np.ndarray = field(default_factory=lambda: np.array([]))  # the curve's, by wavelength
    vs: np.ndarray = field(default_factory=lambda: np.array([]))
    wavelengths: np.ndarray = field(default_factory=lambda: np.array([]))
    shortest: float | None = None  # λmin, twice the receiver spacing
    longest: float | None = None  # λmax, three window lengths
    aliased: float = 0.0  # share of the points under λmin
    beyond: float = 0.0  # ...over λmax
    ratio: float = 0.0  # the longest wavelength over the shortest
    jump: float = 0.0  # the largest relative step between points by wavelength
    air: float = 0.0  # share of the points in the air wave's band
    trend: float = float("nan")  # velocity against wavelength, rank correlation
    uncertainty: float | None = None  # median over the velocity, over the points with one
    near_limit: float | None = None  # the nearest shot's offset should reach it, m


def pick_of(curve: DispersionCurve, image: DispersionImage) -> PickedMode:
    """A saved curve as a pick on `image` (a curve PAC saved, picked by hand or by PAC: the
    picker's own is gone): sampled as a pick is, one point on each of the image's frequencies
    within the curve's band, every point kept. A curve resampled over wavelength crowds its
    points into the long wavelengths, where the ridge is broadest: measured on them, a pick
    would read less on the data than the picker's own."""
    order = np.argsort(np.asarray(curve.fs, dtype=float))
    fs = np.asarray(curve.fs, dtype=float)[order]
    vs = np.asarray(curve.vs, dtype=float)[order]
    grid = np.asarray(image.fs, dtype=float)
    frequencies = grid[(grid >= fs.min()) & (grid <= fs.max())] if fs.size else grid[:0]
    velocities = np.interp(frequencies, fs, vs) if fs.size else frequencies
    return PickedMode(
        number=0,
        frequencies=frequencies,
        velocities=velocities,
        coherence=np.zeros(frequencies.size),
        pinned=np.zeros(frequencies.size, dtype=bool),
        kept=np.ones(frequencies.size, dtype=bool),
        noise_floor=noise_floor(image),
        curve=curve,
    )


def measure_curve(
    image: DispersionImage,
    m0: PickedMode | None,
    limits: CurveLimits,
    nearest_offset: float | None = None,
) -> CurveReport:
    """`image`'s M0 pick `m0` measured against `limits`, each measure saying what it covers:
    the pick on its image (its kept points), then its curve (resampled by wavelength), with the
    distance from the window's nearest shot to its receivers when known. One definition for
    PACo's G3 and PAC's views."""
    pick = measure_pick(image, m0)
    over = (
        f"the pick's {pick.n_points} points, {figure(pick.band_hz[0])}-{figure(pick.band_hz[1])} Hz"
        if pick.band_hz is not None
        else f"the pick's {pick.n_points} points"
    )
    pick_limits = limits.metrics
    measures = [
        Measure(
            name="sharpness",
            value=pick.sharpness,
            threshold=pick_limits.min_sharpness,
            bound="min",
            passed=pick.sharpness is None or pick.sharpness >= pick_limits.min_sharpness,
            of="curve",
            over=f"{over}: each peak's width against a plane wave's through the window",
        ),
        Measure(
            name="prominence",
            value=pick.prominence,
            threshold=pick_limits.min_prominence,
            bound="min",
            passed=pick.prominence is None or pick.prominence >= pick_limits.min_prominence,
            of="curve",
            over=f"{over}: each peak over its column, against a plane wave's",
        ),
        Measure(
            name="on_data",
            value=pick.on_data,
            threshold=pick_limits.min_on_data,
            bound="min",
            passed=pick.on_data is None or pick.on_data >= pick_limits.min_on_data,
            of="curve",
            over=f"{over}: on their column's brightest value, within 10 %",
        ),
        Measure(
            name="constant_wavelength",
            value=pick.constant_wavelength,
            threshold=pick_limits.max_constant_wavelength,
            bound="max",
            passed=pick.constant_wavelength is None
            or pick.constant_wavelength <= pick_limits.max_constant_wavelength,
            of="curve",
            over=f"{over}: velocity growing like frequency",
        ),
        Measure(
            name="n_points",
            value=pick.n_points,
            threshold=2,
            bound="min",
            passed=pick.n_points >= 2,
            of="curve",
            over="the pick's points kept on its ridge",
        ),
    ]
    if m0 is None or m0.curve is None:
        return CurveReport(tuple(measures), pick)

    curve = m0.curve
    fs, vs = np.asarray(curve.fs, dtype=float), np.asarray(curve.vs, dtype=float)
    lengths = vs / fs
    order = np.argsort(lengths)
    fs, vs, lengths = fs[order], vs[order], lengths[order]
    n_points = int(vs.size)
    points = f"the curve's {n_points} points"
    # The picker follows its ridge as far as it holds, at either end (the user, 2026-09-28): its
    # points under twice the spacing (the aliasing zone, where the ridge may be its alias) and
    # over three window lengths (beyond the window's reach, where it resolves no velocity). A
    # point on a limit is within it: resampled every metre, a curve can hold one at exactly twice
    # a 1.5 m spacing, which its float32 values put a hair under.
    shortest = min_resolvable_wavelength(image.acquisition)
    aliased = float(np.mean(_past(lengths, shortest, below=True))) if shortest else 0.0
    longest = longest_reached_wavelength(image.acquisition)
    beyond = float(np.mean(_past(lengths, longest, below=False))) if longest else 0.0
    measures += [
        Measure(
            name="aliased_points",
            value=round(aliased, 3),
            threshold=0,
            bound="max",
            passed=aliased == 0,
            of="curve",
            over=f"{points}: under twice the receiver spacing"
            + (f", {figure(shortest)} m" if shortest else ""),
        ),
        Measure(
            name="beyond_reach_points",
            value=round(beyond, 3),
            threshold=0,
            bound="max",
            passed=beyond == 0,
            of="curve",
            over=f"{points}: over three window lengths" + (f", {longest:.3g} m" if longest else ""),
        ),
        Measure(
            name="curve_points",
            value=n_points,
            threshold=limits.min_points,
            bound="min",
            passed=n_points >= limits.min_points,
            of="curve",
            over="the pick resampled by wavelength",
        ),
    ]
    # The wavelengths the curve spans, as a ratio: however many its points, a narrow span
    # resolves no layered model.
    ratio = float(lengths.max() / lengths.min()) if n_points else 0.0
    # A step between consecutive points of the resampled curve: the pick jumped onto another mode.
    jump = float(np.max(np.abs(np.diff(vs)) / vs[:-1])) if n_points > 1 else 0.0
    low, high = limits.air_wave_band
    air = float(np.mean((vs >= low) & (vs <= high))) if n_points else 0.0
    # Normal dispersion: velocity rising with wavelength.
    trend = _spearman(lengths, vs) if n_points >= 3 else float("nan")
    # Over the points with an uncertainty: one without must not make the median unknown.
    relative = (
        np.asarray(curve.vs_err, dtype=float)[order] / vs
        if curve.vs_err is not None and n_points
        else np.array([])
    )
    known = relative[np.isfinite(relative)]
    uncertainty = float(np.median(known)) if known.size else None
    spans = f"{points}, {lengths.min():.3g}-{lengths.max():.3g} m" if n_points else points
    measures += [
        Measure(
            name="wavelength_ratio",
            value=round(ratio, 2),
            threshold=limits.min_wavelength_ratio,
            bound="min",
            passed=ratio >= limits.min_wavelength_ratio,
            of="curve",
            over=f"{spans}: the longest wavelength over the shortest",
        ),
        Measure(
            name="max_jump",
            value=round(jump, 3),
            threshold=limits.max_jump,
            bound="max",
            passed=jump <= limits.max_jump,
            of="curve",
            over=f"{points}: each step to the next by wavelength",
        ),
        Measure(
            name="air_wave_share",
            value=round(air, 3),
            threshold=limits.max_air_share,
            bound="max",
            passed=air <= limits.max_air_share,
            of="curve",
            over=f"{points}: at {low:g}-{high:g} m/s",
        ),
        Measure(
            name="trend",
            value=None if not np.isfinite(trend) else round(trend, 3),
            threshold=0,
            bound="min",
            passed=not (np.isfinite(trend) and trend < 0),
            of="curve",
            over=f"{points}: velocity against wavelength, rank correlation",
        ),
        # Reported (the user, 2026-09-29): the picker caps each point's at 0.4 of its velocity,
        # so no limit over it could fail, and G5's depth informed judges what a loose curve does
        # to the model.
        Measure(
            name="uncertainty",
            value=None if uncertainty is None else round(uncertainty, 3),
            passed=True,
            of="curve",
            over=f"the {known.size} of its points with one: over the velocity",
        ),
    ]
    near_limit: float | None = None
    if nearest_offset is not None and n_points:
        near_limit = limits.near_offset_wavelengths * float(lengths.max())
        measures.append(
            Measure(
                name="near_offset",
                value=round(nearest_offset, 2),
                threshold=round(near_limit, 2),
                bound="min",
                passed=nearest_offset >= near_limit,
                unit="m",
                of="curve",
                over=f"the window's nearest shot, against {limits.near_offset_wavelengths:g} of "
                "the longest wavelength",
            )
        )
    return CurveReport(
        tuple(measures),
        pick,
        fs=fs,
        vs=vs,
        wavelengths=lengths,
        shortest=shortest,
        longest=longest,
        aliased=aliased,
        beyond=beyond,
        ratio=ratio,
        jump=jump,
        air=air,
        trend=trend,
        uncertainty=uncertainty,
        near_limit=near_limit,
    )


def _past(wavelengths: np.ndarray, limit: float | None, below: bool) -> np.ndarray:
    """The points strictly past `limit` (under it with `below`, else over it): a point on the
    limit, to float32's precision, is within it."""
    if limit is None:
        return np.zeros(wavelengths.size, dtype=bool)
    past = wavelengths < limit if below else wavelengths > limit
    return past & ~np.isclose(wavelengths, limit)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rank correlation; NaN when either input is constant."""
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
