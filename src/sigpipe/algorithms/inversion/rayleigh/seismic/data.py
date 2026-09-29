"""What both samplers of the Rayleigh-wave inversion share: the picked curves, the forward model
(disba's root search on a layered model) and the likelihood, with its noise factor.

The picks' uncertainties are scaled by a noise factor sampled with the model (hierarchical
Bayes, Bodin et al. 2012). An array's resolving power, which sets them, is the width of the
image's peak, not the error of its maximum: taken as they are, they weigh the data well below
what they tell. The factor's prior is uniform in its logarithm within NOISE_BOUNDS: the picks'
own uncertainties at most, a third of them at least. Down to a hundredth, the factor goes to
0.02-0.08 on a real line: the models' curves hug the picks, their 10th to 90th percentiles 3 % as
wide as the uncertainties, and the chains crawl on so sharp a posterior. A third widens that band
to about 13 % of the uncertainties, still inside them.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from disba import DispersionError
from disba._cps import surf96  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]

from sigpipe.base.dispersion_curve import DispersionCurve

NOISE_BOUNDS = (1 / 3, 1.0)
# A curve is inverted on at most this many points, evenly spaced in the logarithm of wavelength.
# Neighbouring picks of an image are not independent measurements (the image's resolution
# spreads each over its neighbours): a dense curve, taken point by point, would weigh far more
# than it tells, and the chains crawl on so sharp a posterior. Each point averages the picks of
# its wavelength band, with their mean uncertainty (errors that go together do not average out).
MAX_POINTS = 30
# A pick without an uncertainty gets this share of its velocity, before the noise factor.
FALLBACK_UNCERTAINTY = 0.1
# disba's Rayleigh root search (Dunkin's matrix) and its phase-velocity increment, in km/s: those
# of disba's PhaseDispersion, which the rest of sigpipe calls.
_RAYLEIGH = 2
_DC_KM_S = 0.005
# The half-space's thickness given to disba, which takes the last layer as a half-space anyway.
HALF_SPACE_M = 1_000.0


@dataclass(frozen=True, slots=True)
class Curve:
    """A picked curve, by increasing period as disba takes it; `order` puts its values back in
    the picked curve's order."""

    mode: int
    periods: np.ndarray  # s, increasing
    observed: np.ndarray  # m/s, along `periods`
    sigma: np.ndarray  # m/s, the picks' uncertainties, before the noise factor
    order: np.ndarray  # the picked curve's order, from `periods`'


def curves_of(
    dispersion_curves: Sequence[DispersionCurve], max_points: int | None = MAX_POINTS
) -> tuple[Curve, ...]:
    """The picked curves as the samplers take them, one per mode, by mode: each on at most
    `max_points` points (None: every pick), in the picked order when it is not resampled."""
    curves: list[Curve] = []
    for curve in sorted(dispersion_curves, key=lambda one: one.mode.number):
        fs = np.asarray(curve.fs, dtype=np.float64)
        vs = np.asarray(curve.vs, dtype=np.float64)
        sigma = (
            np.asarray(curve.vs_err, dtype=np.float64)
            if curve.vs_err is not None
            else FALLBACK_UNCERTAINTY * vs
        )
        if max_points is not None and fs.size > max_points:
            fs, vs, sigma = _by_wavelength(fs, vs, sigma, max_points)
        by_period = np.argsort(1 / fs)
        curves.append(
            Curve(
                mode=curve.mode.number,
                periods=(1 / fs)[by_period],
                observed=vs[by_period],
                sigma=np.maximum(sigma[by_period], 1e-3),
                order=np.argsort(by_period),
            )
        )
    return tuple(curves)


def _by_wavelength(
    fs: np.ndarray, vs: np.ndarray, sigma: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The picks averaged in `count` bands evenly spaced in the logarithm of wavelength (the
    empty bands left out): each band's mean frequency, velocity and uncertainty."""
    wavelength = np.log(vs / fs)
    edges = np.linspace(wavelength.min(), wavelength.max(), count + 1)
    band = np.clip(np.searchsorted(edges, wavelength, side="right") - 1, 0, count - 1)
    used = np.unique(band)

    def mean(values: np.ndarray) -> np.ndarray:
        return np.array([values[band == b].mean() for b in used])

    return mean(fs), mean(vs), mean(sigma)


def phase_velocities(
    curves: Sequence[Curve], thickness: np.ndarray, vs: np.ndarray, vp_vs: float
) -> list[np.ndarray] | None:
    """Each curve's phase velocities (m/s, by increasing period) for the layered model (m, m/s,
    top down; the last layer a half-space), or None where a mode does not exist over its
    curve's periods."""
    vp = vs * vp_vs
    rho = 0.32 * vp + 770.0  # kg/m3, Gardner's as vp_rho_from_vs
    d, a, b, r = thickness / 1_000, vp / 1_000, vs / 1_000, rho / 1_000
    predicted: list[np.ndarray] = []
    for curve in curves:
        try:
            c = surf96(curve.periods, d, a, b, r, curve.mode, 0, _RAYLEIGH, _DC_KM_S)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
        except DispersionError:
            return None
        c = np.asarray(c, dtype=np.float64)
        if not np.all(c > 0):
            return None
        predicted.append(c * 1_000)
    return predicted


def misfit(curves: Sequence[Curve], predicted: Sequence[np.ndarray]) -> float:
    """The sum of the squared residuals, each in its pick's uncertainty."""
    return float(
        sum(
            np.sum(((curve.observed - c) / curve.sigma) ** 2)
            for curve, c in zip(curves, predicted, strict=True)
        )
    )


def rms(curves: Sequence[Curve], predicted: Sequence[np.ndarray]) -> float:
    """The residuals' root mean square, m/s."""
    squares = sum(
        float(np.sum((curve.observed - c) ** 2)) for curve, c in zip(curves, predicted, strict=True)
    )
    return math.sqrt(squares / sum(curve.observed.size for curve in curves))


def log_likelihood(misfit_value: float, log_noise: float, n_points: int) -> float:
    """The likelihood's logarithm with the picks' uncertainties scaled by exp(`log_noise`)
    (its constant left out)."""
    return -n_points * log_noise - misfit_value / (2 * math.exp(2 * log_noise))


def allowed(vs: np.ndarray, least_ratio: float) -> bool:
    """Whether no layer's Vs falls below `least_ratio` of the one above it. Under a layer much
    stiffer than the one below, the lowest root of the dispersion equation, which the forward
    model takes for the fundamental mode, is a wave trapped in the soft layer: it matches slow
    picks while the surface records the stiff layer's, and such models draw the chains away."""
    return least_ratio <= 0 or bool(np.all(vs[1:] >= least_ratio * vs[:-1]))
