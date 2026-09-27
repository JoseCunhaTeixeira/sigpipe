"""Layered Vs models whose number of layers the data choose: reversible-jump Markov chains
(Green 1995; for surface waves, Bodin et al. 2012), with parallel tempering.

The model: k layers (1 to `max_layers`, the half-space included), their interfaces' depths and
each layer's Vs, and the noise factor of the picks' uncertainties (data.py). The priors:

- k uniform, the models breaking a constraint below then left out: the constraints remove a
  growing share of the models as layers are added, a parsimony on top of the data's own (made
  uniform over the models allowed instead, the prior on the count favours as many layers as
  allowed: each constraint's toll outweighs the Occam factor of a new layer);
- the interfaces uniform in the logarithm of depth, between `depth_min` and `depth_max`: a
  shallow interface as likely as a deep one relative to its depth, as the wavelengths resolve
  them; a layer at least `THINNEST` of its top's depth thick;
- each Vs uniform in its logarithm within `vs_bounds`, each at least `least_ratio` of the one
  above it (data.allowed);
- the noise factor uniform in its logarithm (data.NOISE_BOUNDS).

A chain moves by births (an interface anywhere, the new layer's Vs near the one it splits from)
and deaths (the reverse), an interface's or a layer's own move, an interface drawn anew between
its neighbours (between two places the data leave it, at once), a shift of every Vs together or
a stretch of every depth together (the trade-offs of depth and velocity), and the noise
factor's move; each move accepted with its exact ratio. Hotter copies of each chain, whose
likelihood is flattened, cross between the posterior's modes and exchange models with it
(parallel tempering). The steps adapt during the burn-in, then stay.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .data import NOISE_BOUNDS, Curve, allowed, log_likelihood, misfit, phase_velocities, rms

# A layer is at least this share of its top's depth thick (a gap in the logarithm of depth).
THINNEST = 0.15
# Each move's probability: birth, death, an interface's step, an interface anywhere between its
# neighbours, a Vs, the noise factor, every Vs, every depth.
_MOVES = ("birth", "death", "interface", "relocate", "vs", "noise", "shift", "stretch")
_WEIGHTS = np.array([0.12, 0.12, 0.15, 0.06, 0.29, 0.08, 0.09, 0.09])
# The new layer's Vs around the one it splits from, on the logarithm.
BIRTH_SPREAD = 0.25
# The local moves' acceptance aimed at during the burn-in.
TARGET = 0.3
_LOG_2PI = math.log(2 * math.pi)
# A candidate's predicted curves, misfit and log-likelihood.
_Evaluated = tuple[list[np.ndarray], float, float]


@dataclass(frozen=True, slots=True)
class Space:
    """The data and the priors."""

    curves: tuple[Curve, ...]
    vs_bounds: tuple[float, float]  # m/s
    depth_bounds: tuple[float, float]  # m: the shallowest and deepest interface
    max_layers: int  # the half-space included
    least_ratio: float  # a layer's Vs over the one above it, at least (0: any)
    vp_vs: float

    @property
    def n_points(self) -> int:
        return sum(curve.observed.size for curve in self.curves)


@dataclass(frozen=True, slots=True)
class Settings:
    n_iterations: int
    n_burnin: int
    save_every: int
    temperatures: int  # copies of the chain, the first at the posterior
    hottest: float  # the hottest copy's temperature


@dataclass(frozen=True, slots=True)
class Chain:
    """One chain's kept models (padded with NaN beyond their layers) and how it ran."""

    layers: np.ndarray  # (kept,) layers, the half-space included
    depths: np.ndarray  # (kept, max_layers - 1) m, the interfaces' depths
    vs: np.ndarray  # (kept, max_layers) m/s
    noise: np.ndarray  # (kept,) the noise factor
    predicted: tuple[np.ndarray, ...]  # per curve: (kept, points), in the picked curve's order
    rms: np.ndarray  # (kept,) m/s
    acceptance: dict[str, float]  # each move's, %, of the first copy
    accepted: float  # every move's, %, of the first copy
    swaps: float  # the exchanges accepted, %
    best_depths: np.ndarray  # the best model visited (least misfit), at any temperature
    best_vs: np.ndarray
    best_misfit: float


class _Copy:
    """One tempered copy: its model (log depths, log Vs, log noise), fit and steps."""

    def __init__(self, depths: np.ndarray, vs: np.ndarray, log_noise: float) -> None:
        self.log_depths = np.log(depths)
        self.log_vs = np.log(vs)
        self.log_noise = log_noise
        self.misfit = math.inf
        self.likelihood = -math.inf
        self.predicted: list[np.ndarray] = []
        self.steps = {"interface": 0.1, "vs": 0.05, "noise": 0.1, "shift": 0.02, "stretch": 0.02}


def run_chain(space: Space, settings: Settings, seed: int) -> Chain:
    """One chain and its hotter copies, from random models of the priors."""
    rng = np.random.default_rng(seed)
    w_low, w_high = (math.log(bound) for bound in space.vs_bounds)
    w_span = w_high - w_low
    u_low, u_high = (math.log(bound) for bound in space.depth_bounds)
    gap = math.log(1 + THINNEST)
    s_low, s_high = (math.log(bound) for bound in NOISE_BOUNDS)
    n_points = space.n_points
    k_count = max(1, settings.temperatures)
    betas = 1 / np.geomspace(1.0, settings.hottest, k_count) if k_count > 1 else np.ones(1)
    cumulative = np.cumsum(_WEIGHTS / _WEIGHTS.sum())

    def evaluate(log_depths: np.ndarray, log_vs: np.ndarray, log_noise: float) -> _Evaluated | None:
        """The candidate's predicted curves, misfit and likelihood, or None outside the priors."""
        if log_depths.size and (
            log_depths[0] < u_low or log_depths[-1] > u_high or np.any(np.diff(log_depths) < gap)
        ):
            return None
        if np.any(log_vs < w_low) or np.any(log_vs > w_high):
            return None
        vs = np.exp(log_vs)
        if not allowed(vs, space.least_ratio):
            return None
        depths = np.exp(log_depths)
        thickness = np.append(np.diff(np.concatenate(([0.0], depths))), 1_000.0)
        predicted = phase_velocities(space.curves, thickness, vs, space.vp_vs)
        if predicted is None:
            return None
        value = misfit(space.curves, predicted)
        return predicted, value, log_likelihood(value, log_noise, n_points)

    copies: list[_Copy] = []
    for _ in range(k_count):
        copies.append(_start(space, rng, evaluate))
    best = min(copies, key=lambda copy: copy.misfit)
    best_misfit = best.misfit
    best_depths, best_vs = np.exp(best.log_depths), np.exp(best.log_vs)

    proposed = dict.fromkeys(_MOVES, 0)
    accepted = dict.fromkeys(_MOVES, 0)
    swaps = [0, 0]
    kept: list[tuple[np.ndarray, np.ndarray, float, list[np.ndarray], float]] = []

    for iteration in range(1, settings.n_iterations + 1):
        burning = iteration <= settings.n_burnin
        gain = 1.0 / (1 + iteration / 100) ** 0.6
        for t, copy in enumerate(copies):
            move = _MOVES[int(np.searchsorted(cumulative, rng.random()))]
            k = copy.log_vs.size
            log_depths, log_vs, log_noise = copy.log_depths, copy.log_vs, copy.log_noise
            extra = 0.0  # the log prior and proposal ratio
            if move == "birth":
                if k >= space.max_layers:
                    continue
                u = rng.uniform(u_low, u_high)
                j = int(np.searchsorted(log_depths, u))
                eps = rng.standard_normal()
                log_depths = np.insert(log_depths, j, u)
                log_vs = np.insert(log_vs, j + 1, log_vs[j] + BIRTH_SPREAD * eps)
                extra = math.log(BIRTH_SPREAD) - math.log(w_span) + 0.5 * _LOG_2PI + 0.5 * eps**2
            elif move == "death":
                if k <= 1:
                    continue
                i = int(rng.integers(k - 1))
                eps = (log_vs[i + 1] - log_vs[i]) / BIRTH_SPREAD
                log_depths = np.delete(log_depths, i)
                log_vs = np.delete(log_vs, i + 1)
                extra = -math.log(BIRTH_SPREAD) + math.log(w_span) - 0.5 * _LOG_2PI - 0.5 * eps**2
            elif move == "interface":
                if k <= 1:
                    continue
                i = int(rng.integers(k - 1))
                log_depths = log_depths.copy()
                log_depths[i] += copy.steps["interface"] * rng.standard_normal()
            elif move == "relocate":
                # Anywhere between its neighbours, whatever its place: an interface whose depth
                # the data leave between two places crosses to the other at once.
                if k <= 1:
                    continue
                i = int(rng.integers(k - 1))
                above = log_depths[i - 1] + gap if i > 0 else u_low
                below = log_depths[i + 1] - gap if i < k - 2 else u_high
                if below <= above:
                    continue
                log_depths = log_depths.copy()
                log_depths[i] = rng.uniform(above, below)
            elif move == "vs":
                j = int(rng.integers(k))
                log_vs = log_vs.copy()
                log_vs[j] = _reflect(
                    log_vs[j] + copy.steps["vs"] * rng.standard_normal(), w_low, w_high
                )
            elif move == "noise":
                log_noise = _reflect(
                    log_noise + copy.steps["noise"] * rng.standard_normal(), s_low, s_high
                )
            elif move == "shift":
                log_vs = log_vs + copy.steps["shift"] * rng.standard_normal()
            else:  # stretch
                if k <= 1:
                    continue
                log_depths = log_depths + copy.steps["stretch"] * rng.standard_normal()
            if t == 0:
                proposed[move] += 1
            result = evaluate(log_depths, log_vs, log_noise)
            ok = False
            if result is not None:
                predicted, value, likelihood = result
                ok = (
                    math.log(rng.random() + 1e-300)
                    < betas[t] * (likelihood - copy.likelihood) + extra
                )
                if ok:
                    copy.log_depths, copy.log_vs, copy.log_noise = log_depths, log_vs, log_noise
                    copy.predicted, copy.misfit, copy.likelihood = predicted, value, likelihood
                    if value < best_misfit:
                        best_misfit = value
                        best_depths, best_vs = np.exp(log_depths), np.exp(log_vs)
            if t == 0 and ok:
                accepted[move] += 1
            if burning and move in copy.steps:
                copy.steps[move] *= math.exp(gain * ((1.0 if ok else 0.0) - TARGET))

        # Exchanges between neighbouring copies, the even pairs then the odd ones in turn.
        for a in range(iteration % 2, k_count - 1, 2):
            swaps[1] += 1
            hot, cold = copies[a + 1], copies[a]
            if math.log(rng.random() + 1e-300) < (betas[a] - betas[a + 1]) * (
                hot.likelihood - cold.likelihood
            ):
                swaps[0] += 1
                # The models change places; each copy keeps its temperature's steps.
                for name in (
                    "log_depths",
                    "log_vs",
                    "log_noise",
                    "misfit",
                    "likelihood",
                    "predicted",
                ):
                    first, second = getattr(cold, name), getattr(hot, name)
                    setattr(cold, name, second)
                    setattr(hot, name, first)

        if not burning and (iteration - settings.n_burnin) % settings.save_every == 0:
            cold = copies[0]
            kept.append(
                (
                    np.exp(cold.log_depths),
                    np.exp(cold.log_vs),
                    math.exp(cold.log_noise),
                    cold.predicted,
                    cold.misfit,
                )
            )

    count = len(kept)
    depths = np.full((count, space.max_layers - 1), np.nan)
    vs = np.full((count, space.max_layers), np.nan)
    for row, (d, v, *_rest) in enumerate(kept):
        depths[row, : d.size] = d
        vs[row, : v.size] = v
    return Chain(
        layers=np.array([v.size for _, v, *_rest in kept], dtype=int),
        depths=depths,
        vs=vs,
        noise=np.array([noise for _, _, noise, _, _ in kept]),
        predicted=tuple(
            np.array([p[i][curve.order] for *_ignored, p, _ in kept]).reshape(count, -1)
            for i, curve in enumerate(space.curves)
        ),
        rms=np.array([rms(space.curves, p) for *_ignored, p, _ in kept]),
        acceptance={
            move: round(100 * accepted[move] / max(proposed[move], 1), 1) for move in _MOVES
        },
        accepted=round(100 * sum(accepted.values()) / max(sum(proposed.values()), 1), 2),
        swaps=round(100 * swaps[0] / max(swaps[1], 1), 1),
        best_depths=best_depths,
        best_vs=best_vs,
        best_misfit=best_misfit,
    )


def _reflect(x: float, low: float, high: float) -> float:
    """`x` folded back within [low, high], as by mirrors at its ends."""
    span = high - low
    y = (x - low) % (2 * span)
    return low + (y if y <= span else 2 * span - y)


def _start(
    space: Space,
    rng: np.random.Generator,
    evaluate: Callable[[np.ndarray, np.ndarray, float], _Evaluated | None],
) -> _Copy:
    """A model of the priors with few layers, its Vs increasing, whose modes exist."""
    u_low, u_high = (math.log(bound) for bound in space.depth_bounds)
    w_low, w_high = (math.log(bound) for bound in space.vs_bounds)
    for _ in range(10_000):
        k = int(rng.integers(1, min(3, space.max_layers) + 1))
        log_depths = np.sort(rng.uniform(u_low, u_high, k - 1))
        log_vs = np.sort(rng.uniform(w_low, w_high, k))
        log_noise = math.log(NOISE_BOUNDS[1])
        copy = _Copy(np.exp(log_depths), np.exp(log_vs), log_noise)
        result = evaluate(log_depths, log_vs, log_noise)
        if result is not None:
            copy.predicted, copy.misfit, copy.likelihood = result
            return copy
    raise ValueError(
        "No model of the priors has the picked modes over their frequencies, within the Vs "
        "ratios allowed: widen the Vs bounds."
    )
