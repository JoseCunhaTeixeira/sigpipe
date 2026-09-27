"""Layered Vs models with the layers given (the fixed layering): each layer's Vs and thickness
sampled within its prior, or fixed; the chains run together (DREAM(ZS): ter Braak and Vrugt
2008, Vrugt 2016).

Each chain proposes along differences of past states of all chains, kept in an archive: the
proposals follow the posterior's trade-offs (a layer thicker and faster, or thinner and slower)
without a step to tune, and one in JUMP_EVERY takes a whole difference, a jump between the
posterior's modes. A proposal moves a share of the parameters, the shares that move the chains
furthest chosen more often during the burn-in; one in ten moves along the line through an
archived state (snooker), across the posterior. Once the burn-in is over, the differences are
drawn from the states archived since its middle: the priors' draws and the chains' first
wanderings, far wider than the posterior, would have most proposals land far off it.

The values are sampled as their logarithms: a layer thicker and faster fits as one thinner and
slower (their travel time), a curved trade-off that turns nearly straight, along which the
differences propose; the priors stay uniform in the values (a Jacobian term). Every value moves
within its prior's bounds, reflected at them, so that the proposals stay symmetric. The data,
the forward model, the noise factor and the Vs-drop limit are data.py's.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .data import (
    HALF_SPACE_M,
    NOISE_BOUNDS,
    Curve,
    allowed,
    log_likelihood,
    misfit,
    phase_velocities,
    rms,
)

# DREAM(ZS)'s settings (Vrugt 2016): pairs of past states summed in a proposal, the jump between
# modes, the snooker's share, the archive's pace, the crossover values, and the noise added.
PAIRS = 3
JUMP_EVERY = 5
SNOOKER = 0.1
ARCHIVE_EVERY = 10
CROSSOVERS = 3
_SPREAD = 0.1  # each difference scaled by 1 +- this
_JITTER = 1e-6  # added to each move, on the unit cube


@dataclass(frozen=True)
class Problem:
    """The curves, the parameters sampled and their priors, and the values fixed."""

    curves: tuple[Curve, ...]
    names: tuple[str, ...]  # the parameters sampled: vs1, ..., thick1, ...
    low: np.ndarray  # each one's prior bounds
    high: np.ndarray
    fixed: Mapping[str, float]  # the values not sampled, by name
    n_layers: int  # the half-space included
    vp_vs: float
    least_ratio: float = 0.0  # a layer's Vs over the one above it, at least (0: any)
    # Where each value goes: the layers' Vs and thicknesses with the fixed ones in place, and
    # the sampled ones' places in them.
    _vs: np.ndarray = field(init=False, repr=False)
    _thickness: np.ndarray = field(init=False, repr=False)
    _vs_from: np.ndarray = field(init=False, repr=False)
    _vs_to: np.ndarray = field(init=False, repr=False)
    _thickness_from: np.ndarray = field(init=False, repr=False)
    _thickness_to: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        vs = np.array([self.fixed.get(f"vs{i + 1}", np.nan) for i in range(self.n_layers)])
        thickness = np.array(
            [self.fixed.get(f"thick{i + 1}", np.nan) for i in range(self.n_layers - 1)]
            + [HALF_SPACE_M]
        )
        places = {name: i for i, name in enumerate(self.names)}
        vs_pairs = [
            (places[f"vs{i + 1}"], i) for i in range(self.n_layers) if f"vs{i + 1}" in places
        ]
        thick_pairs = [
            (places[f"thick{i + 1}"], i)
            for i in range(self.n_layers - 1)
            if f"thick{i + 1}" in places
        ]
        object.__setattr__(self, "_vs", vs)
        object.__setattr__(self, "_thickness", thickness)
        object.__setattr__(self, "_vs_from", np.array([a for a, _ in vs_pairs], dtype=int))
        object.__setattr__(self, "_vs_to", np.array([b for _, b in vs_pairs], dtype=int))
        object.__setattr__(
            self, "_thickness_from", np.array([a for a, _ in thick_pairs], dtype=int)
        )
        object.__setattr__(self, "_thickness_to", np.array([b for _, b in thick_pairs], dtype=int))

    @property
    def n_points(self) -> int:
        return sum(curve.observed.size for curve in self.curves)

    def layers(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each layer's thickness and Vs (m, m/s), top down, from the values sampled."""
        vs = self._vs.copy()
        vs[self._vs_to] = values[self._vs_from]
        thickness = self._thickness.copy()
        thickness[self._thickness_to] = values[self._thickness_from]
        return thickness, vs

    def predict(self, values: np.ndarray) -> list[np.ndarray] | None:
        """Each curve's phase velocities (m/s) for the model `values`, or None where the model
        is outside the prior or a mode does not exist over its curve's periods."""
        thickness, vs = self.layers(values)
        if not allowed(vs, self.least_ratio):
            return None
        return phase_velocities(self.curves, thickness, vs, self.vp_vs)

    def misfit(self, predicted: Sequence[np.ndarray]) -> float:
        return misfit(self.curves, predicted)

    def rms(self, predicted: Sequence[np.ndarray]) -> float:
        return rms(self.curves, predicted)


@dataclass(frozen=True, slots=True)
class Settings:
    n_iterations: int  # each chain's
    n_burnin: int
    save_every: int
    n_chains: int


@dataclass(frozen=True, slots=True)
class ChainResult:
    """One chain's kept models, their noise factor and predicted curves, and how it ran."""

    values: np.ndarray  # (kept, parameters)
    noise: np.ndarray  # (kept,)
    predicted: tuple[np.ndarray, ...]  # per curve: (kept, points), by increasing period
    rms: np.ndarray  # (kept,) m/s
    accepted: int  # moves accepted, the burn-in included
    proposed: int
    steps: np.ndarray  # each parameter's typical accepted move, in its own unit


@dataclass(frozen=True, slots=True)
class Sampled:
    """The chains, and the best model any of them visited."""

    chains: tuple[ChainResult, ...]
    best_values: np.ndarray
    best_predicted: tuple[np.ndarray, ...]
    crossover: tuple[float, ...]  # each crossover value's final probability


def sample(problem: Problem, settings: Settings, seed: int) -> Sampled:
    """The chains together, from models of the priors (DREAM(ZS))."""
    rng = np.random.default_rng(seed)
    names_count = len(problem.names)
    dims = names_count + 1  # the noise factor last, as its logarithm
    low = np.append(np.log(problem.low), math.log(NOISE_BOUNDS[0]))
    high = np.append(np.log(problem.high), math.log(NOISE_BOUNDS[1]))
    span = high - low
    n_points = problem.n_points
    n_chains = settings.n_chains

    def from_unit(u: np.ndarray) -> np.ndarray:
        """The values, then the noise factor's logarithm."""
        x = low + u * span
        return np.append(np.exp(x[:names_count]), x[-1])

    def log_posterior(misfit_value: float, u: np.ndarray) -> float:
        """The likelihood's logarithm and the priors', uniform in the values (their
        logarithms' Jacobian) and in the noise factor's logarithm."""
        x = low + u * span
        log_noise = float(x[-1])
        likelihood = log_likelihood(misfit_value, log_noise, n_points)
        return likelihood + float(np.sum(x[:names_count]))

    # The archive: draws of the priors to begin with, then the chains' states.
    size = max(10 * dims, 100)
    capacity = size + n_chains * (settings.n_iterations // ARCHIVE_EVERY + 1)
    archive = np.empty((capacity, dims))
    for i in range(size):
        archive[i] = np.append(_prior_draw(problem, rng), rng.random())

    units: list[np.ndarray] = []
    misfits: list[float] = []
    predictions: list[list[np.ndarray]] = []
    lls: list[float] = []
    for _ in range(n_chains):
        u, predicted = _start(problem, rng)
        units.append(np.append(u, 1.0))  # the noise factor starts at the picks' own
        misfits.append(problem.misfit(predicted))
        predictions.append(predicted)
        lls.append(log_posterior(misfits[-1], units[-1]))

    best = int(np.argmin(misfits))
    best_misfit, best_unit, best_predicted = misfits[best], units[best].copy(), predictions[best]
    accepted = [0] * n_chains
    squared_moves = [np.zeros(dims) for _ in range(n_chains)]
    kept: list[list[tuple[np.ndarray, list[np.ndarray]]]] = [[] for _ in range(n_chains)]
    # The crossover values, their probabilities, and each one's squared jumps and uses, which
    # adapt the probabilities during the burn-in.
    crossover = np.arange(1, CROSSOVERS + 1) / CROSSOVERS
    p_crossover = np.full(CROSSOVERS, 1 / CROSSOVERS)
    jumped = np.zeros(CROSSOVERS)
    used = np.zeros(CROSSOVERS)

    first = 0  # the archive's first state proposals draw from
    for iteration in range(1, settings.n_iterations + 1):
        burning = iteration <= settings.n_burnin
        if iteration == settings.n_burnin // 2 + 1:
            middle = size
        if iteration == settings.n_burnin + 1 and size - middle > 4 * PAIRS:  # pyright: ignore[reportPossiblyUnboundVariable]
            first = middle  # pyright: ignore[reportPossiblyUnboundVariable]
        spread = np.std(np.array(units), axis=0) + 1e-12  # the chains', per parameter
        for c in range(n_chains):
            m = -1
            if rng.random() < SNOOKER:
                candidate, factor = _snooker(units[c], archive[first:size], rng)
            else:
                m = int(rng.choice(CROSSOVERS, p=p_crossover))
                mask = rng.random(dims) < crossover[m]
                if not mask.any():
                    mask[int(rng.integers(dims))] = True
                moved = int(mask.sum())
                pairs = int(rng.integers(1, PAIRS + 1))
                jump = iteration % JUMP_EVERY == 0
                gamma = 1.0 if jump else 2.38 / math.sqrt(2 * pairs * moved)
                picks = first + rng.choice(size - first, 2 * pairs, replace=False)
                difference = archive[picks[:pairs]].sum(axis=0) - archive[picks[pairs:]].sum(axis=0)
                step = np.zeros(dims)
                scale = 1 + rng.uniform(-_SPREAD, _SPREAD, moved)
                step[mask] = gamma * scale * difference[mask] + _JITTER * rng.standard_normal(moved)
                candidate, factor = _reflect(units[c] + step), 0.0
            z = from_unit(candidate)
            predicted = problem.predict(z[:names_count])
            accept = False
            if predicted is not None:
                misfit = problem.misfit(predicted)
                ll = log_posterior(misfit, candidate)
                accept = math.log(rng.random() + 1e-300) < ll - lls[c] + factor
            if m >= 0 and burning:
                used[m] += 1
                if accept:
                    jumped[m] += float(np.sum(((candidate - units[c]) / spread) ** 2))
            if accept and predicted is not None:
                squared_moves[c] += (candidate - units[c]) ** 2
                accepted[c] += 1
                units[c], misfits[c], predictions[c], lls[c] = candidate, misfit, predicted, ll  # pyright: ignore[reportPossiblyUnboundVariable]
                if misfit < best_misfit:  # pyright: ignore[reportPossiblyUnboundVariable]
                    best_misfit, best_unit, best_predicted = misfit, candidate.copy(), predicted  # pyright: ignore[reportPossiblyUnboundVariable]
            if not burning and (iteration - settings.n_burnin) % settings.save_every == 0:
                kept[c].append((units[c].copy(), predictions[c]))
        if burning and iteration % 10 == 0 and used.all():
            # The crossover values that move the chains furthest per use, more often.
            rates = jumped / used
            if rates.sum() > 0:
                p_crossover = np.maximum(rates / rates.sum(), 0.05)
                p_crossover /= p_crossover.sum()
        if iteration % ARCHIVE_EVERY == 0:
            for c in range(n_chains):
                archive[size] = units[c]
                size += 1

    chains: list[ChainResult] = []
    for c in range(n_chains):
        values = np.array([from_unit(u) for u, _ in kept[c]]).reshape(-1, dims)
        # A typical accepted move, relative (on the logarithms), in each value's own unit.
        relative = (
            np.sqrt(squared_moves[c] / max(accepted[c], 1))[:names_count] * span[:names_count]
        )
        typical = (
            np.median(values[:, :names_count], axis=0) if len(values) else np.exp(low[:names_count])
        )
        chains.append(
            ChainResult(
                values=values[:, :names_count],
                noise=np.exp(values[:, -1]),
                predicted=tuple(
                    np.array([p[i] for _, p in kept[c]]).reshape(len(kept[c]), -1)
                    for i in range(len(problem.curves))
                ),
                rms=np.array([problem.rms(p) for _, p in kept[c]]),
                accepted=accepted[c],
                proposed=settings.n_iterations,
                steps=relative * typical,
            )
        )
    return Sampled(
        chains=tuple(chains),
        best_values=from_unit(best_unit)[:names_count],
        best_predicted=tuple(best_predicted),
        crossover=tuple(float(p) for p in p_crossover),
    )


def _snooker(
    x: np.ndarray, archive: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """A move along the line through `x` and an archived state, by the projected difference of
    two others, and its log acceptance factor (ter Braak and Vrugt 2008)."""
    z, z1, z2 = archive[rng.choice(len(archive), 3, replace=False)]
    direction = x - z
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return x.copy(), -math.inf
    unit = direction / norm
    gamma = rng.uniform(1.2, 2.2)
    candidate = x + gamma * (float(np.dot(z1 - z2, unit))) * unit
    if np.any(candidate < 0) or np.any(candidate > 1):
        return x.copy(), -math.inf  # outside the priors: no reflection, which would bend the line
    after = float(np.linalg.norm(candidate - z))
    if after < 1e-12:
        return x.copy(), -math.inf
    return candidate, (len(x) - 1) * math.log(after / norm)


def _reflect(u: np.ndarray) -> np.ndarray:
    """`u` folded back into the unit cube, as by mirrors at its faces."""
    u = np.mod(u, 2.0)
    return np.where(u > 1.0, 2.0 - u, u)


def _prior_draw(problem: Problem, rng: np.random.Generator) -> np.ndarray:
    """A model of the priors in the unit cube, its Vs increasing where their bounds allow."""
    values = rng.uniform(problem.low, problem.high)
    vs_index = [i for i, name in enumerate(problem.names) if name.startswith("vs")]
    if problem.least_ratio > 0 and vs_index:
        values[vs_index] = np.clip(
            np.sort(values[vs_index]), problem.low[vs_index], problem.high[vs_index]
        )
    return np.log(values / problem.low) / np.log(problem.high / problem.low)


def _start(problem: Problem, rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]]:
    """A model of the priors (unit cube) whose modes exist over the curves' periods, and its
    predicted curves."""
    for _ in range(10_000):
        u = _prior_draw(problem, rng)
        predicted = problem.predict(problem.low * (problem.high / problem.low) ** u)
        if predicted is not None:
            return u, predicted
    raise ValueError(
        "No model of the priors has the picked modes over their frequencies, within the Vs "
        "ratios allowed: check that each layer's bounds hold the curves."
    )
