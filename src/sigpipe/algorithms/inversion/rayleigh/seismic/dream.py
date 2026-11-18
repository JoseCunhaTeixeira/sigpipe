"""Layered Vs models with the layers given (the fixed layering): each layer's Vs and thickness
sampled within its prior, or fixed; the chains run together (DREAM(ZS): ter Braak and Vrugt
2008, Vrugt 2016).

Each chain proposes along differences of past states of all chains, kept in an archive: the
proposals follow the posterior's trade-offs (a layer thicker and faster, or thinner and slower)
without a step to tune, and one in five takes a whole difference, a jump between the
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

The chains run in sigpipe-core (Rust, core/src/dream.rs), which holds DREAM(ZS)'s settings
(Vrugt 2016): up to 3 pairs of past states summed in a proposal, a jump every 5 iterations, a
snooker in 10 proposals, a state archived every 10 iterations, 3 crossover values, each
difference scaled by 1 +- 0.1, and 1e-6 added to each move on the unit cube. A seed gives the
same chains from run to run.

References:
- ter Braak, C. J. F., & Vrugt, J. A. (2008). Differential Evolution Markov Chain with snooker
  updater and fewer chains. Statistics and Computing, 18(4), 435-446.
  https://doi.org/10.1007/s11222-008-9104-9
- Vrugt, J. A., ter Braak, C. J. F., Diks, C. G. H., Robinson, B. A., Hyman, J. M., & Higdon, D.
  (2009). Accelerating Markov chain Monte Carlo simulation by differential evolution with
  self-adaptive randomized subspace sampling. International Journal of Nonlinear Sciences and
  Numerical Simulation, 10(3), 273-290. https://doi.org/10.1515/IJNSNS.2009.10.3.273
- Vrugt, J. A. (2016). Markov chain Monte Carlo simulation using the DREAM software package:
  Theory, concepts, and MATLAB implementation. Environmental Modelling & Software, 75, 273-316.
  https://doi.org/10.1016/j.envsoft.2015.08.013
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import sigpipe_core

from .data import (
    HALF_SPACE_M,
    NOISE_BOUNDS,
    Curve,
    allowed,
    misfit,
    phase_velocities,
    rms,
)


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

    def placement(
        self,
    ) -> tuple[list[float], list[float], list[int], list[int], list[int], list[int]]:
        """Where each value goes, as sigpipe-core takes it: the layers' Vs and thicknesses with
        the fixed ones in place (NaN where sampled), then the sampled Vs' places in the values
        and in the layers, and the sampled thicknesses'."""
        return (
            self._vs.tolist(),
            self._thickness.tolist(),
            self._vs_from.tolist(),
            self._vs_to.tolist(),
            self._thickness_from.tolist(),
            self._thickness_to.tolist(),
        )

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
    names_count = len(problem.names)
    low = np.append(np.log(problem.low), math.log(NOISE_BOUNDS[0]))
    high = np.append(np.log(problem.high), math.log(NOISE_BOUNDS[1]))
    span = high - low
    vs_template, thickness_template, vs_from, vs_to, thickness_from, thickness_to = (
        problem.placement()
    )
    ran = sigpipe_core.dream_chains(
        [
            (
                curve.mode,
                np.ascontiguousarray(curve.periods, dtype=np.float64),
                np.ascontiguousarray(curve.observed, dtype=np.float64),
                np.ascontiguousarray(curve.sigma, dtype=np.float64),
            )
            for curve in problem.curves
        ],
        np.asarray(problem.low, dtype=float).tolist(),
        np.asarray(problem.high, dtype=float).tolist(),
        vs_template,
        thickness_template,
        vs_from,
        vs_to,
        thickness_from,
        thickness_to,
        [i for i, name in enumerate(problem.names) if name.startswith("vs")],
        problem.least_ratio,
        problem.vp_vs,
        NOISE_BOUNDS,
        settings.n_iterations,
        settings.n_burnin,
        settings.save_every,
        settings.n_chains,
        seed,
    )

    def from_unit(u: np.ndarray) -> np.ndarray:
        """The values, then the noise factor's logarithm (rows of states on the unit cube)."""
        x = low + u * span
        return np.concatenate((np.exp(x[..., :names_count]), x[..., -1:]), axis=-1)

    chains: list[ChainResult] = []
    for chain in ran["chains"]:
        values = from_unit(chain["units"])
        # A typical accepted move, relative (on the logarithms), in each value's own unit.
        relative = (
            np.sqrt(chain["squared_moves"] / max(chain["accepted"], 1))[:names_count]
            * span[:names_count]
        )
        typical = (
            np.median(values[:, :names_count], axis=0) if len(values) else np.exp(low[:names_count])
        )
        chains.append(
            ChainResult(
                values=values[:, :names_count],
                noise=np.exp(values[:, -1]),
                predicted=tuple(chain["predicted"]),
                rms=chain["rms"],
                accepted=chain["accepted"],
                proposed=settings.n_iterations,
                steps=relative * typical,
            )
        )
    return Sampled(
        chains=tuple(chains),
        best_values=from_unit(ran["best_unit"])[:names_count],
        best_predicted=tuple(ran["best_predicted"]),
        crossover=tuple(ran["crossover"]),
    )
