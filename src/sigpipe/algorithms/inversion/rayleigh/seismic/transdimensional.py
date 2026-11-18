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

The chains run in sigpipe-core (Rust, core/src/transdimensional.rs), which holds the moves'
probabilities (birth and death 0.12 each, an interface's step 0.15, an interface anywhere 0.06,
a Vs 0.29, the noise factor 0.08, every Vs and every depth 0.09 each), a new layer's Vs spread
(0.25 on the logarithm) and the acceptance the steps aim at (0.3). A seed gives the same chain
from run to run.

References:
- Green, P. J. (1995). Reversible jump Markov chain Monte Carlo computation and Bayesian model
  determination. Biometrika, 82(4), 711-732. https://doi.org/10.1093/biomet/82.4.711
- Bodin, T., Sambridge, M., Tkalcic, H., Arroucau, P., Gallagher, K., & Rawlinson, N. (2012).
  Transdimensional inversion of receiver functions and surface wave dispersion. Journal of
  Geophysical Research: Solid Earth, 117, B02301. https://doi.org/10.1029/2011JB008560
- Earl, D. J., & Deem, M. W. (2005). Parallel tempering: Theory, applications, and new
  perspectives. Physical Chemistry Chemical Physics, 7(23), 3910-3916.
  https://doi.org/10.1039/B509983H
"""

from dataclasses import dataclass

import numpy as np
import sigpipe_core

from .data import NOISE_BOUNDS, Curve

# A layer is at least this share of its top's depth thick (a gap in the logarithm of depth).
THINNEST = 0.15


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
    steps: dict[str, float]  # each move's step after the burn-in, of the first copy (log units)
    best_depths: np.ndarray  # the best model visited (least misfit), at any temperature
    best_vs: np.ndarray
    best_misfit: float


def run_chain(space: Space, settings: Settings, seed: int) -> Chain:
    """One chain and its hotter copies, from random models of the priors."""
    ran = sigpipe_core.transdimensional_chain(
        [
            (
                curve.mode,
                np.ascontiguousarray(curve.periods, dtype=np.float64),
                np.ascontiguousarray(curve.observed, dtype=np.float64),
                np.ascontiguousarray(curve.sigma, dtype=np.float64),
            )
            for curve in space.curves
        ],
        (float(space.vs_bounds[0]), float(space.vs_bounds[1])),
        (float(space.depth_bounds[0]), float(space.depth_bounds[1])),
        space.max_layers,
        space.least_ratio,
        space.vp_vs,
        NOISE_BOUNDS,
        THINNEST,
        settings.n_iterations,
        settings.n_burnin,
        settings.save_every,
        settings.temperatures,
        settings.hottest,
        seed,
    )
    proposed, accepted = ran["proposed"], ran["accepted"]
    swapped, tried = ran["swaps"]
    return Chain(
        layers=ran["layers"],
        depths=ran["depths"],
        vs=ran["vs"],
        noise=ran["noise"],
        predicted=tuple(
            values[:, curve.order]
            for values, curve in zip(ran["predicted"], space.curves, strict=True)
        ),
        rms=ran["rms"],
        acceptance={
            move: round(100 * accepted[move] / max(proposed[move], 1), 1) for move in proposed
        },
        accepted=round(100 * sum(accepted.values()) / max(sum(proposed.values()), 1), 2),
        swaps=round(100 * swapped / max(tried, 1), 1),
        steps=dict(ran["steps"]),
        best_depths=ran["best_depths"],
        best_vs=ran["best_vs"],
        best_misfit=ran["best_misfit"],
    )
