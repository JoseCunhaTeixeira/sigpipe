"""The seismic inversion: layered Vs models sampled against picked phase-velocity curves, the
layers chosen by the data (transdimensional.py) or given (dream.py), and the models it keeps.

Both samplers keep models as layers (sigpipe.base.inversion.LayeredSamples). From them:
- the ensemble model, each depth's median and spread of the kept models' Vs;
- the median model, the kept model nearest the ensemble (least mean squared difference of the
  logarithm of Vs over the depths): a model the chains kept, which fits as they do, where the
  median of each layer's values would glue different models' layers together;
- the best model, the least misfit of every model the chains visited;
- the smooth best and median, their layers' steps eased (VelocityModel.smoothed).
"""

import math
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from sigpipe.base.coordinate import Coordinate
from sigpipe.base.dispersion_curve import DispersionCurve, DispersionCurvesImage
from sigpipe.base.inversion import InversionResult, LayeredSamples
from sigpipe.base.velocity_model import VelocityModel
from sigpipe.workers import one_thread_each

from . import dream, transdimensional
from .data import Curve, curves_of, phase_velocities
from .forward import vp_rho_from_vs
from .parameters import SAVE_EVERY, InversionParameters

# The free layering's tempered copies of each chain, and the hottest one's temperature.
TEMPERATURES = 5
HOTTEST = 100.0
# The medians' distance is measured every so many rows of the depth grid.
_EVERY = 10


@dataclass(frozen=True, slots=True)
class _Run:
    """What either sampler kept, in one form."""

    profiles: LayeredSamples
    samples: dict[str, np.ndarray]
    rms: np.ndarray
    best: tuple[np.ndarray, np.ndarray]  # interfaces' depths, layers' Vs
    acceptance: tuple[float, ...]
    steps: dict[str, float]
    log: str
    # The layers chosen by the data: each move's acceptance and the exchanges between tempered
    # copies, %, the medians of the chains.
    moves: dict[str, float] = field(default_factory=dict)
    exchanges: float | None = None


def inversion_mcmc(
    dispersion_curves: DispersionCurvesImage,
    position: Coordinate,
    *,
    Vp_Vs_ratio: float = 1.77,
    dz: float = 0.01,
    chain_jobs: int = 1,
    seed: int | None = None,
    **parameters: object,
) -> InversionResult:
    """The layered Vs models that fit `dispersion_curves`, sampled by Markov chains.

    `parameters` are InversionParameters' fields, which validate them: the layering (the data
    choosing the layers within bounds, or the layers given), the Vs drop allowed and the chains'
    effort. The free layering's chains run in `chain_jobs` processes (1: one after the other, in
    this process, as when a caller already runs windows in parallel); the fixed layering's run
    together in this one. Vp and density follow Vs by `Vp_Vs_ratio`; the smooth and ensemble
    models are sampled every `dz` metres. `seed` makes a run repeatable.
    """
    settings = InversionParameters.model_validate(parameters)
    if len(dispersion_curves) == 0:
        raise ValueError("At least one dispersion curve must be provided for inversion.")
    picked: list[DispersionCurve] = list(dispersion_curves)
    modes = [curve.mode.number for curve in picked]
    if len(set(modes)) != len(modes):
        raise ValueError("All dispersion curves must have a different mode.")
    if chain_jobs < 1:
        raise ValueError(f"chain_jobs must be at least 1, not {chain_jobs}")

    settings = settings.resolved(
        np.concatenate([np.asarray(curve.fs, dtype=float) for curve in picked]),
        np.concatenate([np.asarray(curve.vs, dtype=float) for curve in picked]),
    )
    curves = curves_of(picked)
    seeds = np.random.SeedSequence(seed).generate_state(settings.n_chains + 1).tolist()
    run = (
        _fixed(settings, curves, Vp_Vs_ratio, seeds[0])
        if settings.layering == "fixed"
        else _free(settings, curves, Vp_Vs_ratio, seeds[1:], chain_jobs)
    )
    return _result(settings, run, picked, Vp_Vs_ratio, dz, position)


def _fixed(
    settings: InversionParameters, curves: tuple[Curve, ...], vp_vs: float, seed: int
) -> _Run:
    """The layers given, sampled together by DREAM(ZS)."""
    fixed = settings.fixed()
    bounds = {
        f"vs{i + 1}": (layer.vs_min, layer.vs_max) for i, layer in enumerate(settings.vs_layers)
    } | {
        f"thick{i + 1}": (layer.thickness_min, layer.thickness_max)
        for i, layer in enumerate(settings.thickness_layers)
    }
    bounds = {name: prior for name, prior in bounds.items() if name not in fixed}
    problem = dream.Problem(
        curves=curves,
        names=tuple(bounds),
        low=np.array([low for low, _ in bounds.values()], dtype=float),
        high=np.array([high for _, high in bounds.values()], dtype=float),
        fixed=fixed,
        n_layers=settings.n_layers,
        vp_vs=vp_vs,
        least_ratio=settings.least_ratio,
    )
    sampled = dream.sample(
        problem,
        dream.Settings(
            n_iterations=settings.n_iterations,
            n_burnin=settings.n_burnin_iterations,
            save_every=SAVE_EVERY,
            n_chains=settings.n_chains,
        ),
        seed,
    )
    values = np.concatenate([chain.values for chain in sampled.chains])
    layered = [problem.layers(row) for row in values]
    thickness = np.array([t for t, _ in layered])
    vs = np.array([v for _, v in layered])
    samples = {f"vs{i + 1}": vs[:, i] for i in range(settings.n_layers)}
    samples |= {f"thick{i + 1}": thickness[:, i] for i in range(settings.n_layers - 1)}
    samples["noise"] = np.concatenate([chain.noise for chain in sampled.chains])
    best_thickness, best_vs = problem.layers(sampled.best_values)
    acceptance = tuple(
        round(100 * chain.accepted / max(chain.proposed, 1), 2) for chain in sampled.chains
    )
    steps = np.mean([chain.steps for chain in sampled.chains], axis=0)
    log = (
        f"DREAM(ZS), {settings.n_chains} chains of {settings.n_iterations:,} iterations "
        f"({settings.n_burnin_iterations:,} burn-in), {settings.n_layers} layers given; "
        f"acceptance {', '.join(f'{rate:g} %' for rate in acceptance)}; crossover "
        f"probabilities {', '.join(f'{p:.2f}' for p in sampled.crossover)}; noise factor, "
        f"median {float(np.median(samples['noise'])):.3g}.\n"
    )
    return _Run(
        profiles=LayeredSamples(
            depths=np.cumsum(thickness[:, :-1], axis=1),
            vs=vs,
            n_chains=settings.n_chains,
        ),
        samples=samples,
        rms=np.concatenate([chain.rms for chain in sampled.chains]),
        best=(np.cumsum(best_thickness[:-1]), best_vs),
        acceptance=acceptance,
        steps={
            name: round(float(step), 4) for name, step in zip(problem.names, steps, strict=True)
        },
        log=log,
    )


def _free(
    settings: InversionParameters,
    curves: tuple[Curve, ...],
    vp_vs: float,
    seeds: Sequence[int],
    chain_jobs: int,
) -> _Run:
    """The layers chosen by the data: one reversible-jump chain each (and its tempered copies),
    in `chain_jobs` processes."""
    free = settings.free
    assert free.vs_min is not None and free.vs_max is not None  # resolved
    assert free.depth_min is not None and free.depth_max is not None
    space = transdimensional.Space(
        curves=curves,
        vs_bounds=(free.vs_min, free.vs_max),
        depth_bounds=(free.depth_min, free.depth_max),
        max_layers=free.max_layers,
        least_ratio=settings.least_ratio,
        vp_vs=vp_vs,
    )
    run_settings = transdimensional.Settings(
        n_iterations=settings.n_iterations,
        n_burnin=settings.n_burnin_iterations,
        save_every=SAVE_EVERY,
        temperatures=TEMPERATURES,
        hottest=HOTTEST,
    )
    jobs = min(chain_jobs, settings.n_chains)
    if jobs > 1:
        one_thread_each()  # each chain's process a core
        with ProcessPoolExecutor(jobs) as executor:
            futures = [
                executor.submit(transdimensional.run_chain, space, run_settings, seed)
                for seed in seeds
            ]
            chains = [future.result() for future in futures]
    else:
        chains = [transdimensional.run_chain(space, run_settings, seed) for seed in seeds]
    best = min(chains, key=lambda chain: chain.best_misfit)
    layers = np.concatenate([chain.layers for chain in chains])
    noise = np.concatenate([chain.noise for chain in chains])
    lines = [
        f"Transdimensional, {settings.n_chains} chains of {settings.n_iterations:,} iterations "
        f"({settings.n_burnin_iterations:,} burn-in), each with {TEMPERATURES - 1} hotter "
        f"copies up to {HOTTEST:g}; Vs {free.vs_min:g}-{free.vs_max:g} m/s, interfaces "
        f"{free.depth_min:g}-{free.depth_max:g} m, at most {free.max_layers} layers."
    ]
    lines += [
        f"Chain {i + 1}: acceptance {chain.accepted:g} % "
        f"({', '.join(f'{move} {rate:g} %' for move, rate in chain.acceptance.items())}), "
        f"exchanges {chain.swaps:g} %."
        for i, chain in enumerate(chains)
    ]
    counts = np.bincount(layers, minlength=free.max_layers + 1)[1:]
    lines.append(
        "Layers kept: "
        + ", ".join(f"{k} in {100 * n / layers.size:.0f} %" for k, n in enumerate(counts, 1) if n)
        + f"; noise factor, median {float(np.median(noise)):.3g}."
    )
    return _Run(
        profiles=LayeredSamples(
            depths=np.concatenate([chain.depths for chain in chains]),
            vs=np.concatenate([chain.vs for chain in chains]),
            n_chains=settings.n_chains,
        ),
        samples={"layers": layers.astype(float), "noise": noise},
        rms=np.concatenate([chain.rms for chain in chains]),
        best=(best.best_depths, best.best_vs),
        acceptance=tuple(chain.accepted for chain in chains),
        # Each move's step as it ran after the burn-in (in the logarithm of the values moved).
        steps={
            move: round(float(np.median([chain.steps[move] for chain in chains])), 4)
            for move in chains[0].steps
        },
        log="\n".join(lines) + "\n",
        moves={
            move: round(float(np.median([chain.acceptance[move] for chain in chains])), 1)
            for move in chains[0].acceptance
        },
        exchanges=round(float(np.median([chain.swaps for chain in chains])), 1),
    )


def _result(
    settings: InversionParameters,
    run: _Run,
    picked: Sequence[DispersionCurve],
    vp_vs: float,
    dz: float,
    position: Coordinate,
) -> InversionResult:
    """The models the kept ones make, and each one's curves at the picked frequencies."""
    bottom = settings.bottom
    rows = int(np.ceil(bottom / dz))
    grid = (np.arange(rows) + 0.5) * dz
    rasters = run.profiles.at(grid)  # models x depths
    median_vs = np.median(rasters, axis=0)
    spread = np.std(rasters, axis=0)
    median_vp, median_rho = vp_rho_from_vs(median_vs, vp_vs)
    ensemble = VelocityModel(
        vs_s=tuple(median_vs.tolist()),
        vs_p=tuple(median_vp.tolist()),
        rhos=tuple(median_rho.tolist()),
        vs_s_std=tuple(spread.tolist()),
        thicknesses=tuple([dz] * rows),
        position=position,
    )
    coarse = np.log(rasters[:, ::_EVERY])
    nearest = int(np.argmin(np.mean((coarse - np.log(median_vs[::_EVERY])) ** 2, axis=1)))
    median = _layered(*run.profiles.model(nearest), spread, grid, bottom, vp_vs, position)
    best = _layered(*run.best, spread, grid, bottom, vp_vs, position)

    full = curves_of(picked, max_points=None)
    dpred: dict[int, np.ndarray] = {}
    predictions = [_predicted(full, run.profiles, i, vp_vs) for i in range(len(run.rms))]
    for c, curve in enumerate(full):
        rows_predicted = [
            prediction[c][curve.order]
            if prediction is not None
            else np.full(curve.order.size, np.nan)
            for prediction in predictions
        ]
        dpred[curve.mode] = np.array(rows_predicted)

    return InversionResult(
        best=best,
        smooth_best=best.smoothed(dz, bottom),
        median=median,
        smooth_median=median.smoothed(dz, bottom),
        ensemble=ensemble,
        n_layers=median.n_layers,
        samples=run.samples,
        misfits=run.rms,
        dpred=dpred,
        log=run.log,
        profiles=run.profiles,
        steps=run.steps,
        acceptance=run.acceptance,
        moves=run.moves,
        exchanges=run.exchanges,
        parameters=settings.model_dump(mode="json"),
    )


def _layered(
    depths: np.ndarray,
    vs: np.ndarray,
    spread: np.ndarray,
    grid: np.ndarray,
    bottom: float,
    vp_vs: float,
    position: Coordinate,
) -> VelocityModel:
    """A layered model, each layer's spread the ensemble's mean spread over its depths; the
    half-space drawn half way down to `bottom`, as the smoothing extends it."""
    tops = np.concatenate(([0.0], depths))
    thickness = np.diff(np.append(tops, bottom)).tolist()
    thickness[-1] = max((bottom - float(tops[-1])) / 2, float(grid[1] - grid[0]))
    below = np.append(tops[1:], math.inf)
    std = [
        float(np.mean(spread[(grid >= top) & (grid < base)]))
        if np.any((grid >= top) & (grid < base))
        else 0.0
        for top, base in zip(tops, below, strict=True)
    ]
    vp, rho = vp_rho_from_vs(np.asarray(vs, dtype=float), vp_vs)
    return VelocityModel(
        vs_s=tuple(float(v) for v in vs),
        vs_p=tuple(vp.tolist()),
        rhos=tuple(rho.tolist()),
        vs_s_std=tuple(std),
        thicknesses=tuple(thickness),
        position=position,
    )


def _predicted(
    curves: Sequence[Curve], profiles: LayeredSamples, index: int, vp_vs: float
) -> list[np.ndarray] | None:
    depths, vs = profiles.model(index)
    thickness = np.append(np.diff(np.concatenate(([0.0], depths))), 1_000.0)
    return phase_velocities(curves, thickness, vs, vp_vs)
