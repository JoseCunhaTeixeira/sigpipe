import contextlib
import io
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
from bayesbay import BayesianInversion, ParameterSpaceState, State
from bayesbay.likelihood import LogLikelihood, Target
from bayesbay.parameterization import Parameterization, ParameterSpace
from bayesbay.prior import Prior, UniformPrior
from disba import DispersionError, PhaseDispersion

from sigpipe.base.coordinate import Coordinate
from sigpipe.base.dispersion_curve import DispersionCurvesImage
from sigpipe.base.inversion import InversionResult
from sigpipe.base.velocity_model import VelocityModel

from .forward import fwd_function, vp_rho_from_vs
from .parameters import SAVE_EVERY, InversionParameters


def _ensemble_model(
    sampled_Vs: np.ndarray,
    sampled_thicknesses: np.ndarray,
    Vp_Vs_ratio: float,
    dz: float,
    depth_max: float,
    position: Coordinate,
) -> VelocityModel:
    """Rasterize every posterior sample onto a uniform depth grid, then take the per-depth median and std."""
    n_samples = sampled_Vs.shape[1]
    n_rows = int(np.ceil(depth_max / dz))

    rasterized_Vs = np.empty((n_samples, n_rows), dtype=np.float64)
    for s in range(n_samples):
        layer_rows = (sampled_thicknesses[:, s] / dz).astype(int)
        raster = np.repeat(sampled_Vs[:, s], layer_rows)
        if raster.shape[0] < n_rows:
            pad = np.full(n_rows - raster.shape[0], sampled_Vs[-1, s])
            raster = np.concatenate((raster, pad))
        rasterized_Vs[s] = raster[:n_rows]

    median_Vs = np.median(rasterized_Vs, axis=0)
    std_Vs = np.std(rasterized_Vs, axis=0)
    median_Vp, median_rho = vp_rho_from_vs(median_Vs, Vp_Vs_ratio)

    return VelocityModel(
        vs_s=tuple(median_Vs.tolist()),
        vs_p=tuple(median_Vp.tolist()),
        rhos=tuple(median_rho.tolist()),
        vs_s_std=tuple(std_Vs.tolist()),
        thicknesses=tuple([dz] * n_rows),
        position=position,
    )


def _make_fwd_function(
    mode: int, fs: np.ndarray, n_layers: int, Vp_Vs_ratio: float
) -> Callable[[dict[str, np.ndarray]], np.ndarray]:
    """A fully-typed def instead of a lambda -- lambda parameters can't carry
    annotations, and mode/fs need to be bound as real arguments (not closed
    over) since they vary per dispersion curve in the list comprehension
    that calls this."""

    def fwd(state: dict[str, np.ndarray]) -> np.ndarray:
        return fwd_function(state, n_layers, mode, fs, Vp_Vs_ratio)

    return fwd


def inversion_mcmc(
    dispersion_curves: DispersionCurvesImage,
    position: Coordinate,
    *,
    Vp_Vs_ratio: float = 1.77,
    dz: float = 0.01,
    **parameters: object,
) -> InversionResult:
    """The layered Vs models that fit `dispersion_curves`, sampled by Markov chains.

    `parameters` are InversionParameters' fields, which validate them: the number of layers,
    each layer's prior (its Vs and thickness bounds, and the sampler's steps) and the sampler's
    effort. Vp and density follow Vs by `Vp_Vs_ratio`; the smooth and ensemble models are
    sampled every `dz` metres.
    """
    settings = InversionParameters.model_validate(parameters)
    n_layers = settings.n_layers

    if len(dispersion_curves) == 0:
        raise ValueError("At least one dispersion curve must be provided for inversion.")

    modes: list[int] = sorted({dc.mode.number for dc in dispersion_curves})

    if len(modes) != len(dispersion_curves):
        raise ValueError("All dispersion curves must have a different mode.")

    ordered_curves = sorted(dispersion_curves, key=lambda dc: dc.mode.number)

    # Targets
    targets: list[Target] = []
    for dispersion_curve in ordered_curves:
        assert dispersion_curve.vs_err is not None, "vs_err must be provided for MCMC inversion"
        covariance_mat_inv = np.diag(1 / dispersion_curve.vs_err**2)
        target = Target(
            name=f"rayleigh_M{dispersion_curve.mode.number}",
            dobs=dispersion_curve.vs,
            covariance_mat_inv=covariance_mat_inv,
        )
        targets.append(target)

    # Forward functions
    fwd_functions = [
        _make_fwd_function(dispersion_curve.mode.number, dispersion_curve.fs, n_layers, Vp_Vs_ratio)
        for dispersion_curve in ordered_curves
    ]

    # Log-likelihood
    log_likelihood = LogLikelihood(
        targets=targets,
        fwd_functions=fwd_functions,  # pyright: ignore[reportArgumentType]
    )

    # Priors
    priors: list[Prior] = []
    for i, vs_layer in enumerate(settings.vs_layers):
        priors.append(
            UniformPrior(
                name=f"vs{i + 1}",
                vmin=vs_layer.vs_min,  # pyright: ignore[reportArgumentType]
                vmax=vs_layer.vs_max,  # pyright: ignore[reportArgumentType]
                perturb_std=vs_layer.vs_perturb_std,  # pyright: ignore[reportArgumentType]
            )
        )
    for i, thickness_layer in enumerate(settings.thickness_layers):
        priors.append(
            UniformPrior(
                name=f"thick{i + 1}",
                vmin=thickness_layer.thickness_min,  # pyright: ignore[reportArgumentType]
                vmax=thickness_layer.thickness_max,  # pyright: ignore[reportArgumentType]
                perturb_std=thickness_layer.thickness_perturb_std,  # pyright: ignore[reportArgumentType]
            )
        )

    # Parameter space
    param_space: ParameterSpace = ParameterSpace(
        name="space",
        n_dimensions=1,
        parameters=priors,  # pyright: ignore[reportArgumentType]
    )

    # Parameterization
    parameterization = CustomParametrization(
        param_space,
        modes,
        [dispersion_curve.fs for dispersion_curve in ordered_curves],
        Vp_Vs_ratio,
    )

    # Inversion
    inversion: BayesianInversion = BayesianInversion(
        log_likelihood=log_likelihood,
        parameterization=parameterization,
        n_chains=settings.n_chains,
    )

    # Run inversion. Force chains to run sequentially within this process: positions
    # are already parallelized across worker processes by the caller, so letting
    # bayesbay also spawn one process per chain (its default) would oversubscribe
    # CPUs by n_workers x n_chains instead of just n_workers.
    inversion.run(
        n_iterations=settings.n_iterations,
        burnin_iterations=settings.n_burnin_iterations,
        save_every=SAVE_EVERY,
        verbose=False,
        parallel_config={"n_jobs": 1},
    )

    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        for chain in inversion.chains:
            chain.print_statistics()
    log = log_buffer.getvalue()

    per_chain = cast(dict[str, list[list[Any]]], inversion.get_results(concatenate_chains=False))
    results, left_out = _saved_with_predictions(
        per_chain, [f"rayleigh_M{mode}.dpred" for mode in modes]
    )
    if left_out:
        log += (
            f"{left_out} saved models left out: saved before their chain computed a likelihood "
            "(a chain still on its starting model), they carry no predicted curve.\n"
        )

    # Extract sampled models
    sampled_Vs = np.array(
        [np.asarray(results[f"space.vs{i + 1}"]).reshape(-1) for i in range(n_layers)]
    )
    n_samples = sampled_Vs.shape[1]
    sampled_thicknesses = np.array(
        [np.asarray(results[f"space.thick{i + 1}"]).reshape(-1) for i in range(n_layers - 1)]
        + [np.full(n_samples, 1000.0)]
    )

    # Misfits, summed across all dispersion curves
    misfits = np.zeros(n_samples)
    n_points = 0
    dpred: dict[int, np.ndarray] = {}
    for dispersion_curve in ordered_curves:
        d_pred = np.asarray(results[f"rayleigh_M{dispersion_curve.mode.number}.dpred"])
        dpred[dispersion_curve.mode.number] = d_pred
        misfits += np.sum((dispersion_curve.vs - d_pred) ** 2, axis=1)
        n_points += len(dispersion_curve.vs)
    misfits = np.sqrt(misfits / n_points)

    depth_max = sum(layer.thickness_max for layer in settings.thickness_layers) + 1.0
    Vs_stds = np.std(sampled_Vs, axis=1)

    # Best layered model (lowest-misfit sample)
    best_idx = np.argmin(misfits)
    best_Vs = sampled_Vs[:, best_idx]
    best_thicknesses = sampled_thicknesses[:, best_idx].copy()
    best_Vp, best_rho = vp_rho_from_vs(best_Vs, Vp_Vs_ratio)
    best_thicknesses[-1] = (depth_max - np.sum(best_thicknesses[:-1])) / 2

    best = VelocityModel(
        vs_s=tuple(best_Vs.tolist()),
        vs_p=tuple(best_Vp.tolist()),
        rhos=tuple(best_rho.tolist()),
        vs_s_std=tuple(Vs_stds.tolist()),
        thicknesses=tuple(best_thicknesses.tolist()),
        position=position,
    )

    # Median layered model (per-layer median across all samples)
    median_Vs = np.median(sampled_Vs, axis=1)
    median_thicknesses = np.median(sampled_thicknesses, axis=1)
    median_Vp, median_rho = vp_rho_from_vs(median_Vs, Vp_Vs_ratio)
    median_thicknesses[-1] = (depth_max - np.sum(median_thicknesses[:-1])) / 2

    median = VelocityModel(
        vs_s=tuple(median_Vs.tolist()),
        vs_p=tuple(median_Vp.tolist()),
        rhos=tuple(median_rho.tolist()),
        vs_s_std=tuple(Vs_stds.tolist()),
        thicknesses=tuple(median_thicknesses.tolist()),
        position=position,
    )

    # Smoothed (cubic-interpolated, continuous-with-depth) best/median profiles
    smooth_best = best.smoothed(dz, depth_max)
    smooth_median = median.smoothed(dz, depth_max)

    # Ensemble model (per-depth median/std after rasterizing every sample)
    ensemble = _ensemble_model(
        sampled_Vs, sampled_thicknesses, Vp_Vs_ratio, dz, depth_max, position
    )

    samples = {f"vs{i + 1}": sampled_Vs[i] for i in range(n_layers)}
    samples.update({f"thick{i + 1}": sampled_thicknesses[i] for i in range(n_layers - 1)})

    return InversionResult(
        best=best,
        smooth_best=smooth_best,
        median=median,
        smooth_median=smooth_median,
        ensemble=ensemble,
        n_layers=n_layers,
        samples=samples,
        misfits=misfits,
        dpred=dpred,
        log=log,
    )


def _saved_with_predictions(
    per_chain: dict[str, list[list[Any]]], dpred_keys: Sequence[str]
) -> tuple[dict[str, list[Any]], int]:
    """The chains' saved states that carry a predicted curve for every target, concatenated
    over the chains, and how many were left out.

    bayesbay saves a state's predicted curve only once its likelihood has been computed: a
    chain whose proposals all fall outside the prior never computes one, and saves its starting
    model with no predicted curve. Those saves come first in their chain (once computed, every
    later state carries it), so each chain keeps its last saves, as many as it has predicted
    curves. A chain that never moved keeps none; no chain keeping any is an error.
    """
    n_chains = len(per_chain["space.vs1"])
    kept: dict[str, list[Any]] = {key: [] for key in per_chain}
    left_out = 0
    for chain in range(n_chains):
        n_saved = len(per_chain["space.vs1"][chain])
        n_predicted = min(
            len(per_chain[key][chain]) if key in per_chain else 0 for key in dpred_keys
        )
        left_out += n_saved - n_predicted
        if n_predicted == 0:
            continue
        for key, chains in per_chain.items():
            kept[key].extend(chains[chain][-n_predicted:])
    if not kept["space.vs1"]:
        raise ValueError(
            "No chain moved from its starting model: every proposal fell outside the prior, so "
            "no model has a predicted curve. Check that each layer's bounds hold the curve."
        )
    return kept, left_out


class CustomParametrization(Parameterization):  # type: ignore[misc]
    def __init__(
        self,
        param_space: ParameterSpace,
        modes: list[int],
        fs_per_mode: list[np.ndarray],
        Vp_Vs_ratio: float = 1.77,
    ) -> None:
        super().__init__(param_space)
        self.modes = modes
        self.fs_per_mode = fs_per_mode
        self.Vp_Vs_ratio = Vp_Vs_ratio
        self._rng = np.random.default_rng()

    def initialize(self) -> State:
        param_values = {}
        for ps_name, ps in self.parameter_spaces.items():
            param_values[ps_name] = self.initialize_param_space(ps)
        return State(param_values)

    def initialize_param_space(self, param_space: ParameterSpace) -> ParameterSpaceState:
        while True:
            vs_vals: list[float] = []
            vs_bounds: list[tuple[float, float]] = []
            thick_vals: list[float] = []
            thick_bounds: list[tuple[float, float]] = []
            for name, param in param_space.parameters.items():
                vmin, vmax = param.get_vmin_vmax(None)  # pyright: ignore[reportAttributeAccessIssue]
                if "vs" in name:
                    vs_vals.append(self._rng.uniform(vmin, vmax))
                    vs_bounds.append((vmin, vmax))
                elif "thick" in name:
                    thick_vals.append(self._rng.uniform(vmin, vmax))
                    thick_bounds.append((vmin, vmax))
            # Sorted, the start is normally dispersive; clipped, each value stays inside its own
            # layer's prior. Sorting alone could move a value outside it when the layers' bounds
            # differ, and a chain starting outside its prior may never move (every proposal
            # rejected) nor compute a predicted curve.
            vs_arr = _within(np.sort(vs_vals), vs_bounds)
            thick_arr = _within(np.sort(thick_vals), thick_bounds)
            vp_vals, rho_vals = vp_rho_from_vs(vs_arr, self.Vp_Vs_ratio)
            velocity_model = np.column_stack(
                (np.append(thick_arr, 1000), vp_vals, vs_arr, rho_vals)
            )
            velocity_model /= 1000  # m to km and kg/m^3 to g/cm^3
            try:
                for mode, fs in zip(self.modes, self.fs_per_mode, strict=False):
                    pd = PhaseDispersion(*velocity_model.T)
                    periods = 1 / fs[::-1]
                    d_pred = pd(periods, mode=mode, wave="rayleigh").velocity
                    if (
                        d_pred.shape[0] != periods.shape[0]
                    ):  # Test if the dispersion curve is too short - It is often the case for low velocities (i.e. high periods) on superior modes
                        raise DispersionError(
                            f"Dispersion curve length for mode {mode} is not the same as the observed one"
                        )
                break
            except DispersionError:
                continue
        vals = np.concatenate((vs_arr, thick_arr))
        param_values = {}
        for i, name in enumerate(param_space.parameters.keys()):
            param_values[name] = np.array([vals[i]])
        return ParameterSpaceState(1, param_values)


def _within(values: np.ndarray, bounds: Sequence[tuple[float, float]]) -> np.ndarray:
    """`values`, each clipped into its own (min, max) bounds."""
    lows = np.array([low for low, _ in bounds], dtype=float)
    highs = np.array([high for _, high in bounds], dtype=float)
    return np.clip(values, lows, highs)
