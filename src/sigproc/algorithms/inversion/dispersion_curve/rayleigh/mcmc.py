import contextlib
import io
from collections.abc import Callable
from typing import cast

import numpy as np
from bayesbay import BayesianInversion, ParameterSpaceState, State
from bayesbay.likelihood import LogLikelihood, Target
from bayesbay.parameterization import Parameterization, ParameterSpace
from bayesbay.prior import Prior, UniformPrior
from disba import DispersionError, PhaseDispersion

from sigproc.base.coordinate import Coordinate
from sigproc.base.dispersion_curve import DispersionCurves
from sigproc.base.inversion import InversionResult
from sigproc.base.velocity_model import VelocityModel

from .forward import fwd_function, vp_rho_from_vs


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


def inversion_mcmc(
    dispersion_curves: DispersionCurves,
    position: Coordinate,
    n_layers: int,
    thicknesses_min: tuple[float, ...],
    thicknesses_max: tuple[float, ...],
    thickness_perturbations: tuple[float, ...],
    Vs_mins: tuple[float, ...],
    Vs_maxs: tuple[float, ...],
    Vs_perturbations: tuple[float, ...],
    n_iterations: int,
    n_burnin: int,
    n_chains: int,
    Vp_Vs_ratio: float = 1.77,
    dz: float = 0.01,
) -> InversionResult:

    if len(dispersion_curves) == 0:
        raise ValueError("At least one dispersion curve must be provided for inversion.")

    if (
        len(thicknesses_min) != n_layers - 1
        or len(thicknesses_max) != n_layers - 1
        or len(thickness_perturbations) != n_layers - 1
    ):
        raise ValueError(
            "thicknesses_min, thicknesses_max, and thickness_perturbations must have length n_layers - 1, "
            f"got {len(thicknesses_min)}, {len(thicknesses_max)}, and {len(thickness_perturbations)}"
        )

    if len(Vs_mins) != n_layers or len(Vs_maxs) != n_layers or len(Vs_perturbations) != n_layers:
        raise ValueError(
            f"Vs_mins, Vs_maxs, and Vs_perturbations must have length n_layers, "
            f"got {len(Vs_mins)}, {len(Vs_maxs)}, and {len(Vs_perturbations)}"
        )

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
    fwd_functions = cast(
        "list[Callable[[dict[str, np.ndarray]], np.ndarray]]",
        [
            lambda state, mode=dispersion_curve.mode.number, fs=dispersion_curve.fs: fwd_function(
                state, n_layers, mode, fs, Vp_Vs_ratio
            )
            for dispersion_curve in ordered_curves
        ],
    )

    # Log-likelihood
    log_likelihood = LogLikelihood(
        targets=targets,
        fwd_functions=fwd_functions,  # pyright: ignore[reportArgumentType]
    )

    # Priors
    priors: list[Prior] = []
    for i in range(n_layers):
        priors.append(
            UniformPrior(
                name=f"vs{i + 1}",
                vmin=Vs_mins[i],  # pyright: ignore[reportArgumentType]
                vmax=Vs_maxs[i],  # pyright: ignore[reportArgumentType]
                perturb_std=Vs_perturbations[i],  # pyright: ignore[reportArgumentType]
            )
        )
    for i in range(n_layers - 1):
        priors.append(
            UniformPrior(
                name=f"thick{i + 1}",
                vmin=thicknesses_min[i],  # pyright: ignore[reportArgumentType]
                vmax=thicknesses_max[i],  # pyright: ignore[reportArgumentType]
                perturb_std=thickness_perturbations[i],  # pyright: ignore[reportArgumentType]
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
        n_chains=n_chains,
    )

    # Run inversion. Force chains to run sequentially within this process: positions
    # are already parallelized across worker processes by the caller, so letting
    # bayesbay also spawn one process per chain (its default) would oversubscribe
    # CPUs by n_workers x n_chains instead of just n_workers.
    inversion.run(
        n_iterations=n_iterations,
        burnin_iterations=n_burnin,
        save_every=150,
        verbose=False,
        parallel_config={"n_jobs": 1},
    )

    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        for chain in inversion.chains:
            chain.print_statistics()
    log = log_buffer.getvalue()

    results = cast(dict[str, np.ndarray], inversion.get_results(concatenate_chains=True))

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

    depth_max = float(np.nansum(thicknesses_max)) + 1
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
            vs_vals = []
            thick_vals = []
            for name, param in param_space.parameters.items():
                vmin, vmax = param.get_vmin_vmax(None)  # pyright: ignore[reportAttributeAccessIssue]
                if "vs" in name:
                    vs_vals.append(self._rng.uniform(vmin, vmax))
                elif "thick" in name:
                    thick_vals.append(self._rng.uniform(vmin, vmax))
            vs_arr = np.sort(vs_vals)
            thick_arr = np.sort(thick_vals)
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
