"""One window's inversion, on a window folder of a run: sigpipe's MCMC on the window's picked
curves, and PAC's files beside them."""

import logging
from collections.abc import Collection
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from disba import DispersionError
from pydantic import BaseModel, ConfigDict

from sigpipe.algorithms.inversion.rayleigh.seismic.forward import (
    fwd_seismic_all_modes,
    fwd_seismic_phase,
)
from sigpipe.algorithms.inversion.rayleigh.seismic.parameters import (
    InversionParameters,
)
from sigpipe.algorithms.picking.dispersion.curve import min_resolvable_wavelength
from sigpipe.base.dispersion_curve import DispersionCurves, Mode
from sigpipe.base.inversion import InversionResult, LayeredSamples
from sigpipe.base.pipeline import Pipeline
from sigpipe.dataio.dispersion.loading import load_dispersion_curves
from sigpipe.dataio.dispersion.plotting import plot_dispersion_image
from sigpipe.dataio.dispersion.saving import save_dispersion_curves
from sigpipe.dataio.inversion.forward import MODEL_NAMES, forward_model_all
from sigpipe.dataio.inversion.plotting import plot_density_curves, plot_posterior_marginals
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.runs import load_image
from sigpipe.transformers import Invert, Plot, Save

logger = logging.getLogger(__name__)

# The inversion's fixed values.
DZ = 0.01  # m
VP_VS_RATIO = 1.77
SAMPLES_FILE = "SeismicInversion_Samples_0000.npz"  # PACo's own, next to PAC's files
PARAMETERS_FILE = "SeismicInversion_Parameters_0000.json"
# The samples file's arrays that are no named value: the layers, and what describes the file.
_DEPTHS = "profile_depths"
_VS = "profile_vs"
_NOT_NAMED = ("n_chains", "misfits", _DEPTHS, _VS)
M0 = Mode("M", 0)  # the fundamental mode, as the pickers label it


class WindowParameters(BaseModel):
    """What a window's chains ran with: the layering and its priors (the free layering's bounds
    as found from the curves), the chains' effort, how often each chain accepted a move, and
    each sampled parameter's typical move (the fixed layering)."""

    model_config = ConfigDict(frozen=True)

    parameters: InversionParameters  # as run
    # The trial runs (step factor, acceptance %) of the runs saved before 2026-09-27.
    tuning: tuple[tuple[float, float], ...] = ()
    acceptance: tuple[float, ...]  # each chain's over the run (%), the burn-in included
    steps: dict[str, float] = {}  # each sampled parameter's typical move (vs1, ..., thick1, ...)


def build_inversion_pipeline(
    parameters: InversionParameters, output_folder: Path, chain_jobs: int = 1
) -> Pipeline:
    """The inversion pipeline: sigpipe's MCMC, its chains in `chain_jobs` processes, then the
    models saved in `output_folder`."""
    mcmc = Invert(
        method="mcmc", Vp_Vs_ratio=VP_VS_RATIO, dz=DZ, chain_jobs=chain_jobs, **dict(parameters)
    )
    return mcmc >> Save(folder_path=output_folder, file_name="SeismicInversion_Model")


def invert_window(
    folder: Path,
    parameters: InversionParameters,
    modes: Collection[Mode] = (M0,),
    chain_jobs: int = 1,
    output_folder: Path | None = None,
) -> InversionResult:
    """Invert the curves of `modes` saved in window folder `folder` (M0 by default), the chains
    in `chain_jobs` processes, and write PAC's files next to them, or in `output_folder` (a
    staging folder: see sigpipe.masw.runs.stopping)."""
    out = output_folder or folder
    image = load_image(folder)
    curves = _curves(folder, modes)
    result: InversionResult = build_inversion_pipeline(parameters, out, chain_jobs).run(
        data=[curves], show_log=False
    )[0]

    (out / "SeismicInversion_Log_0000.log").write_text(result.log)
    save_parameters(parameters, result, out / PARAMETERS_FILE)
    save_samples(result, out / SAMPLES_FILE)

    # The median model's M0 at the picked frequencies, and every mode it supports across the
    # image, drawn over the image.
    median = result.median
    try:
        modeled_curves = DispersionCurves(
            dispersion_curves=tuple(
                fwd_seismic_phase(
                    thickness_per_layer=list(median.thicknesses),
                    Vs_per_layer=list(median.vs_s),
                    mode=curve.mode.number,
                    fs=curve.fs,
                    Vp_Vs_ratio=VP_VS_RATIO,
                )
                for curve in curves
            )
        )
    except DispersionError:
        # A layer over a slower half-space has no normal mode faster than the half-space: the
        # median of the samples can lack one where every sample had it. The figure goes without.
        logger.warning("No mode of the median model at the picked frequencies in %s", folder)
        modeled_curves = None
    full_modeled_curves = fwd_seismic_all_modes(
        thickness_per_layer=list(median.thicknesses),
        Vs_per_layer=list(median.vs_s),
        fs=image.fs,
        Vp_Vs_ratio=VP_VS_RATIO,
    )
    figure = plot_dispersion_image(
        image,
        picked_curves=curves,
        modeled_curves=modeled_curves,
        full_modeled_curves=full_modeled_curves,
        lbmin=min_resolvable_wavelength(image.acquisition),
        normalize=True,
        show_errorbars=True,
    )
    Plot.savefig(path=out / "SeismicInversion_DispersionImage_0000.png", figure=figure)
    plt.close(figure)

    forward_modeled = forward_model_all(result, curves, VP_VS_RATIO)
    for model_name in MODEL_NAMES:
        modeled = forward_modeled[model_name]
        if modeled is None:
            logger.warning(
                "Could not forward-model '%s' in %s; skipping its curves", model_name, folder
            )
            continue
        save_dispersion_curves(
            modeled, path=out / f"SeismicInversion_DispersionCurves_0000_{model_name}.csv"
        )

    figure = plot_density_curves(result, curves, VP_VS_RATIO)
    Plot.savefig(path=out / "SeismicInversion_DensityCurves_0000.png", figure=figure)
    plt.close(figure)

    try:
        figure = plot_posterior_marginals(marginals(result))
        Plot.savefig(path=out / "SeismicInversion_Marginals_0000.png", figure=figure)
        plt.close(figure)
    except Exception:  # a figure must not lose the inversion
        logger.exception("Could not plot the posterior marginals in %s", folder)

    return result


def marginals(result: InversionResult) -> dict[str, np.ndarray]:
    """What the marginals figure shows, one value per kept model: each sampled value of the
    fixed layering; the number of layers and the Vs at three depths, from the shallowest
    interface allowed to the deepest, when the data chose the layers; the noise factor. A value
    the same in every model has no density: left out."""
    shown: dict[str, np.ndarray] = {}
    for name, values in result.samples.items():
        if name.startswith("vs") and name[2:].isdigit():
            shown[f"Vs{name[2:]} [m/s]"] = values
        elif name.startswith("thick") and name[5:].isdigit():
            shown[f"H{name[5:]} [m]"] = values
    if "layers" in result.samples:
        shown["Layers"] = result.samples["layers"]
        free = result.parameters.get("free") or {}
        top, bottom = free.get("depth_min"), free.get("depth_max")
        if result.profiles is not None and top and bottom:
            depths = np.round(np.geomspace(float(top), float(bottom), 3), 1)
            at = result.profiles.at(depths)
            shown |= {f"Vs at {depth:g} m [m/s]": at[:, i] for i, depth in enumerate(depths)}
    if "noise" in result.samples:
        shown["Noise factor"] = result.samples["noise"]
    return {name: values for name, values in shown.items() if np.ptp(values) > 0}


def save_parameters(parameters: InversionParameters, result: InversionResult, path: Path) -> None:
    """The parameters `result`'s chains ran with (the free layering's bounds found from the
    curves), each chain's acceptance and the sampled parameters' typical moves."""
    ran = InversionParameters.model_validate(result.parameters) if result.parameters else parameters
    window = WindowParameters(
        parameters=ran,
        tuning=result.tuning,
        acceptance=result.acceptance,
        steps={name: _significant(step) for name, step in result.steps.items()},
    )
    path.write_text(window.model_dump_json(indent=2))


def load_parameters(path: Path) -> WindowParameters:
    """The parameters `save_parameters` wrote."""
    return WindowParameters.model_validate_json(path.read_text())


def save_samples(result: InversionResult, path: Path) -> None:
    """The kept models, chain after chain: their named values, their layers (interfaces' depths
    and Vs, NaN beyond a model's layers) and each one's misfit."""
    arrays: dict[str, Any] = {name: np.asarray(values) for name, values in result.samples.items()}
    if result.profiles is not None:
        arrays[_DEPTHS] = result.profiles.depths
        arrays[_VS] = result.profiles.vs
        n_chains = result.profiles.n_chains
    else:
        n_chains = len(result.acceptance) or 1
    np.savez_compressed(
        path, n_chains=np.array(n_chains), misfits=np.asarray(result.misfits), **arrays
    )


def load_samples(path: Path) -> tuple[dict[str, np.ndarray], int]:
    """The named values `save_samples` wrote (vs1, ..., thick1, ..., layers, noise), and the
    number of chains they come from."""
    with np.load(path) as saved:
        n_chains = int(saved["n_chains"])
        samples = {name: saved[name] for name in saved.files if name not in _NOT_NAMED}
    return samples, n_chains


def load_profiles(path: Path) -> LayeredSamples:
    """The kept models as layers; for a run saved before they were, rebuilt from its layers'
    named values (vs1, ..., thick1, ...)."""
    with np.load(path) as saved:
        n_chains = int(saved["n_chains"])
        if _VS in saved.files:
            return LayeredSamples(depths=saved[_DEPTHS], vs=saved[_VS], n_chains=n_chains)
        n_layers = sum(1 for name in saved.files if name.startswith("vs") and name[2:].isdigit())
        vs = np.column_stack([saved[f"vs{i + 1}"] for i in range(n_layers)])
        thickness = (
            np.column_stack([saved[f"thick{i + 1}"] for i in range(n_layers - 1)])
            if n_layers > 1
            else np.empty((vs.shape[0], 0))
        )
    return LayeredSamples(depths=np.cumsum(thickness, axis=1), vs=vs, n_chains=n_chains)


def _significant(step: float) -> float:
    """A step to 3 significant digits: never rounded to 0."""
    return float(f"{step:.3g}")


def _curves(folder: Path, modes: Collection[Mode]) -> DispersionCurves:
    path = folder / CURVES_FILE
    saved = load_dispersion_curves([path])[0] if path.exists() else ()
    chosen = tuple(curve for curve in saved if curve.mode in modes)
    if not chosen:
        labels = " or ".join(sorted(mode.label for mode in modes))
        raise ValueError(f"No {labels} curve in {folder.name}: pick it, or pick it again by hand.")
    return DispersionCurves(dispersion_curves=chosen)
