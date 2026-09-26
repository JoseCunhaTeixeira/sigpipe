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
    ThicknessLayer,
    VsLayer,
)
from sigpipe.algorithms.picking.dispersion.curve import min_resolvable_wavelength
from sigpipe.base.dispersion_curve import DispersionCurves, Mode
from sigpipe.base.inversion import InversionResult
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
M0 = Mode("M", 0)  # the fundamental mode, as the pickers label it


class WindowParameters(BaseModel):
    """What a window's sampler ran with: each layer's prior and the sampler's effort, the steps
    as the trial runs tuned them, and how often each chain accepted a proposal."""

    model_config = ConfigDict(frozen=True)

    parameters: InversionParameters  # as run: each step as the trial runs tuned it
    tuning: tuple[tuple[float, float], ...]  # each trial run's step factor and acceptance (%)
    acceptance: tuple[float, ...]  # each chain's over the run (%), the burn-in included


def build_inversion_pipeline(parameters: InversionParameters, output_folder: Path) -> Pipeline:
    """The inversion pipeline: sigpipe's MCMC, then the models saved in `output_folder`."""
    return Invert(method="mcmc", Vp_Vs_ratio=VP_VS_RATIO, dz=DZ, **dict(parameters)) >> Save(
        folder_path=output_folder, file_name="SeismicInversion_Model"
    )


def invert_window(
    folder: Path, parameters: InversionParameters, modes: Collection[Mode] = (M0,)
) -> InversionResult:
    """Invert the curves of `modes` saved in window folder `folder` (M0 by default), and write
    PAC's files next to them."""
    image = load_image(folder)
    curves = _curves(folder, modes)
    result: InversionResult = build_inversion_pipeline(parameters, folder).run(
        data=[curves], show_log=False
    )[0]

    (folder / "SeismicInversion_Log_0000.log").write_text(result.log)
    save_parameters(parameters, result, folder / PARAMETERS_FILE)
    save_samples(result, parameters.n_chains, folder / SAMPLES_FILE)

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
    Plot.savefig(path=folder / "SeismicInversion_DispersionImage_0000.png", figure=figure)
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
            modeled, path=folder / f"SeismicInversion_DispersionCurves_0000_{model_name}.csv"
        )

    figure = plot_density_curves(result, curves, VP_VS_RATIO)
    Plot.savefig(path=folder / "SeismicInversion_DensityCurves_0000.png", figure=figure)
    plt.close(figure)

    samples = {f"Vs{i + 1} [m/s]": result.samples[f"vs{i + 1}"] for i in range(result.n_layers)}
    samples |= {
        f"H{i + 1} [m]": result.samples[f"thick{i + 1}"] for i in range(result.n_layers - 1)
    }
    try:
        figure = plot_posterior_marginals(samples)
        Plot.savefig(path=folder / "SeismicInversion_Marginals_0000.png", figure=figure)
        plt.close(figure)
    except Exception:  # a figure must not lose the inversion
        logger.exception("Could not plot the posterior marginals in %s", folder)

    return result


def save_parameters(parameters: InversionParameters, result: InversionResult, path: Path) -> None:
    """`parameters` as `result`'s sampler ran them, with its tuning and acceptance."""

    def step(name: str, given: float) -> float:
        return _significant(result.steps.get(name, given))

    ran = parameters.model_copy(
        update={
            "vs_layers": tuple(
                VsLayer(
                    vs_min=layer.vs_min,
                    vs_max=layer.vs_max,
                    vs_perturb_std=step(f"vs{i + 1}", layer.vs_perturb_std),
                )
                for i, layer in enumerate(parameters.vs_layers)
            ),
            "thickness_layers": tuple(
                ThicknessLayer(
                    thickness_min=layer.thickness_min,
                    thickness_max=layer.thickness_max,
                    thickness_perturb_std=step(f"thick{i + 1}", layer.thickness_perturb_std),
                )
                for i, layer in enumerate(parameters.thickness_layers)
            ),
        }
    )
    window = WindowParameters(parameters=ran, tuning=result.tuning, acceptance=result.acceptance)
    path.write_text(window.model_dump_json(indent=2))


def load_parameters(path: Path) -> WindowParameters:
    """The parameters `save_parameters` wrote."""
    return WindowParameters.model_validate_json(path.read_text())


def save_samples(result: InversionResult, n_chains: int, path: Path) -> None:
    """The posterior samples, chain after chain as sigpipe concatenates them, with each
    sample's misfit: what G5 judges convergence and the prior's bounds on."""
    arrays: dict[str, Any] = {name: np.asarray(values) for name, values in result.samples.items()}
    np.savez_compressed(
        path, n_chains=np.array(n_chains), misfits=np.asarray(result.misfits), **arrays
    )


def load_samples(path: Path) -> tuple[dict[str, np.ndarray], int]:
    """The samples `save_samples` wrote, by parameter (vs1, ..., thick1, ...), and the number
    of chains they come from."""
    with np.load(path) as saved:
        n_chains = int(saved["n_chains"])
        samples = {name: saved[name] for name in saved.files if name not in ("n_chains", "misfits")}
    return samples, n_chains


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
