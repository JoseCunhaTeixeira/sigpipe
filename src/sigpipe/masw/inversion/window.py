"""One window's inversion, on a window folder of a run: sigpipe's MCMC on the window's picked
curves, and PAC's files beside them."""

import logging
import warnings
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, ConfigDict

from sigpipe.algorithms.inversion.rayleigh.seismic.parameters import (
    InversionParameters,
    SavedInversionParameters,
)
from sigpipe.algorithms.picking.dispersion.curve import (
    longest_reached_wavelength,
    min_resolvable_wavelength,
)
from sigpipe.base.dispersion_curve import DispersionCurves, Mode
from sigpipe.base.inversion import InversionResult, LayeredSamples
from sigpipe.base.pipeline import Pipeline
from sigpipe.dataio.dispersion.loading import load_dispersion_curves
from sigpipe.dataio.dispersion.plotting import plot_dispersion_image
from sigpipe.dataio.dispersion.saving import save_dispersion_curves
from sigpipe.dataio.inversion.forward import MODEL_NAMES, forward_model_all
from sigpipe.dataio.inversion.plotting import plot_inversion_window, plot_posterior_marginals
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.runs import load_image
from sigpipe.transformers import Invert, Plot, Save

logger = logging.getLogger(__name__)

# The inversion's fixed values.
DZ = 0.01  # m
VP_VS_RATIO = 1.77
SAMPLES_FILE = "SeismicInversion_Samples_0000.npz"  # PACo's own, next to PAC's files
PARAMETERS_FILE = "SeismicInversion_Parameters_0000.json"
# The kept models' curves at the picked frequencies, as the density figure's band: their 10th,
# 50th and 90th percentiles.
SPREAD_FILE = "SeismicInversion_DispersionSpread_0000.csv"
SPREAD_PERCENTILES = (10.0, 50.0, 90.0)
# The kept models' Vs at each depth, as the same percentiles: the profile's band, the section's
# uncertainty and the depth informed read from it (measuring.useful_depth).
VS_SPREAD_FILE = "SeismicInversion_VsSpread_0000.csv"
VS_SPREAD_DZ = 0.05  # m between its depths
_SPREAD_HEADER = "depth_m,p10_m/s,p50_m/s,p90_m/s"
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

    parameters: SavedInversionParameters  # as run
    # The trial runs (step factor, acceptance %) of the runs saved before 2026-09-27.
    tuning: tuple[tuple[float, float], ...] = ()
    acceptance: tuple[float, ...]  # each chain's over the run (%), the burn-in included
    # Each sampled parameter's typical move (vs1, ..., thick1, ...); when the data chose the
    # layers, each move's step (vs, interface, noise, shift, stretch: in the logarithm of the
    # values moved), with each move's acceptance and the exchanges between tempered copies (%,
    # the chains' medians; saved since 2026-09-29).
    steps: dict[str, float] = {}
    moves: dict[str, float] = {}
    exchanges: float | None = None


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
    save_spread(result, curves, out / SPREAD_FILE)
    ran = load_parameters(out / PARAMETERS_FILE).parameters
    spread = vs_spread(result.profiles, ran.bottom)
    save_vs_spread(spread, out / VS_SPREAD_FILE)

    # The median of the ensemble's curves at the picked frequencies (as each model's, saved
    # beside it), drawn over the image, and with the models' 10-90 % and the profile's own.
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
    ensemble_curves = forward_modeled["ensemble"]

    figure = plot_dispersion_image(
        image,
        picked_curves=curves,
        modeled_curves=ensemble_curves,
        # Where the checks' flags start: under lbmin the aliasing zone, over lbmax beyond the
        # window's reach.
        lbmin=min_resolvable_wavelength(image.acquisition),
        lbmax=longest_reached_wavelength(image.acquisition),
        normalize=True,
        show_errorbars=True,
    )
    Plot.savefig(path=out / "SeismicInversion_DispersionImage_0000.png", figure=figure)
    plt.close(figure)

    figure = plot_inversion_window(
        curves,
        ensemble_curves,
        _curve_spreads(result),
        result.ensemble,
        (spread.depths, spread.low, spread.high, spread.uncertainty()),
        _useful_depth(spread, result.profiles),
        _interface_shares(result, ran.bottom),
    )
    Plot.savefig(path=out / "SeismicInversion_DensityCurves_0000.png", figure=figure)
    plt.close(figure)

    try:
        figure = plot_posterior_marginals(marginals(result))
        Plot.savefig(path=out / "SeismicInversion_Marginals_0000.png", figure=figure)
        plt.close(figure)
    except Exception:  # a figure must not lose the inversion
        logger.exception("Could not plot the posterior marginals in %s", folder)

    return result


def _curve_spreads(result: InversionResult) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """The kept models' curves at each mode's picked frequencies, as their 10th and 90th
    percentiles, by mode number (a frequency no model reaches: NaN)."""
    spreads: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for mode, predicted in result.dpred.items():
        if not np.isfinite(predicted).any():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            low, high = np.nanpercentile(
                predicted, (SPREAD_PERCENTILES[0], SPREAD_PERCENTILES[2]), axis=0
            )
        spreads[mode] = (low, high)
    return spreads


def _interface_shares(result: InversionResult, bottom: float) -> tuple[float, ...]:
    """Per 0.5 m from the surface down, the share of the kept models with an interface there,
    as the checks measure it."""
    from sigpipe.masw.inversion.measuring import interface_shares  # it imports this module

    return interface_shares(result.profiles, bottom) if result.profiles is not None else ()


def _useful_depth(spread: VsSpread, profiles: LayeredSamples) -> float | None:
    """The depth informed as the checks read it, at their default limit (measuring's)."""
    from sigpipe.masw.inversion.measuring import useful_depth  # it imports this module

    return useful_depth(spread, profiles, 0.25)


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
    curves), each chain's acceptance and the sampled parameters' typical moves; when the data
    chose the layers, each move's step and acceptance, and the exchanges."""
    ran = InversionParameters.model_validate(result.parameters) if result.parameters else parameters
    window = WindowParameters(
        parameters=SavedInversionParameters.model_validate(ran.model_dump()),
        tuning=result.tuning,
        acceptance=result.acceptance,
        steps={name: _significant(step) for name, step in result.steps.items()},
        moves=result.moves,
        exchanges=result.exchanges,
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


@dataclass(frozen=True, slots=True)
class Spread:
    """The kept models' curves at one mode's picked frequencies: their SPREAD_PERCENTILES."""

    fs: np.ndarray  # Hz, in the picked order
    low: np.ndarray  # m/s, the 10th percentile
    middle: np.ndarray  # the 50th
    high: np.ndarray  # the 90th


def save_spread(result: InversionResult, curves: DispersionCurves, path: Path) -> None:
    """The kept models' curves at each mode's picked frequencies, as their SPREAD_PERCENTILES (the
    band the density figure draws): a row per mode and frequency. A model with no such mode at a
    frequency counts for none there."""
    lines = ["mode,frequency_Hz,p10_m/s,p50_m/s,p90_m/s"]
    for curve in curves:
        predicted = result.dpred.get(curve.mode.number)
        if predicted is None or not np.isfinite(predicted).any():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a frequency no model reaches
            low, middle, high = np.nanpercentile(predicted, SPREAD_PERCENTILES, axis=0)
        for row in zip(np.asarray(curve.fs, dtype=float), low, middle, high, strict=True):
            lines.append(",".join([curve.mode.label, *(f"{value:.6g}" for value in row)]))
    path.write_text("\n".join(lines) + "\n")


def load_spread(folder: Path) -> dict[str, Spread]:
    """Per mode label, the spread `save_spread` wrote in window folder `folder`; empty without
    its file (the inversions saved before 2026-09-29)."""
    path = folder / SPREAD_FILE
    if not path.exists():
        return {}
    rows: dict[str, list[tuple[float, float, float, float]]] = {}
    for line in path.read_text().splitlines()[1:]:
        label, *values = line.split(",")
        f, low, middle, high = (float(value) for value in values)
        rows.setdefault(label, []).append((f, low, middle, high))
    return {
        label: Spread(*(np.array(column) for column in zip(*values, strict=True)))
        for label, values in rows.items()
    }


@dataclass(frozen=True, slots=True)
class VsSpread:
    """The kept models' Vs at depths from the surface down: their SPREAD_PERCENTILES."""

    depths: np.ndarray  # m, each cell's middle, VS_SPREAD_DZ apart
    low: np.ndarray  # m/s, the 10th percentile
    middle: np.ndarray  # the 50th
    high: np.ndarray  # the 90th

    def uncertainty(self) -> np.ndarray:
        """U(z), the relative uncertainty of Vs at each depth: (P90 - P10) / (2 P50)."""
        return (self.high - self.low) / (2 * self.middle)


def vs_spread(profiles: LayeredSamples, bottom: float, dz: float = VS_SPREAD_DZ) -> VsSpread:
    """The kept models' (`profiles`) Vs in cells `dz` thick down to `bottom` (m), at each cell's
    middle, as their SPREAD_PERCENTILES."""
    depths = (np.arange(max(int(np.ceil(bottom / dz)), 1)) + 0.5) * dz
    low, middle, high = np.percentile(profiles.at(depths), SPREAD_PERCENTILES, axis=0)
    return VsSpread(depths, low, middle, high)


def save_vs_spread(spread: VsSpread, path: Path) -> None:
    """`spread` as a row per depth. Written whole, then moved in place: a reader never finds half
    of it (PAC writes it on reading an inversion saved before it was, see load_vs_spread)."""
    lines = [_SPREAD_HEADER]
    for row in zip(spread.depths, spread.low, spread.middle, spread.high, strict=True):
        lines.append(",".join(f"{value:.6g}" for value in row))
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text("\n".join(lines) + "\n")
    partial.replace(path)


def load_vs_spread(folder: Path) -> VsSpread | None:
    """The kept models' Vs spread `save_vs_spread` wrote in window folder `folder`; None without
    its file, or with one of an earlier version (its header another: the inversions saved before
    2026-09-29, vs_spread makes it from their samples)."""
    path = folder / VS_SPREAD_FILE
    if not path.exists():
        return None
    with path.open() as file:
        if file.readline().strip() != _SPREAD_HEADER:
            return None
    rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return VsSpread(*(rows[:, k].copy() for k in range(4)))


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
