"""One window's inversion, on a window folder of a run: sigpipe's MCMC on the window's picked
curves, and PAC's files beside them."""

import logging
import warnings
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict

from sigpipe.algorithms.inversion.rayleigh.seismic.data import NOISE_BOUNDS
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
from sigpipe.dataio.inversion.plotting import (
    plot_chains,
    plot_inversion_window,
    plot_posterior_marginals,
)
from sigpipe.dataio.velocity_model.loading import load_velocity_models
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
# The figures saved beside an inversion, as PAC shows them (draw_figures).
IMAGE_FIGURE = "SeismicInversion_DispersionImage_0000.png"
WINDOW_FIGURE = "SeismicInversion_DensityCurves_0000.png"
MARGINALS_FIGURE = "SeismicInversion_Marginals_0000.png"
CHAINS_FIGURE = "SeismicInversion_Chains_0000.png"
M0 = Mode("M", 0)  # the fundamental mode, as the pickers label it


class WindowParameters(BaseModel):
    """What a window's chains ran with: the layering and its priors (the free layering's bounds
    as found from the curves), the chains' effort, how often each chain accepted a move, and
    each sampled parameter's typical move (the fixed layering)."""

    model_config = ConfigDict(frozen=True)

    parameters: SavedInversionParameters  # as run
    # The trial runs (step factor, acceptance %) of runs saved by older versions.
    tuning: tuple[tuple[float, float], ...] = ()
    acceptance: tuple[float, ...]  # each chain's over the run (%), the burn-in included
    # Each sampled parameter's typical move (vs1, ..., thick1, ...); when the data chose the
    # layers, each move's step (vs, interface, noise, shift, stretch: in the logarithm of the
    # values moved), with each move's acceptance and the exchanges between tempered copies (%,
    # the chains' medians; empty in older runs).
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

    # Each model's curves at the picked frequencies, saved beside it; the figures drawn from the
    # files saved.
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
    draw_figures(folder, output_folder)
    return result


def draw_figures(folder: Path, output_folder: Path | None = None) -> None:
    """Draw the figures of the inversion saved in window folder `folder`, or in `output_folder` (a
    staging folder, the picks and the image staying in `folder`), from its files alone: the
    dispersion image with the median of the ensemble's curves (IMAGE_FIGURE), the window as PAC
    shows it (WINDOW_FIGURE), the kept models' marginals (MARGINALS_FIGURE) and the chains as they
    ran (CHAINS_FIGURE). An inversion saved before them is drawn again the same way. A figure that
    fails is logged and left out: it must not lose the inversion."""
    out = output_folder or folder
    drawers = {
        IMAGE_FIGURE: lambda: _image_figure(folder, out),
        WINDOW_FIGURE: lambda: _window_figure(folder, out),
        MARGINALS_FIGURE: lambda: _marginals_figure(out),
        CHAINS_FIGURE: lambda: _chains_figure(folder, out),
    }
    for name, draw in drawers.items():
        figure = None
        try:
            figure = draw()
            Plot.savefig(path=out / name, figure=figure)
        except Exception:  # a figure must not lose the inversion
            logger.exception("Could not draw %s in %s", name, out)
        finally:
            if figure is not None:
                plt.close(figure)


def _drawn_curves(folder: Path, out: Path) -> tuple[DispersionCurves, DispersionCurves | None]:
    """The picked curves the inversion inverted, and the median of the ensemble's at their
    frequencies (None without them): the modes it modelled, by number (the picks label M0 what
    the forward model labels R0), M0 when it modelled none."""
    path = out / "SeismicInversion_DispersionCurves_0000_ensemble.csv"
    modelled = load_dispersion_curves([path])[0] if path.exists() else None
    numbers = {curve.mode.number for curve in modelled} if modelled is not None else {0}
    picked = load_dispersion_curves([folder / CURVES_FILE])[0]
    modes = tuple(curve.mode for curve in picked if curve.mode.number in numbers)
    return _curves(folder, modes or (M0,)), modelled


def _image_figure(folder: Path, out: Path) -> Figure:
    """The dispersion image with the picks and the median of the ensemble's curves."""
    image = load_image(folder)
    curves, modelled = _drawn_curves(folder, out)
    return plot_dispersion_image(
        image,
        picked_curves=curves,
        modeled_curves=modelled,
        # Where the checks' flags start: under lbmin the aliasing zone, over lbmax beyond the
        # window's reach.
        lbmin=min_resolvable_wavelength(image.acquisition),
        lbmax=longest_reached_wavelength(image.acquisition),
        normalize=True,
        show_errorbars=True,
    )


def _window_figure(folder: Path, out: Path) -> Figure:
    """The window as PAC shows it: the picks with the median of the ensemble's curves and the
    kept models' 10-90 %; the median of the ensemble's Vs with the models' band, its uncertainty
    U, the depth informed at the checks' default limit and where the models place interfaces."""
    from sigpipe.masw.inversion.measuring import interface_shares, useful_depth  # imports this

    curves, modelled = _drawn_curves(folder, out)
    ran = load_parameters(out / PARAMETERS_FILE).parameters
    profiles = load_profiles(out / SAMPLES_FILE)
    spread = load_vs_spread(out)
    if spread is None:  # an inversion saved before its band was
        spread = vs_spread(profiles, ran.bottom)
    saved = load_spread(out)
    curve_spreads = {
        curve.mode.number: (saved[curve.mode.label].low, saved[curve.mode.label].high)
        for curve in curves
        if curve.mode.label in saved
    }
    return plot_inversion_window(
        curves,
        modelled,
        curve_spreads,
        load_velocity_models([out / "SeismicInversion_Model_0000_ensemble.csv"])[0][0],
        (spread.depths, spread.low, spread.high, spread.uncertainty()),
        useful_depth(spread, profiles, 0.25),
        interface_shares(profiles, ran.bottom),
        explored=(profiles, load_misfits(out / SAMPLES_FILE)),
    )


def _marginals_figure(out: Path) -> Figure:
    """The kept models' marginals (marginals)."""
    samples, _ = load_samples(out / SAMPLES_FILE)
    ran = load_parameters(out / PARAMETERS_FILE).parameters
    return plot_posterior_marginals(marginals(samples, load_profiles(out / SAMPLES_FILE), ran))


def _chains_figure(folder: Path, out: Path) -> Figure:
    """Each chain's saved samples of what it sampled (chain_series) along the run, Vs at the
    depths the checks watch included, as PAC's Chains view draws them."""
    from sigpipe.masw.inversion.measuring import watched_series  # it imports this module

    ran = load_parameters(out / PARAMETERS_FILE).parameters
    samples, _ = load_samples(out / SAMPLES_FILE)
    profiles = load_profiles(out / SAMPLES_FILE)
    series = chain_series(samples, profiles, ran, watched_series(folder, ran.bottom))
    priors = series_priors(ran, series)
    return plot_chains(
        {series_label(name): profiles.per_chain(values) for name, values in series.items()},
        {series_label(name): bounds for name, bounds in priors.items()},
    )


def chain_series(
    samples: dict[str, np.ndarray],
    profiles: LayeredSamples,
    parameters: InversionParameters,
    watched: Sequence[str] = (),
) -> dict[str, np.ndarray]:
    """What the chains sampled, one value per kept model (chain after chain), in series_order:
    Vs at the depths watched (`watched`: "vs@2.5m", ...: measuring's), the layers' own values
    when given (a value fixed left out: `parameters`), the number of layers, the noise factor."""
    series: dict[str, np.ndarray] = {}
    if watched:
        at = profiles.at(np.array([depth_of(name) for name in watched]))
        series = {name: at[:, j] for j, name in enumerate(watched)}
    fixed = parameters.fixed()
    series |= {name: values for name, values in samples.items() if name not in fixed}
    return dict(sorted(series.items(), key=lambda item: series_order(item[0])))


def depth_of(name: str) -> float:
    """The depth of a series of Vs at a depth: "vs@2.5m" is 2.5."""
    return float(name.removeprefix("vs@").removesuffix("m"))


def series_order(name: str) -> tuple[int, float]:
    """Vs at depths from the top, the layers' Vs then thicknesses from the top, the number of
    layers, the noise factor."""
    if name.startswith("vs@"):
        return 0, depth_of(name)
    if name.startswith("vs"):
        return 1, float(name.removeprefix("vs") or 0)
    if name.startswith("thick"):
        return 2, float(name.removeprefix("thick") or 0)
    return (3, 0.0) if name == "layers" else (4, 0.0)


def series_label(name: str) -> str:
    """A series as PAC names it: Vs at 2.5 m [m/s], Vs1 [m/s], H1 [m], Layers, Noise factor."""
    if name.startswith("vs@"):
        return f"Vs at {depth_of(name):g} m [m/s]"
    if name.startswith("vs"):
        return f"Vs{name.removeprefix('vs')} [m/s]"
    if name.startswith("thick"):
        return f"H{name.removeprefix('thick')} [m]"
    return {"layers": "Layers", "noise": "Noise factor"}.get(name, name)


def series_priors(
    parameters: InversionParameters, names: Iterable[str]
) -> dict[str, tuple[float, float]]:
    """Each series' prior bounds, by name: a layer's own; Vs at a depth, the widest any layer
    allows; the number of layers, 1 to the most allowed; the noise factor's. A value fixed has
    none."""
    if parameters.layering == "free":
        free = parameters.free
        vs_range = (free.vs_min or 0.0, free.vs_max or 0.0)
        layers = (1.0, float(free.max_layers))
    else:
        vs_range = (
            min(layer.vs_min for layer in parameters.vs_layers),
            max(layer.vs_max for layer in parameters.vs_layers),
        )
        layers = (float(parameters.n_layers), float(parameters.n_layers))
    own = {
        f"vs{i + 1}": (layer.vs_min, layer.vs_max) for i, layer in enumerate(parameters.vs_layers)
    } | {
        f"thick{i + 1}": (layer.thickness_min, layer.thickness_max)
        for i, layer in enumerate(parameters.thickness_layers)
    }
    fixed = parameters.fixed()
    found: dict[str, tuple[float, float]] = {}
    for name in names:
        if name.startswith("vs@"):
            found[name] = vs_range
        elif name == "layers":
            found[name] = layers
        elif name == "noise":
            found[name] = NOISE_BOUNDS
        elif parameters.layering == "fixed" and name in own and name not in fixed:
            found[name] = own[name]
    return found


def marginals(
    samples: dict[str, np.ndarray], profiles: LayeredSamples, parameters: InversionParameters
) -> dict[str, np.ndarray]:
    """What the marginals figure shows, one value per kept model (their named `samples`, their
    layers `profiles`, run with `parameters`): each sampled value of the fixed layering; the
    number of layers and the Vs at three depths, from the shallowest interface allowed to the
    deepest, when the data chose the layers; the noise factor. A value the same in every model
    has no density: left out."""
    shown: dict[str, np.ndarray] = {}
    for name, values in samples.items():
        if name.startswith("vs") and name[2:].isdigit():
            shown[f"Vs{name[2:]} [m/s]"] = values
        elif name.startswith("thick") and name[5:].isdigit():
            shown[f"H{name[5:]} [m]"] = values
    if "layers" in samples:
        shown["Layers"] = samples["layers"]
        top, bottom = parameters.free.depth_min, parameters.free.depth_max
        if top and bottom:
            depths = np.round(np.geomspace(top, bottom, 3), 1)
            at = profiles.at(depths)
            shown |= {f"Vs at {depth:g} m [m/s]": at[:, i] for i, depth in enumerate(depths)}
    if "noise" in samples:
        shown["Noise factor"] = samples["noise"]
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


def load_misfits(path: Path) -> np.ndarray:
    """Each kept model's misfit, as `save_samples` wrote them, the models' order."""
    with np.load(path) as saved:
        return np.asarray(saved["misfits"], dtype=float)


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
    its file (inversions saved by older versions)."""
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
    its file, or with one of an earlier version (its header another: vs_spread makes it from the
    inversion's samples)."""
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
