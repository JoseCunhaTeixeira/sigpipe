"""The line's velocity section and pseudo-section comparison, from its windows' inversions: PAC's
views and files. Each inverted window folder holds the five model variants sigpipe's inversion
saves (MODEL_NAMES), and the curves each predicts; PAC's default view is the smooth median.
Functions take a run folder and window folders of it (`units`, named xmid_<x>)."""

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from sigpipe.base.dispersion_curve import DispersionCurve, DispersionCurvesSection, Mode
from sigpipe.base.velocity_model import VelocityModel, VelocityModelsSection
from sigpipe.dataio.dispersion.loading import load_dispersion_curves
from sigpipe.dataio.dispersion.section import (
    plot_pseudo_section_comparison,
    pseudo_section_comparison_grids,
)
from sigpipe.dataio.inversion.forward import MODEL_NAMES
from sigpipe.dataio.velocity_model.loading import load_velocity_models
from sigpipe.dataio.velocity_model.section import (
    plot_velocity_and_std_section,
    save_velocity_models_sections,
)
from sigpipe.masw.inversion.measuring import INTERFACE_DZ
from sigpipe.masw.inversion.window import DZ, M0
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.transformers import Plot

logger = logging.getLogger(__name__)

# sigpipe's MODEL_NAMES, spelled out so that an API can validate one.
type ModelName = Literal["best", "smooth_best", "median", "smooth_median", "ensemble"]
# The model shown unless another is asked for: at each depth, the kept models' median Vs, with
# their spread (what the data support; the smooth models' curves are drawn, and fit worse).
DEFAULT_MODEL: ModelName = "ensemble"

SECTION_FIGURE = "SeismicInversion_VelocitySection_0000.png"
SECTION_FILE = "SeismicInversion_VelocitySection_0000.hdf5"
COMPARISON_FIGURE = "SeismicInversion_PseudoSectionComparison_0000_M0.png"
# The depths of a section drawn on screen (PAC's canvas is 200 to 320 px high): at the
# inversion's 1 cm, a modest line has about 4,000, 60 MB of JSON, for no visible difference.
VIEW_NZ = 200
# Smoothed along the line, a section's columns at most: it is spread over metres, not columns.
VIEW_NX = 600
# A window's length when not given: MASW windows are commonly about eight steps long.
STEPS_PER_WINDOW = 8
# The smoothing's full width at half height, of a window's length: over the whole of it (what a
# window's model describes) the section blurred; a third keeps what is a few windows wide.
SMOOTHED_SHARE = 1 / 3


def model_path(folder: Path, model: ModelName) -> Path:
    """Where window folder `folder` keeps its model `model`."""
    return folder / f"SeismicInversion_Model_0000_{model}.csv"


def predicted_path(folder: Path, model: ModelName) -> Path:
    """Where window folder `folder` keeps the curves its model `model` predicts."""
    return folder / f"SeismicInversion_DispersionCurves_0000_{model}.csv"


def is_inverted(folder: Path) -> bool:
    """Whether window folder `folder` holds every model variant of an inversion."""
    return all(model_path(folder, model).exists() for model in MODEL_NAMES)


def window_model(folder: Path, model: ModelName = DEFAULT_MODEL) -> VelocityModel | None:
    """The model `model` of window folder `folder`; None when it has none."""
    path = model_path(folder, model)
    return load_velocity_models([path])[0][0] if path.exists() else None


def models_section(
    run_folder: Path, units: Sequence[str], model: ModelName = DEFAULT_MODEL
) -> VelocityModelsSection | None:
    """The model `model` of each window folder of `units` that holds one, sorted by position;
    None with fewer than two, the least a section needs."""
    models = [
        found for unit in units if (found := window_model(run_folder / unit, model)) is not None
    ]
    if len(models) < 2:
        return None
    ordered = tuple(sorted(models, key=lambda one: one.position.x))
    return VelocityModelsSection(velocity_models=ordered)


@dataclass(frozen=True, slots=True)
class VelocityGrid:
    """A section on a grid: Vs and its spread by position and elevation."""

    positions: np.ndarray
    elevations: np.ndarray
    vs: np.ndarray  # (positions, elevations)
    vs_std: np.ndarray


def velocity_grid(
    section: VelocityModelsSection,
    lateral_smoothing: bool = False,
    nz: int = VIEW_NZ,
    window_m: float | None = None,
) -> VelocityGrid:
    """`section` on a grid of about `nz` elevations, its models sampled at them.

    Unsmoothed: one column per window, at its middle, empty above its ground.

    Smoothed along the line (`smoothed`): over SMOOTHED_SHARE of a window's length, `window_m`
    (by default STEPS_PER_WINDOW of the windows' median step).
    """
    models = section.velocity_models
    tops = np.array([model.position.z for model in models], dtype=float)
    bottom = min(model.position.z - sum(model.thicknesses) for model in models)
    n = int(np.floor((tops.max() - bottom) / DZ)) + 1
    elevations = (tops.max() - np.arange(n, dtype=np.float32) * DZ)[:: max(n // nz, 1)]
    xs = np.array([model.position.x for model in models], dtype=np.float32)
    vs = np.array([model.sample_vs(elevations) for model in models], dtype=np.float32)
    vs_std = np.array([model.sample_vs_std(elevations) for model in models], dtype=np.float32)
    if not lateral_smoothing:
        return VelocityGrid(positions=xs, elevations=elevations, vs=vs, vs_std=vs_std)
    positions = smoothed_positions(xs)
    ground = np.interp(positions, xs, tops)
    above = elevations[None, :] > ground[:, None] + 1e-3
    width = window_m or default_window(xs)
    return VelocityGrid(
        positions=positions,
        elevations=elevations,
        vs=smoothed(vs, xs, positions, width, above).astype(np.float32),
        vs_std=smoothed(vs_std, xs, positions, width, above).astype(np.float32),
    )


def smoothed_positions(xs: np.ndarray) -> np.ndarray:
    """The columns of a section smoothed along the line: every half of the windows' smallest step
    (each window two columns at least), VIEW_NX at most."""
    step = max(float(np.min(np.diff(xs))) / 2, 1e-6) if xs.size > 1 else 1.0
    count = min(int(np.floor((xs[-1] - xs[0]) / step)) + 1, VIEW_NX)
    return np.linspace(float(xs[0]), float(xs[-1]), max(count, 2), dtype=np.float32)


def default_window(xs: np.ndarray) -> float:
    """A window's length when not known: STEPS_PER_WINDOW of the windows' median step."""
    return STEPS_PER_WINDOW * float(np.median(np.diff(xs))) if xs.size > 1 else 1.0


def smoothed(
    values: np.ndarray,
    xs: np.ndarray,
    positions: np.ndarray,
    window_m: float,
    empty: np.ndarray | None = None,
) -> np.ndarray:
    """`values` (a row per window at `xs`, NaN where it has none) along the line at `positions`:
    each window's value the median of its and its two neighbours' (one odd model does not
    spread); between windows, linear from those holding a value; then a Gaussian along the line
    whose full width at half height is SMOOTHED_SHARE of `window_m`, a window's length. The
    cells `empty` (above the ground) left empty, and none of them weighed in."""
    padded = np.pad(np.asarray(values, dtype=float), ((1, 1), (0, 0)), mode="edge")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # a row without a value
        robust = np.nanmedian(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)
    out = np.full((positions.size, robust.shape[1]), np.nan)
    for j in range(robust.shape[1]):
        held = ~np.isnan(robust[:, j])
        if held.any():
            out[:, j] = np.interp(positions, xs[held], robust[held, j])
    if empty is not None:
        out[empty] = np.nan
    step = float(positions[1] - positions[0]) if positions.size > 1 else 1.0
    sigma = SMOOTHED_SHARE * window_m / (2 * np.sqrt(2 * np.log(2))) / step
    weights = (~np.isnan(out)).astype(float)
    total = gaussian_filter1d(np.nan_to_num(out) * weights, sigma, axis=0, mode="nearest")
    weight = gaussian_filter1d(weights, sigma, axis=0, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        result = total / weight
    result[weights == 0] = np.nan
    return result


def informed_levels(
    grid: VelocityGrid,
    windows: Sequence[tuple[float, float, float | None]],
    lateral_smoothing: bool = False,
    window_m: float | None = None,
) -> np.ndarray:
    """Per column of `grid` (`velocity_grid`'s, of the same windows): the elevation down to which
    the data inform the section, from each window's middle, ground elevation and depth informed
    (m; None: not known), NaN where not known. Smoothed along the line as the grid's Vs when
    `lateral_smoothing`: the depths, so that no level rises above its ground; a column nearest a
    window without one left without."""
    xs = np.array([x for x, _, _ in windows], dtype=np.float32)
    grounds = np.array([ground for _, ground, _ in windows], dtype=float)
    depths = np.array([np.nan if depth is None else depth for _, _, depth in windows], dtype=float)
    positions = grid.positions
    # Each column's window: its own unsmoothed, the nearest smoothed.
    nearest = np.abs(positions[:, None] - xs[None, :]).argmin(axis=1)
    if not lateral_smoothing:
        return (grounds - depths)[nearest]
    width = window_m or default_window(xs)
    along = smoothed(depths[:, None], xs, positions, width)[:, 0]
    along[np.isnan(depths)[nearest]] = np.nan
    return np.interp(positions, xs, grounds) - along


def interface_grid(
    grid: VelocityGrid,
    windows: Sequence[tuple[float, float, tuple[float, ...]]],
    lateral_smoothing: bool = False,
    window_m: float | None = None,
) -> np.ndarray:
    """On `grid` (`velocity_grid`'s, of the same windows): the share of the kept models with an
    interface at each cell, from each window's middle, ground elevation and shares per
    INTERFACE_DZ from its ground down (none: not known); NaN where not known or above the
    ground. Smoothed along the line as the grid's Vs when `lateral_smoothing`; a column nearest a
    window without shares left without."""
    xs = np.array([x for x, _, _ in windows], dtype=np.float32)
    grounds = np.array([ground for _, ground, _ in windows], dtype=float)
    values = np.full((len(windows), grid.elevations.size), np.nan)
    for i, (_, ground, shares) in enumerate(windows):
        which = np.floor((ground - grid.elevations) / INTERFACE_DZ).astype(int)
        inside = (which >= 0) & (which < len(shares))
        values[i, inside] = np.asarray(shares, dtype=float)[which[inside]]
    nearest = np.abs(grid.positions[:, None] - xs[None, :]).argmin(axis=1)
    unknown = np.array([not shares for _, _, shares in windows])[nearest]
    known = [i for i, (_, _, shares) in enumerate(windows) if shares]
    if not lateral_smoothing or not known:
        return values[nearest]
    ground = np.interp(grid.positions, xs, grounds)
    above = grid.elevations[None, :] > ground[:, None] + 1e-3
    width = window_m or default_window(xs)
    along = smoothed(values[known], xs[known], grid.positions, width, above)
    along[unknown] = np.nan
    return along


def section_suffix(model: ModelName, lateral_smoothing: bool) -> str:
    """The file name suffix of a view other than PAC's default (DEFAULT_MODEL, not smoothed
    laterally), which has none."""
    parts: list[str] = []
    if model != DEFAULT_MODEL:
        parts.append(model)
    if lateral_smoothing:
        parts.append("lateralsmooth")
    return ("_" + "_".join(parts)) if parts else ""


def save_section(
    run_folder: Path,
    units: Sequence[str],
    model: ModelName = DEFAULT_MODEL,
    lateral_smoothing: bool = False,
) -> Path | None:
    """The figure of the line's `model` and its spread, in `run_folder`; None with fewer than
    two windows holding the model."""
    section = models_section(run_folder, units, model)
    if section is None:
        return None
    figure = plot_velocity_and_std_section(section, dz=DZ, lateral_smoothing=lateral_smoothing)
    suffix = section_suffix(model, lateral_smoothing)
    path = run_folder / f"SeismicInversion_VelocitySection_0000{suffix}.png"
    Plot.savefig(path=path, figure=figure)
    plt.close(figure)
    return path


def save_sections_file(run_folder: Path, units: Sequence[str]) -> Path | None:
    """Every model variant's section in one HDF5 file in `run_folder`, a variant with fewer
    than two windows left out; None when none has two."""
    sections: dict[str, VelocityModelsSection] = {}
    for model in MODEL_NAMES:
        section = models_section(run_folder, units, model)
        if section is None:
            logger.warning(
                "No '%s' section in %s: fewer than two windows hold it", model, run_folder
            )
            continue
        sections[model] = section
    if not sections:
        return None
    path = run_folder / SECTION_FILE
    save_velocity_models_sections(sections, path, dz=DZ)
    return path


def picked_curve(folder: Path, mode: Mode = M0) -> DispersionCurve | None:
    """The curve of `mode` picked in window folder `folder`; None when it has none."""
    path = folder / CURVES_FILE
    if not path.exists():
        return None
    return next((curve for curve in load_dispersion_curves([path])[0] if curve.mode == mode), None)


def predicted_curve(
    folder: Path, observed: DispersionCurve, model: ModelName = DEFAULT_MODEL
) -> DispersionCurve | None:
    """The curve `model` predicts for `observed`'s mode, as the inversion saved it in window
    folder `folder`, at `observed`'s position (the forward model knows none); None when the
    inversion left that mode out. Modes match by number: the forward model labels its curves
    R, the pickers M."""
    path = predicted_path(folder, model)
    if not path.exists():
        return None
    number = observed.mode.number
    predicted = next(
        (curve for curve in load_dispersion_curves([path])[0] if curve.mode.number == number),
        None,
    )
    if predicted is None:
        return None
    return DispersionCurve(
        fs=predicted.fs,
        vs=predicted.vs,
        mode=predicted.mode,
        acquisition=observed.acquisition,
        type=predicted.type,
    )


def comparison_sections(
    run_folder: Path,
    units: Sequence[str],
    mode: Mode = M0,
    model: ModelName = DEFAULT_MODEL,
) -> tuple[DispersionCurvesSection, DispersionCurvesSection] | None:
    """The picked curves of `mode` along the line, and the curves `model` predicts for them,
    over the window folders of `units` holding both; None with fewer than two."""
    observed: list[DispersionCurve] = []
    predicted: list[DispersionCurve] = []
    for unit in units:
        picked = picked_curve(run_folder / unit, mode)
        modeled = None if picked is None else predicted_curve(run_folder / unit, picked, model)
        if picked is None or modeled is None:
            continue
        observed.append(picked)
        predicted.append(modeled)
    if len(observed) < 2:
        return None
    return (
        DispersionCurvesSection(dispersion_curves=tuple(observed)),
        DispersionCurvesSection(dispersion_curves=tuple(predicted)),
    )


def save_comparison(
    run_folder: Path,
    units: Sequence[str],
    mode: Mode = M0,
    model: ModelName = DEFAULT_MODEL,
) -> Path | None:
    """The figure of the picked curves of `mode` against the curves `model` predicts, along the
    line, in `run_folder`; None with fewer than two windows holding both."""
    sections = comparison_sections(run_folder, units, mode, model)
    if sections is None:
        return None
    figure = plot_pseudo_section_comparison(*sections)
    suffix = section_suffix(model, lateral_smoothing=False)
    path = run_folder / f"SeismicInversion_PseudoSectionComparison_0000{suffix}_{mode.label}.png"
    Plot.savefig(path=path, figure=figure)
    plt.close(figure)
    return path


@dataclass(frozen=True, slots=True)
class ComparisonGrids:
    """The pseudo-section comparison on grids: picked, predicted and their residual, by
    position and frequency, and by position and wavelength (as a pseudo-section's)."""

    positions: np.ndarray
    fs: np.ndarray
    observed: np.ndarray
    predicted: np.ndarray
    residual: np.ndarray
    lambdas: np.ndarray
    observed_by_wavelength: np.ndarray
    predicted_by_wavelength: np.ndarray
    residual_by_wavelength: np.ndarray


def grids_of(
    observed: DispersionCurvesSection, predicted: DispersionCurvesSection
) -> ComparisonGrids:
    """`observed` against `predicted` on both grids."""
    positions, fs, by_f_observed, by_f_predicted, by_f_residual = pseudo_section_comparison_grids(
        observed, predicted
    )
    _, lambdas, by_l_observed, by_l_predicted, by_l_residual = pseudo_section_comparison_grids(
        observed, predicted, along="wavelength"
    )
    return ComparisonGrids(
        positions,
        fs,
        by_f_observed,
        by_f_predicted,
        by_f_residual,
        lambdas,
        by_l_observed,
        by_l_predicted,
        by_l_residual,
    )


def comparison_grids(
    run_folder: Path,
    units: Sequence[str],
    mode: Mode = M0,
    model: ModelName = DEFAULT_MODEL,
) -> ComparisonGrids | None:
    """The data behind save_comparison's figure; None with fewer than two windows."""
    sections = comparison_sections(run_folder, units, mode, model)
    if sections is None:
        return None
    return grids_of(*sections)
