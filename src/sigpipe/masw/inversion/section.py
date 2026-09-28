"""The line's velocity section and pseudo-section comparison, from its windows' inversions: PAC's
views and files. Each inverted window folder holds the five model variants sigpipe's inversion
saves (MODEL_NAMES), and the curves each predicts; PAC's default view is the smooth median.
Functions take a run folder and window folders of it (`units`, named xmid_<x>)."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

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
    smooth_laterally,
)
from sigpipe.masw.inversion.window import DZ, M0
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.transformers import Plot

logger = logging.getLogger(__name__)

# sigpipe's MODEL_NAMES, spelled out so that an API can validate one.
type ModelName = Literal["best", "smooth_best", "median", "smooth_median", "ensemble"]

SECTION_FIGURE = "SeismicInversion_VelocitySection_0000.png"
SECTION_FILE = "SeismicInversion_VelocitySection_0000.hdf5"
COMPARISON_FIGURE = "SeismicInversion_PseudoSectionComparison_0000_M0.png"
# The depths of a section drawn on screen (PAC's canvas is 200 to 320 px high): at the
# inversion's 1 cm, a modest line has about 4,000, 60 MB of JSON, for no visible difference.
VIEW_NZ = 200


def model_path(folder: Path, model: ModelName) -> Path:
    """Where window folder `folder` keeps its model `model`."""
    return folder / f"SeismicInversion_Model_0000_{model}.csv"


def predicted_path(folder: Path, model: ModelName) -> Path:
    """Where window folder `folder` keeps the curves its model `model` predicts."""
    return folder / f"SeismicInversion_DispersionCurves_0000_{model}.csv"


def is_inverted(folder: Path) -> bool:
    """Whether window folder `folder` holds every model variant of an inversion."""
    return all(model_path(folder, model).exists() for model in MODEL_NAMES)


def window_model(folder: Path, model: ModelName = "smooth_median") -> VelocityModel | None:
    """The model `model` of window folder `folder`; None when it has none."""
    path = model_path(folder, model)
    return load_velocity_models([path])[0][0] if path.exists() else None


def models_section(
    run_folder: Path, units: Sequence[str], model: ModelName = "smooth_median"
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
    section: VelocityModelsSection, lateral_smoothing: bool = False, nz: int = VIEW_NZ
) -> VelocityGrid:
    """`section` on a grid of about `nz` elevations.

    Unsmoothed: one column per inverted position, not to_grid's regular x-grid, whose
    nearest-neighbour lookup breaks ties toward the lower-x position (numpy's argmin picks the
    first match) and shifts every position's band off centre.

    Smoothed: resampled first onto dx = half the smallest gap between adjacent positions (two
    columns across the tightest gap), then smoothed laterally.

    Either way the models are sampled every DZ (so that the thin layers of the smooth variants
    are not skipped, as to_grid itself requires), and the grid thinned to `nz` elevations after.
    """
    models = section.velocity_models
    if lateral_smoothing:
        xs_sorted = sorted(model.position.x for model in models)
        dx = max(min(b - a for a, b in pairwise(xs_sorted)) / 2, 1e-6)
        xs, zs_fine, vs, _vp, _rho, vs_std = section.to_grid(dz=DZ, dx=dx)
    else:
        tops = [model.position.z for model in models]
        bottoms = [model.position.z - sum(model.thicknesses) for model in models]
        n = int(np.floor((max(tops) - min(bottoms)) / DZ)) + 1
        zs_fine = (max(tops) - np.arange(n, dtype=np.float32) * DZ).astype(np.float32)
        xs = np.array([model.position.x for model in models], dtype=np.float32)
        vs = np.array([model.sample_vs(zs_fine) for model in models], dtype=np.float32)
        vs_std = np.array([model.sample_vs_std(zs_fine) for model in models], dtype=np.float32)

    stride = max(len(zs_fine) // nz, 1)
    vs, vs_std = vs[:, ::stride], vs_std[:, ::stride]
    if lateral_smoothing:
        # The median across positions skips the empty cells above a window's ground: kept
        # empty after it, or a higher neighbour's Vs would rise above the ground where it steps.
        above = np.isnan(vs)
        vs, vs_std = smooth_laterally(vs), smooth_laterally(vs_std)
        vs[above] = np.nan
        vs_std[above] = np.nan
    return VelocityGrid(positions=xs, elevations=zs_fine[::stride], vs=vs, vs_std=vs_std)


def informed_levels(
    grid: VelocityGrid,
    windows: Sequence[tuple[float, float, float | None]],
    lateral_smoothing: bool = False,
) -> np.ndarray:
    """Per column of `grid` (`velocity_grid`'s, of the same windows): the elevation down to which
    the data inform the section, from each window's middle, ground elevation and depth informed
    (m; None: not known), NaN where not known. Smoothed across positions as the grid's Vs when
    `lateral_smoothing`: the depths, so that no level rises above its ground, and a column whose
    window has none left without."""
    xs = np.array([x for x, _, _ in windows], dtype=np.float32)
    # Each column's window, as to_grid picks it: the nearest (its own, unsmoothed).
    nearest = np.abs(grid.positions[:, None] - xs[None, :]).argmin(axis=1)
    grounds = np.array([ground for _, ground, _ in windows], dtype=float)[nearest]
    known = [np.nan if depth is None else depth for _, _, depth in windows]
    depths = np.array(known, dtype=float)[nearest]
    if lateral_smoothing:
        unknown = np.isnan(depths)
        depths = smooth_laterally(depths[:, None])[:, 0]
        depths[unknown] = np.nan
    return grounds - depths


def section_suffix(model: ModelName, lateral_smoothing: bool) -> str:
    """The file name suffix of a view other than PAC's default (the smooth median, not
    smoothed laterally), which has none."""
    parts: list[str] = []
    if model != "smooth_median":
        parts.append(model)
    if lateral_smoothing:
        parts.append("lateralsmooth")
    return ("_" + "_".join(parts)) if parts else ""


def save_section(
    run_folder: Path,
    units: Sequence[str],
    model: ModelName = "smooth_median",
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
    folder: Path, observed: DispersionCurve, model: ModelName = "smooth_median"
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
    model: ModelName = "smooth_median",
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
    model: ModelName = "smooth_median",
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
    model: ModelName = "smooth_median",
) -> ComparisonGrids | None:
    """The data behind save_comparison's figure; None with fewer than two windows."""
    sections = comparison_sections(run_folder, units, mode, model)
    if sections is None:
        return None
    return grids_of(*sections)
