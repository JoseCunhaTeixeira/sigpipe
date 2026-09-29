"""The line's velocity section and pseudo-section comparison, from its windows' inversions: PAC's
views and files. Each inverted window folder holds the five model variants sigpipe's inversion
saves (MODEL_NAMES), and the curves each predicts; PAC's default view is the smooth median.
Functions take a run folder and window folders of it (`units`, named xmid_<x>)."""

import logging
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d

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
from sigpipe.masw.inversion.window import DZ, M0, VsSpread
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
    """A section on a grid: Vs and its spread by position and elevation, and each column's
    ground and floor (the elevation its models end at): empty above the one and below the
    other."""

    positions: np.ndarray
    elevations: np.ndarray
    vs: np.ndarray  # (positions, elevations)
    vs_std: np.ndarray
    ground: np.ndarray  # per column
    floor: np.ndarray  # per column

    def outside(self) -> np.ndarray:
        """The cells above a column's ground or below its floor."""
        z = self.elevations[None, :]
        return (z > self.ground[:, None] + 1e-3) | (z < self.floor[:, None] - 1e-3)


def velocity_grid(
    section: VelocityModelsSection,
    lateral_smoothing: bool = False,
    nz: int = VIEW_NZ,
    window_m: float | None = None,
    depths: Sequence[float] | None = None,
) -> VelocityGrid:
    """`section` on a grid of about `nz` elevations, its models sampled at them, each column
    from its ground down to the depth its model was built to (`depths`, one a model; by default
    each model's own), its half-space carried no deeper.

    Unsmoothed: one column per window, at its middle.

    Smoothed along the line (`smoothed`): over SMOOTHED_SHARE of a window's length, `window_m`
    (by default STEPS_PER_WINDOW of the windows' median step), the depths the models reach too;
    the ground straight from a window's middle to the next.
    """
    models = section.velocity_models
    tops = np.array([model.position.z for model in models], dtype=float)
    reach = [sum(model.thicknesses) for model in models] if depths is None else depths
    bottoms = tops - np.asarray(reach, dtype=float)
    n = int(np.floor((tops.max() - bottoms.min()) / DZ)) + 1
    elevations = (tops.max() - np.arange(n, dtype=np.float32) * DZ)[:: max(n // nz, 1)]
    xs = np.array([model.position.x for model in models], dtype=np.float32)
    vs = np.array([model.sample_vs(elevations) for model in models], dtype=np.float32)
    vs_std = np.array([model.sample_vs_std(elevations) for model in models], dtype=np.float32)
    plain = VelocityGrid(xs, elevations, vs, vs_std, tops, bottoms)
    vs[plain.outside()] = np.nan
    vs_std[plain.outside()] = np.nan
    if not lateral_smoothing:
        return plain
    positions = smoothed_positions(xs)
    width = window_m or default_window(xs)
    ground = np.interp(positions, xs, tops)
    # The depths the models reach smoothed as their Vs: the section's bottom a smooth line.
    reached = smoothed((tops - bottoms)[:, None], xs, positions, width)[:, 0]
    grid = VelocityGrid(positions, elevations, vs, vs_std, ground, ground - reached)
    empty = grid.outside()
    return VelocityGrid(
        positions=positions,
        elevations=elevations,
        vs=smoothed(vs, xs, positions, width, empty).astype(np.float32),
        vs_std=smoothed(vs_std, xs, positions, width, empty).astype(np.float32),
        ground=grid.ground,
        floor=grid.floor,
    )


def smoothed_positions(xs: np.ndarray, at_least: int = 0) -> np.ndarray:
    """The columns of a section smoothed along the line: every half of the windows' smallest step
    (each window two columns at least), `at_least` of them, VIEW_NX at most."""
    step = max(float(np.min(np.diff(xs))) / 2, 1e-6) if xs.size > 1 else 1.0
    count = min(max(int(np.floor((xs[-1] - xs[0]) / step)) + 1, at_least), VIEW_NX)
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
    out = _interpolated(values, xs, positions, empty)
    step = float(positions[1] - positions[0]) if positions.size > 1 else 1.0
    sigma = SMOOTHED_SHARE * window_m / (2 * np.sqrt(2 * np.log(2))) / step
    weights = (~np.isnan(out)).astype(float)
    total = gaussian_filter1d(np.nan_to_num(out) * weights, sigma, axis=0, mode="nearest")
    weight = gaussian_filter1d(weights, sigma, axis=0, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        result = total / weight
    result[weights == 0] = np.nan
    return result


def _median_of_three(values: np.ndarray) -> np.ndarray:
    """Each window's values (a row each) the median of its and its two neighbours' (the line's
    ends their own twice), NaN left out."""
    padded = np.pad(np.asarray(values, dtype=float), ((1, 1), (0, 0)), mode="edge")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # a row without a value
        return np.nanmedian(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)


def _interpolated(
    values: np.ndarray, xs: np.ndarray, positions: np.ndarray, empty: np.ndarray | None
) -> np.ndarray:
    """`smoothed` before its Gaussian: each window's value the median of its and its two
    neighbours', linear between windows, NaN in the cells `empty`."""
    robust = _median_of_three(values)
    out = np.full((positions.size, robust.shape[1]), np.nan)
    for j in range(robust.shape[1]):
        held = ~np.isnan(robust[:, j])
        if held.any():
            out[:, j] = np.interp(positions, xs[held], robust[held, j])
    if empty is not None:
        out[empty] = np.nan
    return out


def smoothed_categories(
    labels: np.ndarray,
    xs: np.ndarray,
    positions: np.ndarray,
    dz: float,
    empty: np.ndarray | None = None,
) -> np.ndarray:
    """Categories (`labels`, a row per window at `xs`, a label per cell `dz` apart down its
    column, -1 where it has none) along the line at `positions`, as `smoothed` takes values
    along it before its Gaussian: each category's signed distance to its edges (inside it, how
    far to another; outside, minus how far to it) the median of a window's and its neighbours',
    linear between windows, the largest winning among those the two windows around hold there.
    A boundary deeper in one window than in the next runs straight between them, a category
    one window alone holds does not spread, one two windows hold ends just past them, and none
    shows between two windows that neither holds at that depth. No Gaussian: it would blur a
    thin layer away. -1 in the cells `empty`, and where no window holds a label."""
    labels = np.asarray(labels)
    kinds = np.unique(labels[labels >= 0])
    out = np.full((positions.size, labels.shape[1]), -1, dtype=int)
    if kinds.size == 0:
        return out
    far = labels.shape[1] * dz  # farther than any edge of a column
    robust = np.stack(
        [
            _median_of_three(
                np.array([_signed_distance(column, kind, dz, far) for column in labels])
            )
            for kind in kinds
        ]
    )  # (kinds, windows, cells)
    for j in range(labels.shape[1]):
        held = ~np.isnan(robust[0, :, j])
        if not held.any():
            continue
        at, values = xs[held], robust[:, held, j]
        # Each window's category there: the one it is inside, or the nearest to it.
        own = values.argmax(axis=0)
        if at.size == 1:
            out[:, j] = kinds[own[0]]
            continue
        right = np.clip(np.searchsorted(at, positions), 1, at.size - 1)
        left = right - 1
        t = np.clip((positions - at[left]) / (at[right] - at[left]), 0.0, 1.0)
        between = values[:, left] * (1 - t) + values[:, right] * t  # (kinds, positions)
        which = np.arange(kinds.size)[:, None]
        allowed = (which == own[left]) | (which == own[right])
        out[:, j] = kinds[np.where(allowed, between, -np.inf).argmax(axis=0)]
    if empty is not None:
        out[empty] = -1
    return out


def _signed_distance(column: np.ndarray, kind: int, dz: float, far: float) -> np.ndarray:
    """Down a column of labels (-1: none): inside `kind`, how far (m) to another label, `far`
    without one; elsewhere, minus how far to `kind`, minus `far` without it; NaN where none."""
    inside = column == kind
    other = (column >= 0) & ~inside
    out = np.full(column.shape, np.nan)
    if inside.any():
        out[inside] = _cells_to_false(~other)[inside] * dz if other.any() else far
    if other.any():
        out[other] = -_cells_to_false(~inside)[other] * dz if inside.any() else -far
    return out


def _cells_to_false(mask: np.ndarray) -> np.ndarray:
    """How many cells from each True cell of `mask` to the nearest False one."""
    # An array: no output array is given.
    return cast(np.ndarray, distance_transform_edt(mask))


@dataclass(frozen=True, slots=True)
class AlongLine:
    """A section's columns smoothed along the line (`along_line`): their positions, ground and
    floor, the cells outside them, and the smoothing's window length."""

    positions: np.ndarray
    ground: np.ndarray
    floor: np.ndarray
    empty: np.ndarray  # (positions, elevations)
    window_m: float


def along_line(
    xs: np.ndarray,
    tops: np.ndarray,
    reach: np.ndarray,
    elevations: np.ndarray,
    window_m: float | None = None,
    at_least: int = 0,
) -> AlongLine:
    """The columns of windows at `xs` (their grounds `tops`, reaching `reach` m down) smoothed
    along the line as `velocity_grid` smooths them: `smoothed_positions` (`at_least` of them),
    the ground straight from a window's middle to the next, the depths reached smoothed; over
    `window_m`, by default `default_window`."""
    positions = smoothed_positions(xs, at_least)
    width = window_m or default_window(xs)
    ground = np.interp(positions, xs, tops)
    floor = ground - smoothed(np.asarray(reach, dtype=float)[:, None], xs, positions, width)[:, 0]
    z = elevations[None, :]
    empty = (z > ground[:, None] + 1e-3) | (z < floor[:, None] - 1e-3)
    return AlongLine(positions, ground, floor, empty, width)


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
    depths = np.array([np.nan if depth is None else depth for _, _, depth in windows], dtype=float)
    positions = grid.positions
    # Each column's window: its own unsmoothed, the nearest smoothed.
    nearest = np.abs(positions[:, None] - xs[None, :]).argmin(axis=1)
    if not lateral_smoothing:
        return grid.ground - depths[nearest]
    width = window_m or default_window(xs)
    along = smoothed(depths[:, None], xs, positions, width)[:, 0]
    along[np.isnan(depths)[nearest]] = np.nan
    return grid.ground - along


def interface_grid(
    grid: VelocityGrid,
    windows: Sequence[tuple[float, float, tuple[float, ...]]],
    lateral_smoothing: bool = False,
    window_m: float | None = None,
) -> np.ndarray:
    """On `grid` (`velocity_grid`'s, of the same windows): the share of the kept models with an
    interface at each cell, from each window's middle, ground elevation and shares per
    INTERFACE_DZ from its ground down (none: not known); NaN where not known, and outside the
    grid's columns (above their ground, below their floor). Smoothed along the line as the
    grid's Vs when `lateral_smoothing`; a column nearest a window without shares left
    without."""
    xs = np.array([x for x, _, _ in windows], dtype=np.float32)
    values = np.full((len(windows), grid.elevations.size), np.nan)
    for i, (_, ground, shares) in enumerate(windows):
        which = np.floor((ground - grid.elevations) / INTERFACE_DZ).astype(int)
        inside = (which >= 0) & (which < len(shares))
        values[i, inside] = np.asarray(shares, dtype=float)[which[inside]]
    nearest = np.abs(grid.positions[:, None] - xs[None, :]).argmin(axis=1)
    unknown = np.array([not shares for _, _, shares in windows])[nearest]
    known = [i for i, (_, _, shares) in enumerate(windows) if shares]
    if not lateral_smoothing or not known:
        along = values[nearest]
    else:
        width = window_m or default_window(xs)
        along = smoothed(values[known], xs[known], grid.positions, width, grid.outside())
        along[unknown] = np.nan
    along[grid.outside()] = np.nan
    return along


def uncertainty_grid(
    grid: VelocityGrid,
    windows: Sequence[tuple[float, float, VsSpread | None]],
    lateral_smoothing: bool = False,
    window_m: float | None = None,
) -> np.ndarray:
    """On `grid` (`velocity_grid`'s, of the same windows): the kept models' relative uncertainty
    of Vs, U(z) = (P90 - P10) / (2 P50) (what the depth informed is read from), from each
    window's middle, ground elevation and spread (None: not known); `_spread_grid`'s."""
    return _spread_grid(grid, windows, VsSpread.uncertainty, lateral_smoothing, window_m)


def correlation_grid(
    grid: VelocityGrid,
    windows: Sequence[tuple[float, float, VsSpread | None]],
    lateral_smoothing: bool = False,
    window_m: float | None = None,
) -> np.ndarray:
    """On `grid`: how far below each depth (m) the kept models' Vs stays correlated with its own
    (VsSpread.correlation), from each window's middle, ground elevation and spread (None: not
    known); `_spread_grid`'s."""
    return _spread_grid(
        grid, windows, lambda spread: spread.correlation, lateral_smoothing, window_m
    )


def _spread_grid(
    grid: VelocityGrid,
    windows: Sequence[tuple[float, float, VsSpread | None]],
    measure: Callable[[VsSpread], np.ndarray],
    lateral_smoothing: bool,
    window_m: float | None,
) -> np.ndarray:
    """`measure` of each window's spread (by depth) on `grid`, from its ground down to its last
    depth; NaN where not known, and outside the grid's columns. Smoothed along the line as the
    grid's Vs when `lateral_smoothing`; a column nearest a window without a spread left
    without."""
    xs = np.array([x for x, _, _ in windows], dtype=np.float32)
    values = np.full((len(windows), grid.elevations.size), np.nan)
    for i, (_, ground, spread) in enumerate(windows):
        if spread is not None:
            # From the ground (its first cell's), down to its last cell.
            values[i] = np.interp(
                ground - grid.elevations, spread.depths, measure(spread), right=np.nan
            )
    nearest = np.abs(grid.positions[:, None] - xs[None, :]).argmin(axis=1)
    unknown = np.array([spread is None for _, _, spread in windows])[nearest]
    known = [i for i, (_, _, spread) in enumerate(windows) if spread is not None]
    if not lateral_smoothing or not known:
        along = values[nearest]
    else:
        width = window_m or default_window(xs)
        along = smoothed(values[known], xs[known], grid.positions, width, grid.outside())
        along[unknown] = np.nan
    along[grid.outside()] = np.nan
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
