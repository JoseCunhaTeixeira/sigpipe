"""The line's petrophysical views and files, from its windows' petrophysical inversions (PAC's):
the soils and their N values, the rock physics with depth (shear modulus and Vs), and the picked
curves against those the models give back. Functions take a run folder and window folders of it
(`units`); each needs two inverted windows, the least a section has, and returns None with
fewer."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from sigpipe.base.dispersion_curve import DispersionCurve, DispersionCurvesSection
from sigpipe.base.petro_model import PetroModel, PetroModelsSection, SoilType
from sigpipe.dataio.dispersion.section import pseudo_section_comparison_grids
from sigpipe.dataio.petro_model.section import plot_petro_models_section
from sigpipe.dataio.plot_config import CM, DISP_DPI, DOUBLE_COLUMN_CM, HEIGHT_CM
from sigpipe.masw.inversion.section import VIEW_NZ, ComparisonGrids
from sigpipe.masw.petro.window import (
    QUANTITIES,
    Quantity,
    fundamental_curve,
    load_modeled_curve,
    load_petro_model,
    load_profile,
)
from sigpipe.transformers import Plot

SECTION_FIGURE = "PetroInversion_Section_0000.png"
SECTION_FILE = "PetroInversion_Section_0000.hdf5"
# The rock physics' sampling step (petro.forward.rock_physics' default), the grid its sections
# share.
PROFILE_DZ = 0.01


def petro_models(run_folder: Path, units: Sequence[str]) -> PetroModelsSection | None:
    """The petrophysical model of each window folder of `units` that holds one, sorted by
    position."""
    models = [found for unit in units if (found := load_petro_model(run_folder / unit)) is not None]
    if len(models) < 2:
        return None
    return PetroModelsSection(petro_models=tuple(sorted(models, key=lambda one: one.position.x)))


@dataclass(frozen=True, slots=True)
class PetroGrid:
    """The soils and N values on screen: one column per inverted window (soils and N are
    categorical, nothing to interpolate between windows), at most VIEW_NZ elevations."""

    positions: np.ndarray
    elevations: np.ndarray
    # A cell below a window's own column holds "" (sample_soil's fill round-trips numpy's object
    # array as a str), not SoilType.NONE.
    soil_grid: list[list[SoilType | str]]
    n_grid: np.ndarray
    water_table_elevations: np.ndarray


def petro_grid(section: PetroModelsSection) -> PetroGrid:
    models: Sequence[PetroModel] = section.petro_models
    tops = [model.position.z for model in models]
    bottoms = [model.position.z - sum(model.thicknesses) for model in models]
    dz = min(min(model.thicknesses) for model in models) / 10
    nz = int(np.floor((max(tops) - min(bottoms)) / dz)) + 1
    elevations = (max(tops) - np.arange(nz, dtype=np.float32) * dz).astype(np.float32)
    stride = max(nz // VIEW_NZ, 1)
    kept = elevations[::stride]
    return PetroGrid(
        positions=np.array([model.position.x for model in models], dtype=np.float32),
        elevations=kept,
        soil_grid=[list(model.sample_soil(kept)) for model in models],
        n_grid=np.array([model.sample_N(kept) for model in models], dtype=np.float32),
        water_table_elevations=np.array(
            [model.position.z - model.water_table_depth for model in models], dtype=np.float32
        ),
    )


def save_petro_section(run_folder: Path, units: Sequence[str]) -> Path | None:
    """The figure of the line's soils and N values, in `run_folder`."""
    section = petro_models(run_folder, units)
    if section is None:
        return None
    figure = plot_petro_models_section(section)
    path = run_folder / SECTION_FIGURE
    Plot.savefig(path=path, figure=figure)
    plt.close(figure)
    return path


def save_petro_sections_file(run_folder: Path, units: Sequence[str]) -> Path | None:
    """The line's soils, N values and water table on the figure's grid, in one HDF5 file in
    `run_folder`, with the velocity section file's x and z."""
    section = petro_models(run_folder, units)
    if section is None:
        return None
    xs, zs, soil_grid, n_grid, water_table_elevations = section.to_grid(dz=0.01, dx=None)
    # str(), not .value: a cell below a window's column holds "" (see PetroGrid).
    soils = np.array(  # pyright: ignore[reportUnknownVariableType]
        [[str(soil) for soil in row] for row in soil_grid],
        dtype=h5py.string_dtype(),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    )
    path = run_folder / SECTION_FILE
    with h5py.File(path, "w") as file:
        file.create_dataset("x", data=xs)  # pyright: ignore[reportUnknownMemberType]
        file.create_dataset("z", data=zs)  # pyright: ignore[reportUnknownMemberType]
        file.create_dataset("soil", data=soils)  # pyright: ignore[reportUnknownMemberType]
        file.create_dataset("N", data=n_grid)  # pyright: ignore[reportUnknownMemberType]
        file.create_dataset("water_table_elevation", data=water_table_elevations)  # pyright: ignore[reportUnknownMemberType]
    return path


@dataclass(frozen=True, slots=True)
class RockPhysicsGrid:
    """A rock-physics quantity along the line, in the section's units: one column per inverted
    window, at most VIEW_NZ elevations."""

    positions: np.ndarray
    elevations: np.ndarray
    values: np.ndarray


def rock_physics_grid(
    run_folder: Path, units: Sequence[str], quantity: Quantity
) -> RockPhysicsGrid | None:
    """Each window's saved profile of `quantity`, interpolated within itself onto the line's
    elevations: shear modulus and Vs vary continuously with depth, within a soil layer too
    (saturation and effective pressure do)."""
    found = QUANTITIES[quantity]
    entries: list[tuple[float, float, np.ndarray, np.ndarray]] = []  # x, top, elevations, values
    for unit in units:
        model = load_petro_model(run_folder / unit)
        profile = None if model is None else load_profile(run_folder / unit, quantity)
        if model is None or profile is None:
            continue
        entries.append((model.position.x, model.position.z, *profile))
    if len(entries) < 2:
        return None
    entries.sort(key=lambda entry: entry[0])

    top = max(entry[1] for entry in entries)
    bottom = min(float(entry[2][-1]) for entry in entries)
    nz = int(np.floor((top - bottom) / PROFILE_DZ)) + 1
    elevations = (top - np.arange(nz, dtype=np.float32) * PROFILE_DZ).astype(np.float32)
    values = np.full((len(entries), nz), np.nan, dtype=np.float32)
    for i, (_x, _top, profile_elevations, profile_values) in enumerate(entries):
        # Saved deepest last; np.interp takes its points in ascending order.
        ascending = profile_elevations[::-1]
        inside = (elevations >= ascending[0]) & (elevations <= ascending[-1])
        values[i, inside] = np.interp(
            elevations[inside], ascending, profile_values[::-1] * found.scale
        )
    stride = max(nz // VIEW_NZ, 1)
    return RockPhysicsGrid(
        positions=np.array([entry[0] for entry in entries], dtype=np.float32),
        elevations=elevations[::stride],
        values=values[:, ::stride],
    )


def save_rock_physics_section(
    run_folder: Path, units: Sequence[str], quantity: Quantity
) -> Path | None:
    """The figure of `quantity` along the line, in `run_folder`."""
    grid = rock_physics_grid(run_folder, units, quantity)
    if grid is None:
        return None
    found = QUANTITIES[quantity]
    figure, ax = plt.subplots(figsize=(DOUBLE_COLUMN_CM * CM, HEIGHT_CM * CM), dpi=DISP_DPI)
    mesh = ax.pcolormesh(  # pyright: ignore[reportUnknownMemberType]
        grid.positions, grid.elevations, grid.values.T, shading="nearest", cmap=found.cmap
    )
    ax.set_xlim(grid.positions[0], grid.positions[-1])
    figure.colorbar(mesh, ax=ax, label=found.label)  # pyright: ignore[reportUnknownMemberType]
    ax.set_xlabel("Position [m]")  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel("Elevation [m]")  # pyright: ignore[reportUnknownMemberType]
    figure.tight_layout()
    path = run_folder / found.section_figure
    Plot.savefig(path=path, figure=figure)
    plt.close(figure)
    return path


def save_rock_physics_file(
    run_folder: Path, units: Sequence[str], quantity: Quantity
) -> Path | None:
    """`quantity` along the line, on the figure's grid, in an HDF5 file in `run_folder`."""
    grid = rock_physics_grid(run_folder, units, quantity)
    if grid is None:
        return None
    found = QUANTITIES[quantity]
    path = run_folder / found.section_file
    with h5py.File(path, "w") as file:
        file.create_dataset("x", data=grid.positions)  # pyright: ignore[reportUnknownMemberType]
        file.create_dataset("z", data=grid.elevations)  # pyright: ignore[reportUnknownMemberType]
        file.create_dataset(found.dataset, data=grid.values)  # pyright: ignore[reportUnknownMemberType]
    return path


def comparison_grids(run_folder: Path, units: Sequence[str]) -> ComparisonGrids | None:
    """The fundamental modes picked along the line against the curves the models give back,
    over the window folders holding both."""
    observed: list[DispersionCurve] = []
    predicted: list[DispersionCurve] = []
    for unit in units:
        try:
            picked = fundamental_curve(run_folder / unit)
        except ValueError:
            continue
        modeled = load_modeled_curve(run_folder / unit)
        if modeled is None:
            continue
        observed.append(picked)
        # The forward model knows no position: the pick's, so that both sort alike.
        predicted.append(replace(modeled, acquisition=picked.acquisition))
    if len(observed) < 2:
        return None
    positions, fs, observed_grid, predicted_grid, residual = pseudo_section_comparison_grids(
        DispersionCurvesSection(dispersion_curves=tuple(observed)),
        DispersionCurvesSection(dispersion_curves=tuple(predicted)),
    )
    return ComparisonGrids(positions, fs, observed_grid, predicted_grid, residual)
