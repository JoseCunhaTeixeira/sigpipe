"""One window's petrophysical inversion, on a window folder of a run. A Silex model predicts
the soils, their N values and the water table from the window's fundamental mode; the curve
that prediction gives back, and its rock physics with depth (the Hertz-Mindlin shear modulus
and Vs), are saved beside it, so that the line's views only read files."""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from sigpipe.algorithms.inversion.rayleigh.petro.forward import (
    fwd_petro_phase,
    parse_under_layers,
    rock_physics,
)
from sigpipe.algorithms.inversion.rayleigh.petro.silex_catalog import (
    bundled_silex_model_dir,
    load_silex_card,
)
from sigpipe.base.coordinate import Coordinate
from sigpipe.base.dispersion_curve import DispersionCurve, DispersionCurves, Mode
from sigpipe.base.petro_model import PetroModel
from sigpipe.dataio.dispersion.loading import load_dispersion_curves
from sigpipe.dataio.dispersion.saving import save_dispersion_curves
from sigpipe.dataio.petro_model.loading import load_petro_models
from sigpipe.masw.picks import load_curves
from sigpipe.transformers import Invert, Save

MODEL_FILE = "PetroInversion_Model_0000.csv"
MODELED_CURVE_FILE = "PetroInversion_DispersionCurves_0000.csv"
# Silex takes the fundamental mode as Mode("R", 0): a pick of mode number 0 is relabelled so,
# whatever its letter (PAC's pickers label it M0).
FUNDAMENTAL = Mode("R", 0)

type Quantity = Literal["shear_modulus", "vs"]


@dataclass(frozen=True, slots=True)
class RockPhysicsQuantity:
    """A rock-physics profile a window saves, and how the line's section shows it."""

    file: str  # the window's profile: a `position:` line, then `thickness_m,<column>` rows
    column: str  # in SI units
    scale: float  # from SI units to the section's
    label: str  # the section's colour bar
    cmap: str
    dataset: str  # its name in the section's HDF5 file
    section_figure: str
    section_file: str


QUANTITIES: dict[Quantity, RockPhysicsQuantity] = {
    "shear_modulus": RockPhysicsQuantity(
        file="PetroInversion_ShearModulus_0000.csv",
        column="mu_hm_pa",
        scale=1e-9,  # Pa to GPa
        label="Hertz-Mindlin shear modulus $\\mu_{HM}$ [GPa]",
        cmap="viridis",
        dataset="mu",
        section_figure="PetroInversion_ShearModulusSection_0000.png",
        section_file="PetroInversion_ShearModulusSection_0000.hdf5",
    ),
    "vs": RockPhysicsQuantity(
        file="PetroInversion_Vs_0000.csv",
        column="vs_m_s",
        scale=1.0,
        label="$V_s$ [m/s]",
        cmap="terrain",
        dataset="vs",
        section_figure="PetroInversion_VsSection_0000.png",
        section_file="PetroInversion_VsSection_0000.hdf5",
    ),
}


def fundamental_curve(folder: Path) -> DispersionCurve:
    """The fundamental mode picked in window folder `folder`, labelled as Silex takes it."""
    curves = load_curves(folder)
    curve = None if curves is None else next((one for one in curves if one.mode.number == 0), None)
    if curve is None:
        raise ValueError(f"No fundamental-mode (mode number 0) pick in {folder.name}")
    return curve if curve.mode == FUNDAMENTAL else replace(curve, mode=FUNDAMENTAL)


def invert_window_petro(
    folder: Path, model_name: str, output_folder: Path | None = None
) -> PetroModel:
    """Invert the fundamental mode picked in window folder `folder` with the bundled Silex model
    `model_name`, and write PAC's files beside it, or in `output_folder` (a staging folder: see
    sigpipe.masw.runs.stopping): the model, the curve it gives back, and its rock-physics
    profiles. Raises ValueError when the curve falls outside the range the model was trained
    on."""
    out = output_folder or folder
    observed = fundamental_curve(folder)
    model_dir = bundled_silex_model_dir(model_name)
    pipeline = Invert(method="silex", model_dir=model_dir) >> Save(
        folder_path=out, file_name="PetroInversion_Model"
    )
    result: PetroModel = pipeline.run(
        data=[DispersionCurves(dispersion_curves=(observed,))], show_log=False
    )[0]

    under_layers = parse_under_layers(load_silex_card(model_dir).under_layers)
    modeled = fwd_petro_phase(result, mode=0, fs=observed.fs, under_layers=under_layers)
    save_dispersion_curves(
        DispersionCurves(dispersion_curves=(modeled,)), path=out / MODELED_CURVE_FILE
    )

    # The rock physics of the forward model above, computed once: the sections read it back.
    profile = rock_physics(result)
    _save_profile(out, "shear_modulus", result.position, profile.dz, profile.muHMs)
    _save_profile(out, "vs", result.position, profile.dz, profile.VSs)
    return result


def load_petro_model(folder: Path) -> PetroModel | None:
    """The petrophysical model of window folder `folder`; None when it has none."""
    path = folder / MODEL_FILE
    return load_petro_models([path])[0][0] if path.exists() else None


def load_modeled_curve(folder: Path) -> DispersionCurve | None:
    """The curve the model of window folder `folder` gives back; None when it has none."""
    path = folder / MODELED_CURVE_FILE
    if not path.exists():
        return None
    (curves,) = load_dispersion_curves([path])
    return curves[0]


def _save_profile(
    folder: Path, quantity: Quantity, position: Coordinate, dz: float, values: np.ndarray
) -> None:
    """The shape of sigpipe's VelocityModel CSVs: every row's thickness is the rock physics'
    sampling step, since the values vary continuously with depth."""
    found = QUANTITIES[quantity]
    with (folder / found.file).open("w", encoding="utf-8") as file:
        file.write(f"position: {json.dumps(position.to_tuple())}\n")
        file.write(f"thickness_m,{found.column}\n")
        for value in values:
            file.write(f"{float(dz):.6f},{float(value):.6f}\n")


def load_profile(folder: Path, quantity: Quantity) -> tuple[np.ndarray, np.ndarray] | None:
    """(elevations, values) of window folder `folder`'s profile of `quantity`, in SI units,
    deepest last; None when it has none."""
    path = folder / QUANTITIES[quantity].file
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        position_line = file.readline()
    top = json.loads(position_line.removeprefix("position:").strip())[2]
    data = np.loadtxt(path, delimiter=",", skiprows=2, dtype=np.float32, ndmin=2)
    thicknesses, values = data[:, 0], data[:, 1]
    return top - np.cumsum(thicknesses), values
