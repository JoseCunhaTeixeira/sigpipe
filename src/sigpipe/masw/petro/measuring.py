"""What a window's petrophysical inversion gives, measured for PACo's gates (G7 on the window, G8
along the line): the fit of the curve its soil column gives back to the pick, by band as the
seismic models' (G5), and the model with depth. Measures only: PACo judges them."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from sigpipe.masw.inversion.measuring import ModelFit, fit_by_band
from sigpipe.masw.petro.window import (
    fundamental_curve,
    load_modeled_curve,
    load_petro_model,
    load_profile,
)


class PetroMeasures(BaseModel):
    """Everything G7 judges, and G8 compares along the line, of one window's petrophysical
    inversion."""

    model_config = ConfigDict(frozen=True)

    silex_model: str
    fit: ModelFit  # the curve the soil column gives back, against the pick
    soils: tuple[str, ...]  # top down
    thicknesses_m: tuple[float, ...]
    ns: tuple[int, ...]  # SPT N values
    water_table_m: float
    vs_at_depths: tuple[tuple[float, float], ...]  # (depth m, Vs m/s of the rock physics)


def measure_petro(
    folder: Path, silex_model: str, depths: Sequence[float], n_bands: int = 3
) -> PetroMeasures:
    """The petrophysical inversion of window folder `folder`, made with Silex model
    `silex_model`, measured: its fit in `n_bands` bands of the pick's points, and the Vs of its
    rock physics at `depths` (those within the soil column)."""
    model = load_petro_model(folder)
    if model is None:
        raise ValueError(f"{folder.name} holds no petrophysical model")
    fit = fit_by_band("petro", fundamental_curve(folder), load_modeled_curve(folder), n_bands)
    vs_at_depths: tuple[tuple[float, float], ...] = ()
    if (profile := load_profile(folder, "vs")) is not None:
        elevations, values = profile
        found = np.asarray(model.position.z - elevations, dtype=float)  # increasing
        vs_at_depths = tuple(
            (float(depth), round(float(np.interp(depth, found, values)), 1))
            for depth in depths
            if found[0] <= depth <= found[-1]
        )
    return PetroMeasures(
        silex_model=silex_model,
        fit=fit,
        soils=tuple(str(soil) for soil in model.soils),
        thicknesses_m=tuple(float(thickness) for thickness in model.thicknesses),
        ns=tuple(int(n) for n in model.Ns),
        water_table_m=float(model.water_table_depth),
        vs_at_depths=vs_at_depths,
    )
