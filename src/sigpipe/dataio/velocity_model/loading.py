import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sigpipe.base.coordinate import UNKNOWN_COORDINATE, Coordinate
from sigpipe.base.velocity_model import VelocityModel, VelocityModels

_HEADER = "thickness_m,vp_m/s,vs_m/s,rho_kg/m3,vs_std_m/s"


def load_velocity_models(file_paths: Sequence[Path]) -> list[VelocityModels]:
    velocity_models_out: list[VelocityModels] = []
    for path in file_paths:
        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("r", encoding="utf-8") as file:
            content = file.read()

        blocks = [block.strip() for block in content.split("---") if block.strip()]

        velocity_models = []
        for block in blocks:
            velocity_models.append(_parse_velocity_model(block))

        velocity_models_out.append(
            VelocityModels(
                velocity_models=tuple(
                    velocity_models,
                )
            )
        )

    return velocity_models_out


def _parse_velocity_model(block: str) -> VelocityModel:
    lines = [line.strip() for line in block.splitlines() if line.strip()]

    position = UNKNOWN_COORDINATE
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith("position:"):
            position = Coordinate.from_tuple(json.loads(line.removeprefix("position:").strip()))
        elif line == _HEADER:
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"Could not find data table in block:\n{block}")

    # Parsed in C: a smooth model holds a row every centimetre, thousands of them.
    data = lines[data_start:]
    rows = np.loadtxt(data, delimiter=",", ndmin=2) if data else np.empty((0, 5))
    thicknesses, vs_p, vs_s, rhos, vs_s_std = (tuple(column.tolist()) for column in rows.T)

    return VelocityModel(
        vs_s=vs_s,
        vs_p=vs_p,
        rhos=rhos,
        vs_s_std=vs_s_std,
        thicknesses=thicknesses,
        position=position,
    )
