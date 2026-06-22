import ast
from collections.abc import Sequence
from pathlib import Path

from sigproc.base.coordinate import UNKNOWN_COORDINATE, Coordinate
from sigproc.base.velocity_model import VelocityModel, VelocityModels

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
            position = Coordinate.from_tuple(
                ast.literal_eval(line.removeprefix("position:").strip())
            )
        elif line == _HEADER:
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"Could not find data table in block:\n{block}")

    thicknesses = []
    vs_p = []
    vs_s = []
    rhos = []
    vs_s_std = []
    for line in lines[data_start:]:
        parts = line.split(",")
        thicknesses.append(float(parts[0]))
        vs_p.append(float(parts[1]))
        vs_s.append(float(parts[2]))
        rhos.append(float(parts[3]))
        vs_s_std.append(float(parts[4]))

    return VelocityModel(
        vs_s=tuple(vs_s),
        vs_p=tuple(vs_p),
        rhos=tuple(rhos),
        vs_s_std=tuple(vs_s_std),
        thicknesses=tuple(thicknesses),
        position=position,
    )
