import json
from collections.abc import Sequence
from pathlib import Path

from sigpipe.base.coordinate import UNKNOWN_COORDINATE, Coordinate
from sigpipe.base.petro_model import PetroModel, PetroModels, SoilType

_HEADER = "soil,thickness_m,N"


def load_petro_models(file_paths: Sequence[Path]) -> list[PetroModels]:
    petro_models_out: list[PetroModels] = []
    for path in file_paths:
        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("r", encoding="utf-8") as file:
            content = file.read()

        blocks = [block.strip() for block in content.split("---") if block.strip()]

        petro_models = []
        for block in blocks:
            petro_models.append(_parse_petro_model(block))

        petro_models_out.append(
            PetroModels(
                petro_models=tuple(
                    petro_models,
                )
            )
        )

    return petro_models_out


def _parse_petro_model(block: str) -> PetroModel:
    lines = [line.strip() for line in block.splitlines() if line.strip()]

    position = UNKNOWN_COORDINATE
    water_table_depth = 0.0
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith("position:"):
            position = Coordinate.from_tuple(json.loads(line.removeprefix("position:").strip()))
        elif line.startswith("water_table_depth_m:"):
            water_table_depth = float(line.removeprefix("water_table_depth_m:").strip())
        elif line == _HEADER:
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"Could not find data table in block:\n{block}")

    soils = []
    thicknesses = []
    ns = []
    for line in lines[data_start:]:
        parts = line.split(",")
        soils.append(SoilType(parts[0]))
        thicknesses.append(float(parts[1]))
        ns.append(int(parts[2]))

    return PetroModel(
        soils=tuple(soils),
        thicknesses=tuple(thicknesses),
        Ns=tuple(ns),
        water_table_depth=water_table_depth,
        position=position,
    )
