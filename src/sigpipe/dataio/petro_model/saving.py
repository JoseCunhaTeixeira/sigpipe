import json
from pathlib import Path
from typing import TextIO

from sigpipe.base.petro_model import PetroModel, PetroModels


def save_petro_model(petro_model: PetroModel, path: Path) -> None:
    path = path.with_suffix(".csv")
    with path.open("w", encoding="utf-8") as file:
        _write_petro_model(file, petro_model)


def save_petro_models(petro_models: PetroModels, path: Path) -> None:
    path = path.with_suffix(".csv")
    with path.open("w", encoding="utf-8") as file:
        for petro_model in petro_models.petro_models:
            _write_petro_model(file, petro_model)
            file.write("\n---\n\n")


def _write_petro_model(
    file: TextIO,
    petro_model: PetroModel,
) -> None:
    file.write(f"position: {json.dumps(petro_model.position.to_tuple())}\n")
    file.write(f"water_table_depth_m: {float(petro_model.water_table_depth):.6f}\n")
    file.write("soil,thickness_m,N\n")
    for soil, thickness, n in zip(
        petro_model.soils,
        petro_model.thicknesses,
        petro_model.Ns,
        strict=True,
    ):
        file.write(f"{soil.value},{float(thickness):.6f},{int(n)}\n")
