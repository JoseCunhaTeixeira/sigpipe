from pathlib import Path
from typing import TextIO

from sigpipe.base.velocity_model import VelocityModel, VelocityModels


def save_velocity_model(velocity_model: VelocityModel, path: Path) -> None:
    path = path.with_suffix(".csv")
    with path.open("w", encoding="utf-8") as file:
        _write_velocity_model(file, velocity_model)


def save_velocity_models(velocity_models: VelocityModels, path: Path) -> None:
    path = path.with_suffix(".csv")
    with path.open("w", encoding="utf-8") as file:
        for velocity_model in velocity_models.velocity_models:
            _write_velocity_model(file, velocity_model)
            file.write("\n---\n\n")


def _write_velocity_model(
    file: TextIO,
    velocity_model: VelocityModel,
) -> None:
    file.write(f"position: {velocity_model.position.to_tuple()}\n")
    file.write("thickness_m,vp_m/s,vs_m/s,rho_kg/m3,vs_std_m/s\n")
    for thickness, vp, vs, rho, vs_std in zip(
        velocity_model.thicknesses,
        velocity_model.vs_p,
        velocity_model.vs_s,
        velocity_model.rhos,
        velocity_model.vs_s_std,
        strict=True,
    ):
        file.write(
            f"{float(thickness):.6f},{float(vp):.6f},{float(vs):.6f},{float(rho):.6f},{float(vs_std):.6f}\n"
        )
