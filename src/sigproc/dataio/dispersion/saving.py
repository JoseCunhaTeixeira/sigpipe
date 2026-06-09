from pathlib import Path

import h5py

from src.sigproc.base.coordinate import (
    coordinates_to_tuples,
)
from src.sigproc.base.dispersion import DispersionCurves, DispersionImage


def save_dispersion_image(
    dispersion_image: DispersionImage,
    path: Path,
    **kwargs,
) -> None:
    sources = []
    receivers = []
    path = path.with_suffix(".hd5")
    for acquisition in dispersion_image.acquisitions:
        sources.append(acquisition.source.to_tuple())
        receivers.append(coordinates_to_tuples(acquisition.receivers))
    with h5py.File(path, "w") as file:
        file.create_dataset("fv_map", data=dispersion_image.fv_map)
        file.create_dataset("fs", data=dispersion_image.fs)
        file.create_dataset("vs", data=dispersion_image.vs)
        file.create_dataset("type", data=dispersion_image.type)
        file.create_dataset("sources", data=tuple(sources))
        file.create_dataset("receivers", data=tuple(receivers))
        for key, value in kwargs.items():
            file.create_dataset(key, data=value)
    if dispersion_image.dispersion_curves is not None:
        save_dispersion_curves(dispersion_image.dispersion_curves, path=path)


def save_dispersion_curves(
    dispersion_curves: DispersionCurves,
    path: Path,
) -> None:
    if dispersion_curves:
        path = path.with_name(
            path.name.replace("DispersionImage", "DispersionCurves")
        ).with_suffix(".csv")
        with open(path, "w", encoding="utf-8") as file:
            for dispersion_curve in dispersion_curves:
                sources = []
                receivers = []
                for acquisition in dispersion_curve.acquisitions:
                    sources.append(acquisition.source.to_tuple())
                    receivers.append(coordinates_to_tuples(acquisition.receivers))
                file.write(f"label: {dispersion_curve.label}\n")
                file.write(f"type: {dispersion_curve.type}\n")
                file.write(f"sources: {tuple(sources)}\n")
                file.write(f"receivers: {tuple(receivers)}\n")
                file.write("frequency_Hz,phase_velocity_m/s\n")
                for f, v in zip(dispersion_curve.fs, dispersion_curve.vs, strict=True):
                    file.write(f"{float(f):.6f},{float(v):.6f}\n")
                file.write("\n---\n\n")
