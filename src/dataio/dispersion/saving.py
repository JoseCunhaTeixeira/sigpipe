from pathlib import Path

import h5py
import numpy as np

from src.base.coordinate import (
    coordinates_to_tuples,
)
from src.base.dispersion import DispersionCurve, DispersionImage


def save_dispersion_image(
    dispersion_image: DispersionImage,
    path: Path,
    **kwargs,
) -> None:
    sources = []
    receivers = []
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


def save_dispersion_curve(
    dispersion_curve: DispersionCurve,
    path: Path,
) -> None:
    sources = []
    receivers = []
    for acquisition in dispersion_curve.acquisitions:
        sources.append(acquisition.source.to_tuple())
        receivers.append(coordinates_to_tuples(acquisition.receivers))
    header = (
        f"name: {dispersion_curve.name}\n"
        f"type: {dispersion_curve.type}\n"
        f"sources: {tuple(sources)}\n"
        f"receivers: {tuple(receivers)}\n"
        "frequency_Hz,phase_velocity_m/s"
    )
    arr = np.column_stack([dispersion_curve.fs, dispersion_curve.vs])
    np.savetxt(
        path,
        arr,
        header=header,
        delimiter=",",
        fmt="%.6f",
    )
