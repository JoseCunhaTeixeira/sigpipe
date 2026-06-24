from pathlib import Path
from typing import TextIO

import h5py

from sigproc.base.acquisition import acquisition_kind
from sigproc.base.coordinate import (
    coordinates_to_tuples,
)
from sigproc.base.dispersion_curve import DispersionCurve, DispersionCurves
from sigproc.base.dispersion_image import DispersionImage


def save_dispersion_image(
    dispersion_image: DispersionImage,
    path: Path,
    **kwargs: object,
) -> None:
    path = path.with_suffix(".hdf5")
    with h5py.File(path, "w") as file:
        file.create_dataset("fv_map", data=dispersion_image.fv_map)
        file.create_dataset("fs", data=dispersion_image.fs)
        file.create_dataset("vs", data=dispersion_image.vs)
        file.create_dataset("type", data=str(dispersion_image.type))
        file.create_dataset("source", data=dispersion_image.acquisition.source.to_tuple())
        file.create_dataset(
            "receivers", data=coordinates_to_tuples(dispersion_image.acquisition.receivers)
        )
        file.create_dataset("acquisition_kind", data=acquisition_kind(dispersion_image.acquisition))
        for key, value in kwargs.items():
            file.create_dataset(key, data=value)
    if dispersion_image.dispersion_curves is not None:
        curves_path = path.with_stem(f"{path.stem}_curves")
        save_dispersion_curves(dispersion_image.dispersion_curves, path=curves_path)


def save_dispersion_curve(dispersion_curve: DispersionCurve, path: Path) -> None:
    path = path.with_suffix(".csv")
    with path.open("w", encoding="utf-8") as file:
        _write_dispersion_curve(file, dispersion_curve)


def save_dispersion_curves(
    dispersion_curves: DispersionCurves,
    path: Path,
) -> None:
    path = path.with_suffix(".csv")
    with path.open("w", encoding="utf-8") as file:
        for dispersion_curve in dispersion_curves:
            _write_dispersion_curve(file, dispersion_curve)
            file.write("\n---\n\n")


def _write_dispersion_curve(
    file: TextIO,
    dispersion_curve: DispersionCurve,
) -> None:
    file.write(f"type: {dispersion_curve.type}\n")
    file.write(f"mode: {tuple(dispersion_curve.mode)}\n")
    file.write(f"acquisition_kind: {acquisition_kind(dispersion_curve.acquisition)}\n")
    file.write(f"source: {dispersion_curve.acquisition.source.to_tuple()}\n")
    file.write(f"receivers: {coordinates_to_tuples(dispersion_curve.acquisition.receivers)}\n")
    if dispersion_curve.vs_err is not None:
        file.write("frequency_Hz,phase_velocity_m/s,velocity_std_m/s\n")
        for f, v, v_err in zip(
            dispersion_curve.fs,
            dispersion_curve.vs,
            dispersion_curve.vs_err,
            strict=True,
        ):
            file.write(f"{float(f):.6f},{float(v):.6f},{float(v_err):.6f}\n")
    else:
        file.write("frequency_Hz,phase_velocity_m/s\n")
        for f, v in zip(
            dispersion_curve.fs,
            dispersion_curve.vs,
            strict=True,
        ):
            file.write(f"{float(f):.6f},{float(v):.6f}\n")
