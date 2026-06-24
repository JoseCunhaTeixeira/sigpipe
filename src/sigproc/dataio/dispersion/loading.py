import ast
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from sigproc.base.acquisition import UNKNOWN_ACQUISITION, acquisition_from_kind
from sigproc.base.coordinate import (
    Coordinate,
    tuples_to_coordinates,
)
from sigproc.base.dispersion_curve import (
    DispersionCurve,
    DispersionCurves,
    DispersionCurvesImage,
    Mode,
    VelocityType,
)
from sigproc.base.dispersion_image import DispersionImage
from sigproc.dataio._h5 import dataset


def load_dispersion_image(
    file_paths: Sequence[Path],
    curves_paths: Sequence[Path] | None = None,
) -> list[DispersionImage]:

    dispersion_curves_out: list[DispersionCurves] = []
    if curves_paths is not None:
        dispersion_curves_out = load_dispersion_curves(file_paths=curves_paths)

    dispersion_images_out: list[DispersionImage] = []
    for i, path in enumerate(file_paths):
        if not path.exists():
            raise FileNotFoundError(path)

        with h5py.File(path, "r") as f:
            fv_map = np.asarray(
                dataset(f, "fv_map")[:],
                dtype=np.float32,
            )

            fs = np.asarray(
                dataset(f, "fs")[:],
                dtype=np.float32,
            )

            vs = np.asarray(
                dataset(f, "vs")[:],
                dtype=np.float32,
            )

            type = dataset(f, "type")[()].decode() if "type" in f else ""

            source = Coordinate.from_tuple(dataset(f, "source")[:])

            receivers = tuples_to_coordinates(dataset(f, "receivers")[:])

            kind = (
                dataset(f, "acquisition_kind")[()].decode() if "acquisition_kind" in f else ""
            )

        acquisition = acquisition_from_kind(
            kind,
            source=source,
            receivers=receivers,
        )

        dispersion_curves: DispersionCurvesImage | None = None
        if curves_paths is not None and dispersion_curves_out:
            dispersion_curves = DispersionCurvesImage(
                dispersion_curves=dispersion_curves_out[i].dispersion_curves,
            )

        dispersion_images_out.append(
            DispersionImage(
                fv_map=fv_map,
                fs=fs,
                vs=vs,
                type=VelocityType(type),
                acquisition=acquisition,
                dispersion_curves=dispersion_curves,
            )
        )

    return dispersion_images_out


def load_dispersion_curves(
    file_paths: Sequence[Path],
) -> list[DispersionCurves]:
    dispersion_curves_out: list[DispersionCurves] = []
    for path in file_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix == ".txt":
            dispersion_curves_out.append(_load_modeled_dispersion_curves(path=path))
        elif path.suffix == ".csv":
            dispersion_curves_out.append(_load_picked_dispersion_curves(path=path))
        else:
            raise TypeError(f"File must be .txt or .csv, got {path.suffix}")
    return dispersion_curves_out


def _load_picked_dispersion_curves(
    path: Path,
) -> DispersionCurves:

    if not path.exists():
        raise FileNotFoundError(path)

    curves = []

    with path.open("r", encoding="utf-8") as file:
        content = file.read()

    blocks = [block.strip() for block in content.split("---") if block.strip()]

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]

        type = ""
        mode = Mode("X", 999)
        kind = ""
        source = None
        receivers = None

        data_start = None
        has_std = False

        for i, line in enumerate(lines):
            if line.startswith("type:"):
                type = line.removeprefix("type:").strip()

            elif line.startswith("mode:"):
                mode = Mode(*ast.literal_eval(line.removeprefix("mode:").strip()))

            elif line.startswith("acquisition_kind:"):
                kind = line.removeprefix("acquisition_kind:").strip()

            elif line.startswith("source:"):
                raw_source = ast.literal_eval(
                    line.removeprefix("source:").strip(),
                )

                source = Coordinate.from_tuple(raw_source)

            elif line.startswith("receivers:"):
                raw_receivers = ast.literal_eval(
                    line.removeprefix("receivers:").strip(),
                )

                receivers = tuples_to_coordinates(raw_receivers)

            elif line in (
                "frequency_Hz,phase_velocity_m/s",
                "frequency_Hz,phase_velocity_m/s,velocity_std_m/s",
            ):
                has_std = line.endswith("velocity_std_m/s")
                data_start = i + 1
                break

        if source is None or receivers is None:
            acquisition = UNKNOWN_ACQUISITION
        else:
            acquisition = acquisition_from_kind(
                kind,
                source=source,
                receivers=receivers,
            )

        if data_start is None:
            raise ValueError(f"Could not find frequency table in '{path}'.")

        fs = []
        vs = []
        vs_err: list[float] | None = [] if has_std else None

        for line in lines[data_start:]:
            parts = line.split(",")
            fs.append(float(parts[0]))
            vs.append(float(parts[1]))
            if vs_err is not None:
                vs_err.append(float(parts[2]))

        curves.append(
            DispersionCurve(
                fs=np.asarray(fs, dtype=np.float32),
                vs=np.asarray(vs, dtype=np.float32),
                mode=mode,
                type=VelocityType(type),
                acquisition=acquisition,
                vs_err=np.asarray(vs_err, dtype=np.float32) if vs_err is not None else None,
            )
        )

    return DispersionCurves(
        dispersion_curves=tuple(curves),
    )


def _load_modeled_dispersion_curves(
    path: Path,
) -> DispersionCurves:

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        sep=";",
        skiprows=4,
    )

    df.columns = [c.strip() for c in df.columns]

    fcol = "Frequency"

    curves = []

    for col in df.columns:
        if col == fcol:
            continue

        valid = df[col].notna()

        fs = (
            df.loc[
                valid,
                fcol,
            ]
            .to_numpy(dtype=np.float32)
            .reshape(-1)
        )

        vs = (
            df.loc[
                valid,
                col,
            ]
            .to_numpy(dtype=np.float32)
            .reshape(-1)
        )

        if len(fs) == 0:
            continue

        curves.append(
            DispersionCurve(
                fs=fs * 1e6,
                vs=vs * 1e3,
                mode=Mode(*ast.literal_eval(col.strip())),
                acquisition=UNKNOWN_ACQUISITION,
                type=VelocityType.UNKNOWN,
            )
        )

    return DispersionCurves(
        dispersion_curves=tuple(curves),
    )
