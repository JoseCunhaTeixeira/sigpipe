import ast
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.sigproc.base.acquisition import UNKNOWN_ACQUISITION, Acquisition
from src.sigproc.base.coordinate import (
    Coordinate,
    tuples_to_coordinates,
)
from src.sigproc.base.dispersion import (
    DispersionCurve,
    DispersionCurves,
    DispersionImage,
)


def load_dispersion_image(
    path: Path,
    curves_path: Path | None = None,
) -> DispersionImage:

    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        fv_map = np.asarray(
            f["fv_map"][:],  # type: ignore
            dtype=np.float32,
        )

        fs = np.asarray(
            f["fs"][:],  # type: ignore
            dtype=np.float32,
        )

        vs = np.asarray(
            f["vs"][:],  # type: ignore
            dtype=np.float32,
        )

        type = f["type"][()].decode() if "type" in f else ""  # type: ignore

        sources = tuple(Coordinate.from_tuple(source) for source in f["sources"][:])  # type: ignore

        receivers = tuple(
            tuples_to_coordinates(receiver_group)
            for receiver_group in f["receivers"][:]  # type: ignore
        )

    acquisitions = tuple(
        Acquisition(
            source=source,
            receivers=receiver_group,
        )
        for source, receiver_group in zip(
            sources,
            receivers,
            strict=True,
        )
    )

    dispersion_curves = DispersionCurves(curves=())
    if curves_path is not None:
        dispersion_curves = load_dispersion_curves(path=curves_path)

    return DispersionImage(
        fv_map=fv_map,
        fs=fs,
        vs=vs,
        type=type,
        acquisitions=acquisitions,
        dispersion_curves=dispersion_curves,
    )


def load_dispersion_curves(
    path: Path,
) -> DispersionCurves:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".txt":
        return load_modeled_dispersion_curves(path=path)
    elif path.suffix == ".csv":
        return load_picked_dispersion_curves(path=path)
    else:
        raise TypeError(f"File must be .txt or .csv, got {path.suffix}")


def load_picked_dispersion_curves(
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

        label = ""
        curve_type = ""

        sources: tuple[Coordinate, ...] = ()
        receivers: tuple[tuple[Coordinate, ...], ...] = ()

        data_start = None

        for i, line in enumerate(lines):
            if line.startswith("label:"):
                label = line.removeprefix("label:").strip()

            elif line.startswith("type:"):
                curve_type = line.removeprefix("type:").strip()

            elif line.startswith("sources:"):
                raw_sources = ast.literal_eval(
                    line.removeprefix("sources:").strip(),
                )

                sources = tuple(Coordinate.from_tuple(source) for source in raw_sources)

            elif line.startswith("receivers:"):
                raw_receivers = ast.literal_eval(
                    line.removeprefix("receivers:").strip(),
                )

                receivers = tuple(
                    tuples_to_coordinates(receiver_group)
                    for receiver_group in raw_receivers
                )

            elif line == "frequency_Hz,phase_velocity_m/s":
                data_start = i + 1
                break

        if data_start is None:
            raise ValueError(f"Could not find frequency table in '{path}'.")

        fs = []
        vs = []

        for line in lines[data_start:]:
            f, v = line.split(",")
            fs.append(float(f))
            vs.append(float(v))

        acquisitions = tuple(
            Acquisition(
                source=source,
                receivers=receiver_group,
            )
            for source, receiver_group in zip(
                sources,
                receivers,
                strict=True,
            )
        )

        curves.append(
            DispersionCurve(
                fs=np.asarray(fs, dtype=np.float32),
                vs=np.asarray(vs, dtype=np.float32),
                label=label,
                type=curve_type,
                acquisitions=acquisitions,
            )
        )

    return DispersionCurves(
        curves=tuple(curves),
    )


def load_modeled_dispersion_curves(
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
                label=col.strip(),
                acquisitions=(UNKNOWN_ACQUISITION,),
                type="",
            )
        )

    return DispersionCurves(
        curves=tuple(curves),
    )
