import ast
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.base.acquisition import UNKNOWN_ACQUISITION, Acquisition
from src.base.coordinate import (
    Coordinate,
    tuples_to_coordinates,
)
from src.base.dispersion import DispersionCurve, DispersionImage


def load_dispersion_image(
    path: Path,
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

    return DispersionImage(
        fv_map=fv_map,
        fs=fs,
        vs=vs,
        type=type,
        acquisitions=acquisitions,
    )


def load_dispersion_curve(
    path: Path,
) -> DispersionCurve | list[DispersionCurve]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == "txt":
        return load_modeled_dispersion_curves(path=path)
    elif path.suffix == "csv":
        return load_picked_dispersion_curve(path=path)
    else:
        raise TypeError(f"File must be .txt or .csv, got {path.suffix}")


def load_picked_dispersion_curve(
    path: Path,
) -> DispersionCurve:

    if not path.exists():
        raise FileNotFoundError(path)

    name = "unknown"

    sources = ()
    receivers = ()

    with path.open("r") as f:
        for line in f:
            line = line.strip()

            if not line.startswith("#"):
                break

            line = line.removeprefix("#").strip()

            if line.startswith("name:"):
                name = line.removeprefix("name:").strip()

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

    data = np.loadtxt(
        path,
        delimiter=",",
        comments="#",
    )

    fs = data[:, 0]
    vs = data[:, 1]

    return DispersionCurve(
        fs=fs,
        vs=vs,
        name=name,
        acquisitions=acquisitions,
    )


def load_modeled_dispersion_curves(
    path: Path,
) -> list[DispersionCurve]:

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        sep=";",
        skiprows=4,
    )

    df.columns = [c.strip() for c in df.columns]

    fcol = "Frequency"

    curves: list[DispersionCurve] = []

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
                name=col.strip(),
                acquisitions=(UNKNOWN_ACQUISITION,),
            )
        )

    return curves
