from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

from sigproc.base.acquisition import PlanarAcquisition, acquisition_from_kind
from sigproc.base.beamforming import Beam
from sigproc.base.coordinate import Coordinate, tuples_to_coordinates
from sigproc.dataio._h5 import dataset


def load_beam(
    path: Path,
) -> Beam:
    path = path.with_suffix(".hd5")

    with h5py.File(path, "r") as file:
        xy_map = np.asarray(
            dataset(file, "xy_map")[:],
            dtype=np.float32,
        )

        xs = np.asarray(
            dataset(file, "xs")[:],
            dtype=np.float32,
        )

        ys = np.asarray(
            dataset(file, "ys")[:],
            dtype=np.float32,
        )

        source = tuple(dataset(file, "source")[:])
        receivers = list(dataset(file, "receivers")[:])

        kind = (
            dataset(file, "acquisition_kind")[()].decode()
            if "acquisition_kind" in file
            else ""
        )

    acquisition = acquisition_from_kind(
        kind,
        source=Coordinate(*source),
        receivers=tuples_to_coordinates(receivers),
    )

    if not isinstance(acquisition, PlanarAcquisition):
        raise TypeError(
            f"Beam file '{path}' has no/unrecognized acquisition_kind tag "
            f"(got {kind!r}); expected PlanarAcquisition"
        )

    return Beam(
        xy_map=xy_map,
        xs=xs,
        ys=ys,
        acquisition=acquisition,
    )


def load_beams(file_paths: Sequence[Path]) -> list[Beam]:
    return [load_beam(path) for path in file_paths]
