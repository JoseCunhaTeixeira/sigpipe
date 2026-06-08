from pathlib import Path

import h5py
import numpy as np

from base.beamforming import Beam
from src.base.acquisition import Acquisition
from src.base.coordinate import Coordinate, tuples_to_coordinates


def load_beam(
    path: Path,
) -> Beam:
    path = path.with_suffix(".hd5")

    with h5py.File(path, "r") as file:
        xy_map = np.asarray(
            file["xy_map"][:],  # type: ignore
            dtype=np.float32,
        )

        xs = np.asarray(
            file["xs"][:],  # type: ignore
            dtype=np.float32,
        )

        ys = np.asarray(
            file["ys"][:],  # type: ignore
            dtype=np.float32,
        )

        source = tuple(file["source"][:])  # type: ignore
        receivers = list(file["receivers"][:])  # type: ignore

    acquisition = Acquisition(
        source=Coordinate(*source),
        receivers=tuples_to_coordinates(receivers),
    )

    return Beam(
        xy_map=xy_map,
        xs=xs,
        ys=ys,
        acquisition=acquisition,
    )
