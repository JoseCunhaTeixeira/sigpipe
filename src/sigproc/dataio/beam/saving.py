from pathlib import Path

import h5py

from src.sigproc.base.beamforming import Beam
from src.sigproc.base.coordinate import (
    coordinates_to_tuples,
)


def save_beam(
    beam: Beam,
    path: Path,
    **kwargs,
) -> None:
    path = path.with_suffix(".hd5")
    source = beam.acquisition.source.to_tuple()
    receivers = coordinates_to_tuples(beam.acquisition.receivers)
    with h5py.File(path, "w") as file:
        file.create_dataset("xy_map", data=beam.xy_map)
        file.create_dataset("xs", data=beam.xs)
        file.create_dataset("ys", data=beam.ys)
        file.create_dataset("source", data=tuple(source))
        file.create_dataset("receivers", data=receivers)
        for key, value in kwargs.items():
            file.create_dataset(key, data=value)
