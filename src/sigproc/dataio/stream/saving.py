from pathlib import Path

import h5py

from sigproc.base.acquisition import acquisition_kind
from sigproc.base.coordinate import coordinates_to_tuples
from sigproc.base.stream import Stream


def save_stream(
    stream: Stream,
    path: Path,
    **kwargs: object,
) -> None:
    path = path.with_suffix(".hdf5")
    source = stream.acquisition.source.to_tuple()
    receivers = coordinates_to_tuples(stream.acquisition.receivers)
    with h5py.File(path, "w") as file:
        file.create_dataset("xt", data=stream.xt)
        file.create_dataset("ts", data=stream.ts)
        file.create_dataset("sampling_freq", data=stream.sampling_freq)
        file.create_dataset("source", data=source)
        file.create_dataset("receivers", data=receivers)
        file.create_dataset("acquisition_kind", data=acquisition_kind(stream.acquisition))
        for key, value in kwargs.items():
            file.create_dataset(key, data=value)
