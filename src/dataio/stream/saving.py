from pathlib import Path

import h5py

from src.base.coordinate import coordinates_to_tuples
from src.base.stream import Stream


def save_stream(
    stream: Stream,
    path: Path,
    **kwargs,
) -> None:
    path = path.with_suffix(".hd5")
    source = stream.acquisition.source.to_tuple()
    receivers = coordinates_to_tuples(stream.acquisition.receivers)
    with h5py.File(path, "w") as file:
        file.create_dataset("xt", data=stream.xt)
        file.create_dataset("ts", data=stream.ts)
        file.create_dataset("sampling_freq", data=stream.sampling_freq)
        file.create_dataset("source", data=source)
        file.create_dataset("receivers", data=receivers)
        for key, value in kwargs.items():
            file.create_dataset(key, data=value)
