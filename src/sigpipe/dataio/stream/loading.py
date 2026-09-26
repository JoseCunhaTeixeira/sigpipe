import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TypeGuard

import h5py
import numpy as np
from obspy import read as read_obspy

from sigpipe.base.acquisition import UNKNOWN_ACQUISITION, Acquisition, acquisition_from_kind
from sigpipe.base.coordinate import Coordinate, tuples_to_coordinates
from sigpipe.base.stream import Stream
from sigpipe.dataio._h5 import dataset


def load_stream(
    file_paths: Sequence[Path],
    sort: bool = False,
    acquisitions: Sequence[Acquisition] | None = None,  # one per file, for the loaded receivers
    receivers_to_load: Sequence[int] | Sequence[Sequence[int]] | None = None,
) -> list[Stream]:
    """
    Streams saved by save_stream. `receivers_to_load` keeps the same receivers
    of every file, or each file its own (one sequence per file); the saved
    acquisition, cut to those receivers, stands unless `acquisitions` is given.
    """
    if acquisitions is not None and len(acquisitions) != len(file_paths):
        raise ValueError(
            f"requires len(file_paths) == len(acquisitions), got {len(file_paths)} and {len(acquisitions)}"
        )
    receivers_per_file = _receivers_per_file(receivers_to_load, len(file_paths))

    streams_out: list[Stream] = []
    for i_file, path in enumerate(file_paths):
        path = path.with_suffix(".hdf5")

        with h5py.File(path, "r") as file:
            xt = np.asarray(
                dataset(file, "xt")[:],
                dtype=np.float32,
            )

            ts = np.asarray(
                dataset(file, "ts")[:],
                dtype=np.float32,
            )

            sampling_freq = float(dataset(file, "sampling_freq")[()])

            source = tuple(dataset(file, "source")[:])
            receivers = list(dataset(file, "receivers")[:])

            kind = (
                dataset(file, "acquisition_kind")[()].decode() if "acquisition_kind" in file else ""
            )

        acquisition = acquisition_from_kind(
            kind,
            source=Coordinate(*source),
            receivers=tuples_to_coordinates(receivers),
        )

        indices = receivers_per_file[i_file]
        if indices is not None:
            xt = xt[indices, :]
            acquisition = type(acquisition)(
                source=acquisition.source,
                receivers=tuple(acquisition.receivers[i] for i in indices),
            )
        if acquisitions is not None:
            acquisition = acquisitions[i_file]
            if xt.shape[0] != len(acquisition.receivers):
                raise ValueError(
                    "requires shot.shape[0] = number of receivers. "
                    f"Got {xt.shape[0]} and {len(acquisition.receivers)}"
                )

        if not acquisition.is_unknown and sort:
            order = np.argsort(acquisition.offsets)
            xt = xt[order]
            acquisition = type(acquisition)(
                source=acquisition.source,
                receivers=tuple(acquisition.receivers[i] for i in order),
            )

        streams_out.append(
            Stream(
                xt=xt,
                ts=ts,
                sampling_freq=sampling_freq,
                acquisition=acquisition,
            )
        )

    return streams_out


def load_seismic(
    file_paths: Sequence[Path],
    acquisitions: Sequence[Acquisition],  # one acquisition per file
    sort: bool = False,
    receivers_to_load: Sequence[int] | Sequence[Sequence[int]] | None = None,
) -> list[Stream]:

    if not isinstance(acquisitions, Sequence) or isinstance(acquisitions, (str, bytes)):
        raise TypeError(f"Expected Sequence for acquisitions, got {type(acquisitions).__name__}")

    if len(file_paths) != len(acquisitions):
        raise ValueError(
            f"requires len(file_paths) == len(acquisitions), got {len(file_paths)} and {len(acquisitions)}"
        )

    if not all(isinstance(s, Acquisition) for s in acquisitions):
        raise TypeError("All elements in acquisitions must be Acquisition")

    receivers_per_file = _receivers_per_file(receivers_to_load, len(file_paths))

    streams_out: list[Stream] = []
    for path, acquisition, indices in zip(
        file_paths, acquisitions, receivers_per_file, strict=False
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                module=r"obspy\.io\.seg2\.seg2",
            )
            ob_stream = read_obspy(path)
        sampling_freq = ob_stream[0].stats.sampling_rate
        nx = len(ob_stream)
        nt = ob_stream[0].stats.npts
        xt = np.zeros((nx, nt), dtype=np.float32)
        for i, trace in enumerate(ob_stream):
            xt[i, :] = trace.data

        if indices is not None:
            xt = xt[indices, :]

        ts = compute_time_vector(
            nt=xt.shape[1],
            sampling_freq=sampling_freq,
        )

        if xt.shape[0] != len(acquisition.receivers):
            raise ValueError(
                "requires shot.shape[0] = number of receivers. "
                f"Got {xt.shape[0]} and {len(acquisition.receivers)}"
            )

        if not acquisition.is_unknown and sort:
            order = np.argsort(acquisition.offsets)
            xt = xt[order]
            acquisition = type(acquisition)(
                source=acquisition.source,
                receivers=tuple(acquisition.receivers[i] for i in order),
            )

        streams_out.append(
            Stream(
                xt=xt,
                ts=ts,
                sampling_freq=sampling_freq,
                acquisition=acquisition,
            )
        )

    return streams_out


def load_gero_passive(
    file_paths: Sequence[Path],
    *,
    key: str = "signal",
    sampling_freq: float | None = None,
    acquisition: Acquisition = UNKNOWN_ACQUISITION,  # one unique acquisition for all files
    sort: bool = False,
    receivers_to_load: Sequence[int] | None = None,
) -> list[Stream]:
    user_sampling_freq = sampling_freq
    streams_out: list[Stream] = []
    for path in file_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as f:
            if key not in f:
                raise ValueError(f"Missing dataset '{key}'. Available objects: {list(f)}")
            record = np.array(
                dataset(f, key)[:],
                dtype=np.float32,
            )

            if record.ndim == 3 and record.shape[0] == 1:
                record = np.squeeze(record, axis=0)

            if record.ndim != 2:
                raise ValueError("record must be 2D for passive workflow")

            if receivers_to_load is not None:
                record = record[_indices(receivers_to_load, "receivers_to_load"), :]

            file_sampling_freq = f[key].attrs.get("fs")
            if file_sampling_freq is None:
                if user_sampling_freq is None:
                    raise ValueError(
                        "Missing 'fs' attribute. "
                        f"Available attributes: {list(f[key].attrs.keys())}. "
                        "One may use sampling_freq method paramater"
                    )
                sampling_freq = user_sampling_freq
            else:
                file_sampling_freq = float(file_sampling_freq)
                sampling_freq = (
                    user_sampling_freq if user_sampling_freq is not None else file_sampling_freq
                )
                if not np.isclose(
                    file_sampling_freq,
                    sampling_freq,
                ):
                    raise ValueError(
                        f"Sampling frequency mismatch ({sampling_freq} != {file_sampling_freq})"
                    )

            ts = compute_time_vector(
                nt=record.shape[1],
                sampling_freq=sampling_freq,
            )

        if record.shape[0] != len(acquisition.receivers):
            raise ValueError(
                "requires shot.shape[0] = number of receivers. "
                f"Got {record.shape[0]} and {len(acquisition.receivers)}"
            )

        if not acquisition.is_unknown and sort:
            order = np.argsort(acquisition.offsets)
            record = record[order]
            acquisition = type(acquisition)(
                source=acquisition.source,
                receivers=tuple(acquisition.receivers[i] for i in order),
            )

        streams_out.append(
            Stream(
                xt=record,
                ts=ts,
                sampling_freq=sampling_freq,
                acquisition=acquisition,
            )
        )

    return streams_out


def load_gero_active(
    file_paths: Sequence[Path],
    *,
    key: str = "signal",
    sampling_freq: float | None = None,
    acquisitions_per_file: Sequence[Sequence[Acquisition]],  # one acquisition per shot per file
    sort: bool = False,
    sources_to_load: Sequence[int] | None = None,
    receivers_to_load: Sequence[int] | None = None,
) -> list[Stream]:

    if not isinstance(acquisitions_per_file, Sequence) or isinstance(
        acquisitions_per_file, (str, bytes)
    ):
        raise TypeError(
            f"Expected Sequence for acquisitions_per_file, got {type(acquisitions_per_file).__name__}"
        )

    if len(file_paths) != len(acquisitions_per_file):
        raise ValueError(
            f"requires len(file_paths) == len(acquisitions_per_file), got {len(file_paths)} and {len(acquisitions_per_file)}"
        )

    if not all(isinstance(s, Acquisition) for s in acquisitions_per_file for s in s):
        raise TypeError("All elements in acquisitions_per_file must be Acquisition")

    user_sampling_freq = sampling_freq
    streams_out: list[Stream] = []
    for path, acquisitions in zip(file_paths, acquisitions_per_file, strict=False):
        if not path.exists():
            raise FileNotFoundError(path)

        streams_per_file: list[Stream] = []
        with h5py.File(path, "r") as f:
            if key not in f:
                raise ValueError(f"Missing dataset '{key}'. Available objects: {list(f)}")
            shots = np.array(
                dataset(f, key)[:],
                dtype=np.float32,
            )

            if shots.ndim != 3:
                raise ValueError("shots must be 3D for active workflow")

            if sources_to_load is not None:
                shots = shots[_indices(sources_to_load, "sources_to_load"), :, :]

            if receivers_to_load is not None:
                shots = shots[:, _indices(receivers_to_load, "receivers_to_load"), :]

            if shots.shape[0] != len(acquisitions):
                raise ValueError(
                    "requires shots.shape[0] = number of acqusisitions. "
                    f"Got {shots.shape[0]} and {len(acquisitions)}"
                )

            file_sampling_freq = f[key].attrs.get("fs")
            if file_sampling_freq is None:
                if user_sampling_freq is None:
                    raise ValueError(
                        f"Missing 'fs' attribute. Available attributes: {list(f[key].attrs.keys())}"
                    )
                sampling_freq = user_sampling_freq
            else:
                file_sampling_freq = float(file_sampling_freq)
                sampling_freq = (
                    user_sampling_freq if user_sampling_freq is not None else file_sampling_freq
                )
                if not np.isclose(
                    file_sampling_freq,
                    sampling_freq,
                ):
                    raise ValueError(
                        f"Sampling frequency mismatch ({sampling_freq} != {file_sampling_freq})"
                    )

        for shot, acquisition in zip(shots, acquisitions, strict=False):
            if shot.shape[0] != len(acquisition.receivers):
                raise ValueError(
                    "requires shot.shape[0] = number of receivers. "
                    f"Got {shot.shape[0]} and {len(acquisition.receivers)}"
                )
            if not acquisition.is_unknown and sort:
                order = np.argsort(acquisition.offsets)
                shot = shot[order]
                acquisition = type(acquisition)(
                    source=acquisition.source,
                    receivers=tuple(acquisition.receivers[i] for i in order),
                )
            streams_per_file.append(
                Stream(
                    xt=shot,
                    ts=compute_time_vector(
                        nt=shot.shape[1],
                        sampling_freq=sampling_freq,
                    ),
                    sampling_freq=sampling_freq,
                    acquisition=acquisition,
                )
            )

        streams_out.extend(streams_per_file)

    return streams_out


def compute_time_vector(
    nt: int,
    sampling_freq: float,
    delay: float | None = None,
) -> np.ndarray:
    if nt <= 0:
        raise ValueError(f"nt ({nt}) must be greather than 0")
    if sampling_freq <= 0:
        raise ValueError(f"sampling_freq ({sampling_freq}) must be greather than 0")
    time = np.arange(nt, dtype=np.float32) / sampling_freq
    if delay is not None:
        time += delay
    return time


def _is_indices(values: object) -> TypeGuard[Sequence[int]]:
    return (
        isinstance(values, Sequence)
        and not isinstance(values, (str, bytes))
        and all(isinstance(x, int) for x in values)
    )


def _indices(values: object, name: str) -> list[int]:
    if not _is_indices(values):
        raise TypeError(f"Expected Sequence[int] for {name}, got {type(values).__name__}")
    return list(values)


def _receivers_per_file(
    receivers_to_load: Sequence[int] | Sequence[Sequence[int]] | None,
    n_files: int,
) -> list[list[int] | None]:
    """receivers_to_load for each file: the same sequence for every file, or
    one sequence per file."""
    if receivers_to_load is None:
        return [None] * n_files
    if _is_indices(receivers_to_load):
        return [_indices(receivers_to_load, "receivers_to_load")] * n_files
    if (
        isinstance(receivers_to_load, Sequence)
        and not isinstance(receivers_to_load, (str, bytes))
        and all(_is_indices(own) for own in receivers_to_load)
    ):
        if len(receivers_to_load) != n_files:
            raise ValueError(
                f"requires one receivers_to_load per file, got {len(receivers_to_load)} for {n_files} files"
            )
        return [_indices(own, "receivers_to_load") for own in receivers_to_load]
    raise TypeError(
        "Expected Sequence[int], or one Sequence[int] per file, for receivers_to_load, "
        f"got {type(receivers_to_load).__name__}"
    )
