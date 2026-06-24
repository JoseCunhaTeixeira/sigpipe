import warnings
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
from obspy import read as read_obspy

from sigproc.base.acquisition import UNKNOWN_ACQUISITION, Acquisition, acquisition_from_kind
from sigproc.base.coordinate import Coordinate, tuples_to_coordinates
from sigproc.base.stream import Stream
from sigproc.dataio._h5 import dataset


def load_stream(
    file_paths: Sequence[Path],
    sort: bool = False,
) -> list[Stream]:
    streams_out: list[Stream] = []
    for path in file_paths:
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


def load_segd(
    file_paths: Sequence[Path],
    acquisitions: Sequence[Acquisition],
    sort: bool = False,
    receivers_to_load: Sequence[int] | None = None,
) -> list[Stream]:

    if not isinstance(acquisitions, Sequence) or isinstance(acquisitions, (str, bytes)):
        raise TypeError(f"Expected Sequence for acquisitions, got {type(acquisitions).__name__}")

    if len(file_paths) != len(acquisitions):
        raise ValueError(
            f"requires len(file_paths) == len(acquisitions), got {len(file_paths)} and {len(acquisitions)}"
        )

    if not all(isinstance(s, Acquisition) for s in acquisitions):
        raise TypeError("All elements in acquisitions must be Acquisition")

    streams_out: list[Stream] = []
    for path, acquisition in zip(file_paths, acquisitions, strict=False):
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

        if receivers_to_load is not None:
            if (
                not isinstance(receivers_to_load, Sequence)
                or isinstance(receivers_to_load, (str, bytes))
                or not all(isinstance(x, int) for x in receivers_to_load)
            ):
                raise TypeError(
                    "Expected Sequence[int | float] for receivers_to_load, "
                    f"got {type(receivers_to_load).__name__}"
                )
            xt = xt[receivers_to_load, :]

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
    acquisition: Acquisition = UNKNOWN_ACQUISITION,
    sort: bool = False,
    receivers_to_load: Sequence[int] | None = None,
) -> list[Stream]:
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
                if (
                    not isinstance(receivers_to_load, Sequence)
                    or isinstance(receivers_to_load, (str, bytes))
                    or not all(isinstance(x, int) for x in receivers_to_load)
                ):
                    raise TypeError(
                        "Expected Sequence[int | float] for receivers_to_load, "
                        f"got {type(receivers_to_load).__name__}"
                    )
                record = record[receivers_to_load, :]

            file_sampling_freq = f[key].attrs.get("fs")
            if file_sampling_freq is None:
                if sampling_freq is None:
                    raise ValueError(
                        "Missing 'fs' attribute. "
                        f"Available attributes: {list(f[key].attrs.keys())}. "
                        "One may use sampling_freq method paramater"
                    )
            else:
                file_sampling_freq = float(file_sampling_freq)
                if sampling_freq is None:
                    sampling_freq = file_sampling_freq
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
    acquisitions: Sequence[Acquisition],
    sort: bool = False,
    sources_to_load: Sequence[int] | None = None,
    receivers_to_load: Sequence[int] | None = None,
) -> list[Stream]:

    if not isinstance(acquisitions, Sequence) or isinstance(acquisitions, (str, bytes)):
        raise TypeError(f"Expected Sequence for acquisitions, got {type(acquisitions).__name__}")

    if len(file_paths) != len(acquisitions):
        raise ValueError(
            f"requires len(file_paths) == len(acquisitions), got {len(file_paths)} and {len(acquisitions)}"
        )

    if not all(isinstance(s, Acquisition) for s in acquisitions):
        raise TypeError("All elements in acquisitions must be Acquisition")

    streams_out: list[Stream] = []
    for path in file_paths:
        if not path.exists():
            raise FileNotFoundError(path)

        streams_per_file = []
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
                if (
                    not isinstance(sources_to_load, Sequence)
                    or isinstance(sources_to_load, (str, bytes))
                    or not all(isinstance(x, int) for x in sources_to_load)
                ):
                    raise TypeError(
                        "Expected Sequence[int | float] for sources_to_load, "
                        f"got {type(sources_to_load).__name__}"
                    )
                shots = shots[sources_to_load, :, :]

            if receivers_to_load is not None:
                if (
                    not isinstance(receivers_to_load, Sequence)
                    or isinstance(receivers_to_load, (str, bytes))
                    or not all(isinstance(x, int) for x in receivers_to_load)
                ):
                    raise TypeError(
                        "Expected Sequence[int | float] for receivers_to_load, "
                        f"got {type(receivers_to_load).__name__}"
                    )
                shots = shots[:, receivers_to_load, :]

            if shots.shape[0] != len(acquisitions):
                raise ValueError(
                    "requires shots.shape[0] = number of acqusisitions. "
                    f"Got {shots.shape[0]} and {len(acquisitions)}"
                )

            file_sampling_freq = f[key].attrs.get("fs")
            if file_sampling_freq is None:
                if sampling_freq is None:
                    raise ValueError(
                        f"Missing 'fs' attribute. Available attributes: {list(f[key].attrs.keys())}"
                    )
            else:
                file_sampling_freq = float(file_sampling_freq)
                if sampling_freq is None:
                    sampling_freq = file_sampling_freq
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
