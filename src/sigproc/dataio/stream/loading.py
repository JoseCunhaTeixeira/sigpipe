from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
from obspy import read as read_obspy

from sigproc.base.acquisition import UNKNOWN_ACQUISITION, Acquisition
from sigproc.base.coordinate import Coordinate, tuples_to_coordinates
from sigproc.base.stream import Stream


def load_stream(
    path: Path,
    sort: bool = False,
) -> Stream:
    path = path.with_suffix(".hd5")

    with h5py.File(path, "r") as file:
        xt = np.asarray(
            file["xt"][:],  # type: ignore
            dtype=np.float32,
        )

        ts = np.asarray(
            file["ts"][:],  # type: ignore
            dtype=np.float32,
        )

        sampling_freq = float(file["sampling_freq"][()])  # type: ignore

        source = tuple(file["source"][:])  # type: ignore
        receivers = list(file["receivers"][:])  # type: ignore

    acquisition = Acquisition(
        source=Coordinate(*source),
        receivers=tuples_to_coordinates(receivers),
    )

    if not acquisition.is_unknown and sort:
        order = np.argsort(acquisition.offsets)
        xt = xt[order]
        acquisition = Acquisition(
            source=acquisition.source,
            receivers=tuple(acquisition.receivers[i] for i in order),
        )

    return Stream(
        xt=xt,
        ts=ts,
        sampling_freq=sampling_freq,
        acquisition=acquisition,
    )


def load_obspy(
    path: Path,
    acquisition: Acquisition = UNKNOWN_ACQUISITION,
    sort: bool = False,
    receivers_to_load: Sequence[int] | None = None,
) -> Stream:
    ob_stream = read_obspy(path)
    sampling_freq = ob_stream[0].stats.sampling_rate
    nx = len(ob_stream)
    nt = ob_stream[0].stats.npts  # type: ignore
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
        acquisition = Acquisition(
            source=acquisition.source,
            receivers=tuple(acquisition.receivers[i] for i in order),
        )

    return Stream(
        xt=xt,
        ts=ts,
        sampling_freq=sampling_freq,
        acquisition=acquisition,
    )


def load_gero_passive(
    path: Path,
    *,
    key: str = "signal",
    sampling_freq: float | None = None,
    acquisition: Acquisition = UNKNOWN_ACQUISITION,
    sort: bool = False,
    receivers_to_load: Sequence[int] | None = None,
) -> Stream:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        if key not in f:
            raise ValueError(
                f"Missing dataset '{key}'. "
                f"Available objects: {[key for key in f.keys()]}"
            )
        record = np.array(
            f[key][:],  # type: ignore
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
                    f"Sampling frequency mismatch "
                    f"({sampling_freq} != {file_sampling_freq})"
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
        acquisition = Acquisition(
            source=acquisition.source,
            receivers=tuple(acquisition.receivers[i] for i in order),
        )

    return Stream(
        xt=record,
        ts=ts,
        sampling_freq=sampling_freq,
        acquisition=acquisition,
    )


def load_gero_active(
    path: Path,
    *,
    key: str = "signal",
    sampling_freq: float | None = None,
    acquisitions: Sequence[Acquisition],
    sort: bool = False,
    sources_to_load: Sequence[int] | None = None,
    receivers_to_load: Sequence[int] | None = None,
) -> list[Stream]:
    if not path.exists():
        raise FileNotFoundError(path)
    streams = []
    with h5py.File(path, "r") as f:
        if key not in f:
            raise ValueError(
                f"Missing dataset '{key}'. "
                f"Available objects: {[key for key in f.keys()]}"
            )
        shots = np.array(
            f[key][:],  # type: ignore
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
                    "Missing 'fs' attribute. "
                    f"Available attributes: {list(f[key].attrs.keys())}"
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
                    f"Sampling frequency mismatch "
                    f"({sampling_freq} != {file_sampling_freq})"
                )

    for shot, acquisition in zip(shots, acquisitions):
        if shot.shape[0] != len(acquisition.receivers):
            raise ValueError(
                "requires shot.shape[0] = number of receivers. "
                f"Got {shot.shape[0]} and {len(acquisition.receivers)}"
            )
        if not acquisition.is_unknown and sort:
            order = np.argsort(acquisition.offsets)
            shot = shot[order]
            acquisition = Acquisition(
                source=acquisition.source,
                receivers=tuple(acquisition.receivers[i] for i in order),
            )
        streams.append(
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

    return streams


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
