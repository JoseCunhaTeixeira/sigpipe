from pathlib import Path

import h5py
import numpy as np

from src.base.acquisition import UNKNOWN_ACQUISITION
from src.base.stream import Stream


def load_gero_passive(
    path: Path,
    *,
    key: str,
    sampling_freq: float | None = None,
) -> Stream:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        if key not in f:
            raise ValueError(
                f"Missing dataset '{key}'. Available objects: {[key for key in f.keys()]}"
            )
        signals = np.array(
            f[key][:],  # type: ignore
            dtype=np.float32,
        )

        if sampling_freq is None:
            if "fs" in f[key].attrs:
                sampling_freq = float(f[key].attrs["fs"])  # type: ignore
            elif "sampling_freq" in f[key].attrs:
                sampling_freq = float(f[key].attrs["sampling_freq"])  # type: ignore
            else:
                raise ValueError(
                    f"Missing 'sampling_freq' or 'fs' attribute. Available objects: {[key for key in f[key].attrs]}"
                )

        delay = None
        if "delay" in f[key].attrs:
            delay = float(f[key].attrs["delay"])  # type: ignore

        ts = compute_time_vector(
            nt=signals.shape[1],
            sampling_freq=sampling_freq,
            delay=delay,
        )

    return Stream(
        xt=signals,
        ts=ts,
        sampling_freq=sampling_freq,
        acquisition=UNKNOWN_ACQUISITION,
    )


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
