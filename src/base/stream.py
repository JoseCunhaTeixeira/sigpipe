from dataclasses import dataclass

import numpy as np

from src.base.acquisition import Acquisition


@dataclass(slots=True, frozen=True)
class Stream:
    xt: np.ndarray
    ts: np.ndarray
    sampling_freq: float
    acquisition: Acquisition

    def __post_init__(self) -> None:
        xt = np.asarray(self.xt, dtype=np.float32)
        ts = np.asarray(self.ts, dtype=np.float32)

        validate_shot_inputs(xt, ts, self.acquisition)

        object.__setattr__(self, "xt", xt)
        object.__setattr__(self, "ts", ts)
        object.__setattr__(self, "sampling_freq", float(self.sampling_freq))

        xt.setflags(write=False)
        ts.setflags(write=False)

    @property
    def nt(self) -> int:
        return self.ts.shape[0]

    @property
    def nx(self) -> int:
        return self.xt.shape[0]


def validate_shot_inputs(
    xt: np.ndarray, ts: np.ndarray, acquisition: Acquisition
) -> None:
    if xt.ndim != 2:
        raise ValueError("xt must be 2D")

    if ts.ndim != 1:
        raise ValueError("ts must be 1D")

    nx, nt = xt.shape

    if nt != ts.size:
        raise ValueError("xt and ts mismatch")

    if nx != len(acquisition.receivers) and not acquisition.is_unknown:
        raise ValueError("xt and receivers mismatch")
