from dataclasses import dataclass

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import hilbert

from sigproc.base.acquisition import Acquisition
from sigproc.base.arrivals import TraceArrivals


@dataclass(slots=True, frozen=True)
class Stream:
    xt: np.ndarray
    ts: np.ndarray
    sampling_freq: float
    acquisition: Acquisition
    arrivals: tuple[TraceArrivals, ...] | None = None

    def __post_init__(self) -> None:
        xt = np.asarray(self.xt, dtype=np.float32)
        ts = np.asarray(self.ts, dtype=np.float32)

        self.validate_shot_inputs()

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

    @property
    def xt_analytic(self) -> np.ndarray:
        npts = np.shape(self.xt)[1]
        nfft = next_fast_len(npts)
        z = np.array(hilbert(self.xt, N=nfft, axis=1), np.complex64)[:, :npts]
        return z

    @property
    def xt_envelope(self) -> np.ndarray:
        return np.abs(self.xt_analytic).astype(np.float32)

    @property
    def xt_phase(self) -> np.ndarray:
        return np.angle(self.xt_analytic).astype(np.float32)

    def validate_shot_inputs(self) -> None:
        if self.xt.ndim != 2:
            raise ValueError("xt must be 2D")

        if self.ts.ndim != 1:
            raise ValueError("ts must be 1D")

        nx, nt = self.xt.shape

        if nt != self.ts.size:
            raise ValueError("nt and ts mismatch")

        if nx != len(self.acquisition.receivers) and not self.acquisition.is_unknown:
            raise ValueError("xt and receivers mismatch")

        if self.arrivals is not None:
            if not isinstance(self.arrivals, tuple):
                raise TypeError(
                    f"Expected tuple[TraceArrivals, ...], got {type(self.arrivals).__name__}"
                )

            if len(self.arrivals) != self.nx:
                raise ValueError(
                    f"Expected {self.nx} arrivals (one per trace), got {len(self.arrivals)}"
                )

            if not all(isinstance(a, TraceArrivals) for a in self.arrivals):
                raise TypeError("All arrivals must be TraceArrivals")
