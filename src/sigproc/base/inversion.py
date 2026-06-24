from dataclasses import dataclass

import numpy as np

from sigproc.base.velocity_model import VelocityModel

from ._repr import array_repr


@dataclass(slots=True, frozen=True)
class InversionResult:
    best: VelocityModel
    smooth_best: VelocityModel
    median: VelocityModel
    smooth_median: VelocityModel
    ensemble: VelocityModel
    n_layers: int
    samples: dict[str, np.ndarray]
    """Raw per-iteration posterior samples (e.g. "vs1", "thick1"), for diagnostics such as marginal plots."""
    misfits: np.ndarray
    """Per-sample RMS misfit, same ordering/length as the arrays in `samples`."""
    dpred: dict[int, np.ndarray]
    """Per-mode posterior-predicted-data samples (mode number -> (n_samples, n_freq_obs)), for diagnostics such as the dispersion-fit percentile band."""
    log: str
    """Captured per-chain statistics (acceptance rates, etc.) printed by the MCMC sampler."""

    def __repr__(self) -> str:
        samples_repr = ", ".join(f"{k!r}: {array_repr(v)}" for k, v in self.samples.items())
        dpred_repr = ", ".join(f"{k!r}: {array_repr(v)}" for k, v in self.dpred.items())
        return (
            f"InversionResult(best={self.best!r}, smooth_best={self.smooth_best!r}, "
            f"median={self.median!r}, smooth_median={self.smooth_median!r}, "
            f"ensemble={self.ensemble!r}, n_layers={self.n_layers!r}, samples={{{samples_repr}}}, "
            f"misfits={array_repr(self.misfits)}, dpred={{{dpred_repr}}}, log={self.log!r})"
        )
