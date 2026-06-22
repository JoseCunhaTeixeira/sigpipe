from dataclasses import dataclass

import numpy as np

from sigproc.base.velocity_model import VelocityModel

from ._repr import array_repr


@dataclass(slots=True, frozen=True)
class InversionResult:
    best: VelocityModel
    median: VelocityModel
    ensemble: VelocityModel
    n_layers: int
    samples: dict[str, np.ndarray]
    """Raw per-iteration posterior samples (e.g. "vs1", "thick1"), for diagnostics such as marginal plots."""

    def __repr__(self) -> str:
        samples_repr = ", ".join(f"{k!r}: {array_repr(v)}" for k, v in self.samples.items())
        return (
            f"InversionResult(best={self.best!r}, median={self.median!r}, "
            f"ensemble={self.ensemble!r}, n_layers={self.n_layers!r}, samples={{{samples_repr}}})"
        )
