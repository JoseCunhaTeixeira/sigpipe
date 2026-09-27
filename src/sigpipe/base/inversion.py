from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sigpipe.base.velocity_model import VelocityModel

from ._repr import array_repr


@dataclass(slots=True, frozen=True)
class LayeredSamples:
    """The kept models, chain after chain (as many from each), as layers: a row per model, its
    interfaces' depths then its layers' Vs top down, padded with NaN beyond its layers. The one
    form of both layerings' samples, fixed or chosen by the data."""

    depths: np.ndarray  # (models, most layers - 1) m
    vs: np.ndarray  # (models, most layers) m/s, the half-space the last value of a row
    n_chains: int

    @property
    def layers(self) -> np.ndarray:
        """Each model's layers, the half-space included."""
        return np.sum(~np.isnan(self.vs), axis=1)

    def model(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """One model's interfaces' depths and layers' Vs."""
        vs = self.vs[index]
        depths = self.depths[index]
        return depths[~np.isnan(depths)], vs[~np.isnan(vs)]

    def at(self, depths: np.ndarray) -> np.ndarray:
        """Every model's Vs at `depths` (models x depths); a depth on an interface belongs to the
        layer below."""
        depths = np.asarray(depths, dtype=float)
        # The interfaces at or above each depth; the padding (NaN) never is.
        interfaces = np.nan_to_num(self.depths, nan=np.inf)
        above = np.sum(interfaces[:, None, :] <= depths[None, :, None], axis=2)
        return np.take_along_axis(self.vs, above, axis=1)

    def per_chain(self, values: np.ndarray) -> np.ndarray:
        """`values` (one per model, or models x anything) split by chain: chains x models x ..."""
        per = values.shape[0] // self.n_chains
        return values[: per * self.n_chains].reshape(self.n_chains, per, *values.shape[1:])


@dataclass(slots=True, frozen=True)
class InversionResult:
    best: VelocityModel
    smooth_best: VelocityModel
    median: VelocityModel
    smooth_median: VelocityModel
    ensemble: VelocityModel
    n_layers: int
    """The median model's layers, the half-space included."""
    samples: dict[str, np.ndarray]
    """Each kept model's named values, for diagnostics such as marginal plots: the fixed
    layering's own (vs1, ..., thick1, ...), the layers' count ("layers") and the noise factor
    ("noise")."""
    misfits: np.ndarray
    """Per-sample RMS misfit (m/s), same ordering/length as the arrays in `samples`."""
    dpred: dict[int, np.ndarray]
    """Per-mode posterior-predicted-data samples (mode number -> (n_samples, n_freq_obs)), for diagnostics such as the dispersion-fit percentile band."""
    log: str
    """How the chains ran, in words."""
    profiles: LayeredSamples | None = None
    """Every kept model as layers, chain after chain."""
    steps: dict[str, float] = field(default_factory=dict)
    """Each sampled parameter's typical move (vs1, ..., thick1, ...), in the fixed layering."""
    tuning: tuple[tuple[float, float], ...] = ()
    """Each trial run's step factor and acceptance rate (%): runs saved before 2026-09-27."""
    acceptance: tuple[float, ...] = ()
    """Each chain's acceptance rate over the run (%), the burn-in included."""
    parameters: dict[str, Any] = field(default_factory=dict)
    """The parameters as the chains ran them: the free layering's bounds found from the curves."""

    def __repr__(self) -> str:
        samples_repr = ", ".join(f"{k!r}: {array_repr(v)}" for k, v in self.samples.items())
        dpred_repr = ", ".join(f"{k!r}: {array_repr(v)}" for k, v in self.dpred.items())
        return (
            f"InversionResult(best={self.best!r}, smooth_best={self.smooth_best!r}, "
            f"median={self.median!r}, smooth_median={self.smooth_median!r}, "
            f"ensemble={self.ensemble!r}, n_layers={self.n_layers!r}, samples={{{samples_repr}}}, "
            f"misfits={array_repr(self.misfits)}, dpred={{{dpred_repr}}}, log={self.log!r}, "
            f"steps={self.steps!r}, tuning={self.tuning!r}, "
            f"acceptance={self.acceptance!r})"
        )
