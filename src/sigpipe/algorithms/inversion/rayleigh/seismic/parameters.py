"""The seismic inversion's parameters: how the layers are sought, their priors, and the chains'
effort. Their one definition: inversion_mcmc validates its keyword parameters with
InversionParameters, and PAC's form and PACo's agent send the same JSON.

Two layerings:
- "free" (the default): the data choose the number of layers (transdimensional.py), within a Vs
  range, down to a depth and up to a number of layers, each left out uses the curves' own;
- "fixed": the layers given, each with its Vs and thickness ranges or values (sampler.py).
In both, a layer's Vs may fall below the one above it by `max_vs_drop` at most.
"""

from __future__ import annotations

from typing import Any, Literal, Self, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Each chain keeps one model every SAVE_EVERY iterations after the burn-in: iteration i when
# i > n_burnin and (i - n_burnin) is a multiple of SAVE_EVERY.
SAVE_EVERY = 150
# Keys of runs saved before 2026-09-27, accepted and dropped: the trial runs that tuned the steps
# of the sampler sigpipe used then.
_RETIRED = ("tune_steps",)
_FIXED_KEYS = ("n_layers", "vs_layers", "thickness_layers")

# The descriptions are what a form or an agent shows of each value.


class VsLayer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    vs_min: float = Field(default=100.0, gt=0, description="m/s")
    vs_max: float = Field(default=1_000.0, gt=0, description="m/s")
    vs_perturb_std: float = Field(
        default=20.0,
        gt=0,
        description="m/s, not used: the chains' moves follow the posterior (kept for the runs "
        "saved before)",
    )
    vs_fixed: float | None = Field(
        default=None,
        gt=0,
        description="m/s: the layer's Vs, fixed, not sampled (its bounds then unused)",
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.vs_max <= self.vs_min:
            raise ValueError("vs_max must be greater than vs_min")
        return self

    @property
    def bounds(self) -> tuple[float, float]:
        """The range the layer's Vs takes: its prior's, or its fixed value's own."""
        if self.vs_fixed is not None:
            return self.vs_fixed, self.vs_fixed
        return self.vs_min, self.vs_max


class ThicknessLayer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    thickness_min: float = Field(default=1.0, gt=0, description="m")
    thickness_max: float = Field(default=10.0, gt=0, description="m")
    thickness_perturb_std: float = Field(
        default=1.0,
        gt=0,
        description="m, not used: the chains' moves follow the posterior (kept for the runs "
        "saved before)",
    )
    thickness_fixed: float | None = Field(
        default=None,
        gt=0,
        description="m: the layer's thickness, fixed, not sampled (its bounds then unused)",
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.thickness_max <= self.thickness_min:
            raise ValueError("thickness_max must be greater than thickness_min")
        return self

    @property
    def bounds(self) -> tuple[float, float]:
        """The range the layer's thickness takes: its prior's, or its fixed value's own."""
        if self.thickness_fixed is not None:
            return self.thickness_fixed, self.thickness_fixed
        return self.thickness_min, self.thickness_max


class FreeLayers(BaseModel):
    """The priors when the data choose the layers; each bound left out follows the curves."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vs_min: float | None = Field(
        default=None,
        gt=0,
        description="m/s, every layer's least Vs. Left out: half the slowest pick.",
    )
    vs_max: float | None = Field(
        default=None,
        gt=0,
        description="m/s, every layer's greatest Vs. Left out: three times the fastest pick.",
    )
    depth_min: float | None = Field(
        default=None,
        gt=0,
        description="m, the shallowest interface. Left out: a third of the shortest picked "
        "wavelength (thinner is not resolved).",
    )
    depth_max: float | None = Field(
        default=None,
        gt=0,
        description="m, the deepest interface. Left out: half the longest picked wavelength.",
    )
    max_layers: int = Field(
        default=8, ge=1, le=20, description="Layers at most, the half-space included."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.vs_min is not None and self.vs_max is not None and self.vs_max <= self.vs_min:
            raise ValueError("vs_max must be greater than vs_min")
        if (
            self.depth_min is not None
            and self.depth_max is not None
            and self.depth_max <= self.depth_min
        ):
            raise ValueError("depth_max must be greater than depth_min")
        return self

    def resolved(self, fs: np.ndarray, vs: np.ndarray) -> FreeLayers:
        """Each bound left out, from the picks (frequencies Hz, phase velocities m/s)."""
        wavelengths = np.asarray(vs, dtype=float) / np.asarray(fs, dtype=float)
        vs_min = self.vs_min if self.vs_min is not None else round(0.5 * float(np.min(vs)))
        vs_max = self.vs_max if self.vs_max is not None else round(3.0 * float(np.max(vs)))
        depth_min = (
            self.depth_min
            if self.depth_min is not None
            else round(float(np.min(wavelengths)) / 3, 2)
        )
        depth_max = (
            self.depth_max
            if self.depth_max is not None
            else round(float(np.max(wavelengths)) / 2, 2)
        )
        return FreeLayers(
            vs_min=max(1.0, vs_min),
            vs_max=max(vs_max, vs_min + 1.0),
            depth_min=depth_min,
            depth_max=max(depth_max, 1.5 * depth_min),
            max_layers=self.max_layers,
        )


class InversionParameters(BaseModel):
    """The inversion's parameters, validated by inversion_mcmc: the layering and its priors,
    the Vs drop allowed, and the chains' effort. PAC's inversion form sends them, and PACo's
    agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layering: Literal["free", "fixed"] = Field(
        default="free",
        description="free: the data choose the number of layers, within `free`'s bounds; "
        "fixed: the layers given in n_layers, vs_layers and thickness_layers.",
    )
    free: FreeLayers = Field(default=FreeLayers(), description="The priors when layering is free.")
    n_layers: int = Field(
        default=2, ge=2, description="Fixed layering: layers, the half-space included."
    )
    vs_layers: tuple[VsLayer, ...] = Field(
        default=(VsLayer(), VsLayer()),
        description="Fixed layering: each layer's Vs bounds, top down, n_layers of them.",
    )
    thickness_layers: tuple[ThicknessLayer, ...] = Field(
        default=(ThicknessLayer(),),
        description="Fixed layering: each layer's thickness bounds above the half-space, "
        "n_layers - 1 of them.",
    )
    max_vs_drop: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="The share by which a layer's Vs may fall below the one above it (0: never, "
        "1: any). A layer much stiffer than the one below lets the forward model's fundamental "
        "mode be a wave trapped in the soft layer, which fits slow picks the surface never "
        "recorded.",
    )
    n_iterations: int = Field(
        default=200_000,
        gt=0,
        description=f"Iterations of each chain; one model is kept every {SAVE_EVERY} after the "
        "burn-in.",
    )
    n_burnin_iterations: int = Field(
        default=50_000,
        gt=0,
        description="First iterations of each chain, discarded. Left out: a quarter of "
        "n_iterations.",
    )
    # Two at least (the user, 2026-09-28): one chain's halves agree even where two chains would
    # settle on two solutions (an interface above or below, another count of layers).
    n_chains: int = Field(default=5, ge=2, description="Chains per window, compared to judge them.")

    @model_validator(mode="before")
    @classmethod
    def _as_given(cls, data: Any) -> Any:  # noqa: ANN401
        """The keys of runs saved before dropped; the fixed layering when layers are given
        without a layering (runs saved before, and requests naming their layers); a quarter of
        n_iterations for the burn-in when only the iterations are given."""
        if not isinstance(data, dict):
            return data
        values = {
            key: value for key, value in cast(dict[str, Any], data).items() if key not in _RETIRED
        }
        if "layering" not in values and any(key in values for key in _FIXED_KEYS):
            values["layering"] = "fixed"
        iterations = values.get("n_iterations")
        if "n_burnin_iterations" not in values and isinstance(iterations, int | float):
            values["n_burnin_iterations"] = max(1, int(iterations) // 4)
        return values

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.layering == "fixed":
            if len(self.vs_layers) != self.n_layers:
                raise ValueError(f"vs_layers must have length n_layers ({self.n_layers})")
            if len(self.thickness_layers) != self.n_layers - 1:
                raise ValueError(
                    f"thickness_layers must have length n_layers - 1 ({self.n_layers - 1})"
                )
            if len(self.fixed()) == 2 * self.n_layers - 1:
                raise ValueError(
                    "every layer's Vs and thickness are fixed: nothing is left to sample; free one"
                )
        if self.n_iterations - self.n_burnin_iterations < SAVE_EVERY:
            raise ValueError(
                f"n_iterations ({self.n_iterations}) must exceed n_burnin_iterations "
                f"({self.n_burnin_iterations}) by at least {SAVE_EVERY}: each chain keeps one "
                f"model every {SAVE_EVERY} iterations after the burn-in"
            )
        return self

    def resolved(self, fs: np.ndarray, vs: np.ndarray) -> InversionParameters:
        """The parameters with the free layering's bounds left out found from the picks."""
        if self.layering != "free":
            return self
        return self.model_copy(update={"free": self.free.resolved(fs, vs)})

    @property
    def bottom(self) -> float:
        """The bottom of the models built (m): the deepest interface the prior allows (a
        thickness fixed, its own) plus 1 m; in the free layering, a quarter below its deepest
        interface (its bounds resolved)."""
        if self.layering == "fixed":
            return float(sum(layer.bounds[1] for layer in self.thickness_layers)) + 1.0
        if self.free.depth_max is None:
            raise ValueError("the free layering's depth is found from the curves: resolve it")
        return round(1.25 * self.free.depth_max, 2)

    @property
    def least_ratio(self) -> float:
        """A layer's Vs over the one above it, at least."""
        return 1.0 - self.max_vs_drop

    def fixed(self) -> dict[str, float]:
        """The values fixed, by parameter name (vs1, ..., thick1, ...): not sampled. None in the
        free layering."""
        if self.layering != "fixed":
            return {}
        found = {
            f"vs{i + 1}": layer.vs_fixed
            for i, layer in enumerate(self.vs_layers)
            if layer.vs_fixed is not None
        }
        return found | {
            f"thick{i + 1}": layer.thickness_fixed
            for i, layer in enumerate(self.thickness_layers)
            if layer.thickness_fixed is not None
        }


class SavedInversionParameters(InversionParameters):
    """The parameters a run saved: one chain too, as the runs before 2026-09-28 could run."""

    n_chains: int = Field(default=5, ge=1, description="Chains per window, compared to judge them.")
