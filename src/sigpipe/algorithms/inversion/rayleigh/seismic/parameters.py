"""The MCMC inversion's parameters: the layered model sought, each layer's prior, and the
sampler's effort. Their one definition: inversion_mcmc validates its keyword parameters with
InversionParameters, and PAC's form and PACo's agent send the same JSON."""

from typing import Any, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Each chain keeps one model every SAVE_EVERY iterations after the burn-in: bayesbay keeps
# iteration i when i > n_burnin and (i - n_burnin) is a multiple of save_every.
SAVE_EVERY = 150

# The descriptions are what a form or an agent shows of each value.


class VsLayer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    vs_min: float = Field(default=100.0, gt=0, description="m/s")
    vs_max: float = Field(default=1_000.0, gt=0, description="m/s")
    vs_perturb_std: float = Field(
        default=20.0, gt=0, description="m/s, size of the random steps the sampler takes"
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.vs_max <= self.vs_min:
            raise ValueError("vs_max must be greater than vs_min")
        return self


class ThicknessLayer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    thickness_min: float = Field(default=1.0, gt=0, description="m")
    thickness_max: float = Field(default=10.0, gt=0, description="m")
    thickness_perturb_std: float = Field(
        default=1.0, gt=0, description="m, size of the random steps the sampler takes"
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.thickness_max <= self.thickness_min:
            raise ValueError("thickness_max must be greater than thickness_min")
        return self


class InversionParameters(BaseModel):
    """The MCMC's parameters, validated by inversion_mcmc: the layered model sought, each
    layer's prior, and the sampler's effort. PAC's inversion form sends them, and PACo's agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_layers: int = Field(default=2, ge=2, description="Layers, the half-space included.")
    vs_layers: tuple[VsLayer, ...] = Field(
        default=(VsLayer(), VsLayer()),
        description="Shear-wave velocity bounds of each layer, top down: n_layers of them.",
    )
    thickness_layers: tuple[ThicknessLayer, ...] = Field(
        default=(ThicknessLayer(),),
        description="Thickness bounds of each layer above the half-space: n_layers - 1 of them.",
    )
    n_iterations: int = Field(
        default=100_000,
        gt=0,
        description=f"Iterations of each chain; one model is kept every {SAVE_EVERY} after the "
        "burn-in.",
    )
    n_burnin_iterations: int = Field(
        default=10_000,
        gt=0,
        description="First iterations of each chain, discarded. Left out: a tenth of n_iterations.",
    )
    n_chains: int = Field(default=5, gt=0, description="Independent chains per window.")

    @model_validator(mode="before")
    @classmethod
    def _burnin_follows_iterations(cls, data: Any) -> Any:  # noqa: ANN401
        """A tenth of n_iterations, the defaults' ratio, when only the iterations are given: a
        fixed burn-in of 10,000 would leave nothing to sample from 2,000 iterations."""
        if not isinstance(data, dict):
            return data
        values = cast(dict[str, Any], data)
        iterations = values.get("n_iterations")
        if "n_burnin_iterations" in values or not isinstance(iterations, int | float):
            return values
        return {**values, "n_burnin_iterations": max(1, int(iterations) // 10)}

    @model_validator(mode="after")
    def _check(self) -> Self:
        if len(self.vs_layers) != self.n_layers:
            raise ValueError(f"vs_layers must have length n_layers ({self.n_layers})")
        if len(self.thickness_layers) != self.n_layers - 1:
            raise ValueError(
                f"thickness_layers must have length n_layers - 1 ({self.n_layers - 1})"
            )
        if self.n_iterations - self.n_burnin_iterations < SAVE_EVERY:
            raise ValueError(
                f"n_iterations ({self.n_iterations}) must exceed n_burnin_iterations "
                f"({self.n_burnin_iterations}) by at least {SAVE_EVERY}: each chain keeps one "
                f"model every {SAVE_EVERY} iterations after the burn-in"
            )
        return self
