"""The picker's parameters, and what it returns for one dispersion image."""

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sigpipe.base.dispersion_curve import DispersionCurve


class PickingParameters(BaseModel):
    """The picker's knobs: malw-pipe's picker, adapted to Rayleigh waves."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The descriptions are what a form or an agent shows of each value.
    threshold: float = Field(
        default=0.35,
        gt=0,
        le=1,
        description="A bump is a ridge if it reaches this fraction of its column's maximum "
        "(malw-pipe's value).",
    )
    corridor: float = Field(
        default=0.2,
        gt=0,
        lt=1,
        description="Half-width of the corridor around a ridge, as a fraction of its velocity.",
    )
    smoothness: float = Field(
        default=1.0,
        ge=0,
        description="Cost of a relative velocity change per Hz, squared, between neighbouring "
        "frequencies.",
    )
    mode_min_ratio: float = Field(
        default=1.5,
        gt=0,
        description="A mode is kept if its kept points' median coherence reaches this multiple of "
        "the noise floor, 1/sqrt(N).",
    )
    point_min_ratio: float = Field(
        default=1.0,
        gt=0,
        description="A point below this multiple of the noise floor is dropped from the saved "
        "curve.",
    )
    min_relative_coherence: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="A point below this fraction of its mode's median coherence is dropped too: "
        "a sidelobe or noise, not the ridge.",
    )
    max_wavelength: float | None = Field(
        default=None,
        gt=0,
        description="Longest wavelength searched, as a multiple of the window length; null: no "
        "limit.",
    )
    min_contrast: float | None = Field(
        default=0.01,
        gt=0,
        description="A point is kept only where a perfect plane wave for the window varies by "
        "at least this share over the velocity grid (below, the window resolves no velocity: "
        "short windows at low frequencies); null: no floor.",
    )
    max_gap_hz: float | None = Field(
        default=2.0,
        ge=0,
        description="The pick is its longest continuous run: columns too weak to keep are "
        "bridged over at most this many Hz, a wider gap ends the run; null: every kept point.",
    )
    break_slope: float = Field(
        default=2.0,
        gt=0,
        description="A step between consecutive kept points steeper than this |d ln v / d ln f| "
        "ends the run: the ridge broke.",
    )
    fmin: float | None = Field(
        default=None,
        ge=0,
        description="Lowest frequency searched, Hz (G3 cuts a pick where it jumped onto another "
        "mode); null: the image's.",
    )
    fmax: float | None = Field(
        default=None,
        gt=0,
        description="Highest frequency searched, Hz; null: the image's.",
    )
    max_modes: int = Field(
        default=1,
        ge=1,
        description="1 picks M0 only; above 1, each higher mode is searched above the one below.",
    )
    min_frequencies: int = Field(default=5, ge=2, description="Fewest kept points a mode needs.")
    guide: tuple[tuple[float, float], ...] | None = Field(
        default=None,
        description="Points (frequency in Hz, velocity in m/s) the M0 corridor is centred on, "
        "instead of the lowest ridge: e.g. the neighbouring windows' median curve (G4).",
    )


@dataclass(frozen=True, slots=True)
class PickedMode:
    """One tracked mode: the diagnostics of every point, and the curve kept for the inversion."""

    number: int
    frequencies: np.ndarray  # every tracked frequency, Hz
    velocities: np.ndarray  # the pick at each frequency, m/s
    coherence: np.ndarray  # image value along the pick
    # The bound, not the data, decided: the pick is on its corridor's edge, or its column's ridge
    # is cut by the search bounds.
    pinned: np.ndarray
    # Not pinned, not 0 Hz, above the noise floor, near the mode's coherence, and within the
    # pick's continuous run.
    kept: np.ndarray
    noise_floor: float  # 1/sqrt(N) for the N receivers of the window
    curve: DispersionCurve | None  # kept points, resampled over wavelength; None below 2 points

    @property
    def label(self) -> str:
        return f"M{self.number}"
