"""Data shapes of a profile: the validated Profile used by processing, and its short summary
(what a run records of it, and what PACo's agent reads)."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from sigpipe.base.coordinate import Coordinate


class ProfileKind(StrEnum):
    ACTIVE = "active"
    PASSIVE = "passive"


class ProcessingMode(StrEnum):
    """How a profile is processed, PAC's three modes: an active profile as shots (active) or by
    interferometry on its shots (passive-active: each shot's gather cross-correlated with the
    receiver nearest the shot, the correlations stacked); a passive profile as ambient noise."""

    ACTIVE = "active"
    PASSIVE = "passive"
    PASSIVE_ACTIVE = "passive-active"


# The modes each kind of profile can be processed in, its own first.
MODES: dict[ProfileKind, tuple[ProcessingMode, ...]] = {
    ProfileKind.ACTIVE: (ProcessingMode.ACTIVE, ProcessingMode.PASSIVE_ACTIVE),
    ProfileKind.PASSIVE: (ProcessingMode.PASSIVE,),
}


class Record(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    n_traces: int
    sampling_rate_hz: float
    duration_s: float
    source: Coordinate | None


class Profile(BaseModel):
    """Everything processing needs to know about a profile, validated."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: ProfileKind
    folder: Path
    records: tuple[Record, ...]
    receivers: tuple[Coordinate, ...]

    @property
    def sampling_rate_hz(self) -> float:
        return self.records[0].sampling_rate_hz

    @property
    def nyquist_hz(self) -> float:
        return self.sampling_rate_hz / 2


class ProfileSummary(BaseModel):
    """Short description of a profile, with no per-record or per-trace lists."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: ProfileKind
    n_records: int
    n_receivers: int
    receiver_x_range_m: tuple[float, float]
    receiver_spacing_m: float
    sampling_rate_hz: float
    nyquist_hz: float
    record_duration_range_s: tuple[float, float]
    source_x_range_m: tuple[float, float] | None
    # How it can be processed (run_processing's "mode"), its own first: PAC's three modes.
    modes: tuple[ProcessingMode, ...] = ()
