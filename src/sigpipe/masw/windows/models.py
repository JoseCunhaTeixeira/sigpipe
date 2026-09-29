"""MASW window parameters and windows, with PAC's field names."""

import math
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sigpipe.base.acquisition import LinearAcquisition


class MASWParameters(BaseModel):
    """Every field has a default, so that a partial override keeps the other values."""

    # Unknown keys are errors: presets take these parameters in overrides.
    model_config = ConfigDict(frozen=True, extra="forbid")

    # The descriptions say "receivers": a length in metres is an easy mistake.
    length: int = Field(default=5, ge=3, description="receivers, not metres")
    step: int = Field(default=1, gt=0, description="receivers between starts")
    # m from the source to the window's middle, both bounds exclusive; each may be left out, none
    # (the nearest from 0, the farthest at any distance): no stand-in value (the user,
    # 2026-09-28; older runs recorded 0 and 1,000 m).
    distance_min: float | None = Field(default=None, ge=0, description="m; null: none")
    distance_max: float | None = Field(default=None, gt=0, description="m; null: none")

    @model_validator(mode="after")
    def _check_distances(self) -> Self:
        if (
            self.distance_min is not None
            and self.distance_max is not None
            and self.distance_max <= self.distance_min
        ):
            raise ValueError(
                f"distance_max ({self.distance_max:g}) must be greater than "
                f"distance_min ({self.distance_min:g})"
            )
        return self


class Exclusions(BaseModel):
    """What a run leaves out (PACo's signal QC decides it): records no window uses, and traces
    (receiver indices, by record file name) that the windows holding them leave out: from that
    record's image in an active window (from all of them when at least half its records
    excluded the trace), from every record of a passive window."""

    model_config = ConfigDict(frozen=True)

    records: tuple[str, ...] = ()
    traces: dict[str, tuple[int, ...]] = {}

    def with_traces(self, record: str, traces: tuple[int, ...]) -> Exclusions:
        merged = tuple(sorted({*self.traces.get(record, ()), *traces}))
        return self.model_copy(update={"traces": {**self.traces, record: merged}})

    def with_record(self, record: str) -> Exclusions:
        return self.model_copy(update={"records": tuple(sorted({*self.records, record}))})


class MASWWindow(BaseModel):
    xmid: float
    selected_files: list[Path]
    receiver_indices: list[int]
    acquisitions: list[LinearAcquisition]
    # The receivers each record gives the window, when some of its traces are left out: its image
    # is made from those alone (None: every record gives all of receiver_indices).
    record_receivers: list[list[int]] | None = None


def nearest_offset(window: MASWWindow) -> float | None:
    """The distance from the window's nearest shot to its nearest receiver, over the records it
    uses; None when it has none. G3 reports the curve's longest wavelengths against it (the
    near field), PAC alike."""
    distances = [
        math.dist(
            (acquisition.source.x, acquisition.source.y, acquisition.source.z),
            (receiver.x, receiver.y, receiver.z),
        )
        for acquisition in window.acquisitions
        for receiver in acquisition.receivers
    ]
    return round(min(distances), 3) if distances else None
