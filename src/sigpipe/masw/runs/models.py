"""What a run records on disk: its manifest, run.json."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from sigpipe.masw.presets import Preset
from sigpipe.masw.profiles import ProfileSummary
from sigpipe.masw.windows import Exclusions

# Stages sigpipe no longer has, which an older run.json may record: dropped when it is read, so
# that the run still loads (its file keeps them). correlation_window: passive-active's own
# surface-wave mute before correlating, removed on 2026-09-28 (the muting's velocities cut the
# same). A passive line's muting and trigger, removed on 2026-09-28 (no shot for a velocity to
# count from).
REMOVED_STAGES = frozenset({"correlation_window"})
REMOVED_PASSIVE_STAGES = frozenset({"muting", "trigger"})


class RunError(ValueError):
    """A run cannot start, or is unknown. Messages say what to fix, for a person in PAC or the
    agent in PACo."""


class RecordOutcome(BaseModel):
    """One record's preprocessing, done once for every window that uses it."""

    model_config = ConfigDict(frozen=True)

    name: str  # the record's file name, e.g. 1.dat
    folder: str  # records/<stem>, inside the run folder: the preprocessed stream and its figure
    status: Literal["succeeded", "failed"]
    duration_s: float | None = None
    error: str | None = None  # "<type>: <message>"; the traceback is in <folder>/error.log


class WindowOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    xmid: float
    folder: str  # xmid_<x>, inside the run folder, as PAC names it
    status: Literal["succeeded", "failed"]
    duration_s: float | None = None
    error: str | None = None  # "<type>: <message>"; the traceback is in <folder>/error.log


class RunManifest(BaseModel):
    """Everything needed to understand or reproduce a run, written as run.json."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    profile: ProfileSummary
    preset: Preset  # resolved: every value the pipelines received
    versions: dict[str, str]
    started_at: datetime
    finished_at: datetime
    n_positions: int  # windows the line allows, before shot selection
    # In file order; empty for a run whose run.json has no records.
    records: tuple[RecordOutcome, ...] = ()
    windows: tuple[WindowOutcome, ...]  # sorted by xmid
    # What G1 took out of the windows: the phase shift done again leaves them out too.
    exclusions: Exclusions = Exclusions()
    # Stopped on request: its windows are those that had finished.
    stopped: bool = False

    @model_validator(mode="before")
    @classmethod
    def _without_removed_stages(cls, data: Any) -> Any:  # noqa: ANN401
        preset = data.get("preset") if isinstance(data, dict) else None
        if not isinstance(preset, dict):
            return data
        removed = REMOVED_STAGES | (
            REMOVED_PASSIVE_STAGES if preset.get("mode") == "passive" else frozenset()
        )
        if removed & preset.keys():
            kept = {name: value for name, value in preset.items() if name not in removed}
            return {**data, "preset": kept}
        return data
