"""What a run records on disk: its manifest, run.json."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from sigpipe.masw.presets import Preset
from sigpipe.masw.profiles import ProfileSummary
from sigpipe.masw.windows import Exclusions


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
