"""The first steps of the pipelines, on a whole record: trace editing, mute and filter, saved as
a stream for the windows to read."""

from pathlib import Path
from typing import Any

from sigpipe.base import Pipeline
from sigpipe.base.stream import Stream
from sigpipe.masw.pipelines.common import load_record, stage_kwargs
from sigpipe.masw.presets import ActivePreset, PassivePreset
from sigpipe.masw.profiles import Profile, Record
from sigpipe.transformers import Detrend, Filter, Mute, Plot, Save, Shift


def build_preprocessing_pipeline(
    preset: ActivePreset | PassivePreset,
    record: Record,
    profile: Profile,
    output_folder: Path | None,
    muted: bool = True,
) -> Pipeline:
    """The preprocessing of one record, written to `output_folder` (None: in memory): the same
    for every window that uses the record, since each step works trace by trace. In the modes
    that process shots, the shot's time origin is corrected first (trigger_shift_s): the trigger
    is part of the muting (the user, 2026-09-28). Not `muted`, the record before its muting:
    neither its trigger shifted nor muted. A passive line has neither (stages.py)."""
    load = load_record(record, profile)
    fields = type(preset).model_fields
    muting = _muting(preset)
    # The presets with a trigger stage: active and passive-active.
    head = (
        load >> Shift(t0=trigger_shift_s(preset, record))
        if "trigger" in fields and muted
        else Pipeline([load])
    )
    # A muting of the trigger alone (no bound): the shift, and nothing muted.
    bounded = any(muting.get(bound) is not None for bound in ("tmin", "tmax", "vmin", "vmax"))
    chain = (
        head
        >> Detrend(method="constant")
        >> Detrend(method="linear")
        >> (Mute(**muting) if bounded and muted else Mute(method="none"))
        >> Filter(**stage_kwargs(preset, "filtering"))
    )
    if output_folder is None:
        return chain
    # Saved, and its figure, as PAC's gather view draws it.
    return chain >> Save(folder_path=output_folder) >> Plot(folder_path=output_folder)


def _muting(preset: ActivePreset | PassivePreset) -> dict[str, Any]:
    fields = type(preset).model_fields
    return stage_kwargs(preset, "muting") if "muting" in fields else {"method": "none"}


def trigger_shift_s(preset: ActivePreset | PassivePreset, record: Record) -> float:
    """The shift `preset`'s preprocessing gives `record`'s time origin, s: with its muting on,
    the trigger's t0 or, left to None, the record's own trigger from its file; off, none, the
    record left as recorded (a passive line has no trigger)."""
    if "trigger" not in type(preset).model_fields or _muting(preset)["method"] == "none":
        return 0.0
    t0 = stage_kwargs(preset, "trigger")["t0"]
    return float(t0 if t0 is not None else record.trigger_s or 0.0)


def shot_time_s(preset: ActivePreset | PassivePreset, record: Record) -> float:
    """Where `record`'s shot is on it as recorded, s: where its muting's shift puts the time
    origin (trigger_shift_s), else its file's trigger, else 0."""
    if _muting(preset)["method"] != "none" and "trigger" in type(preset).model_fields:
        return trigger_shift_s(preset, record)
    return float(record.trigger_s or 0.0)


def unmuted_record(
    preset: ActivePreset | PassivePreset, record: Record, profile: Profile
) -> Stream:
    """`record` preprocessed as `preset` has it but before its muting, in memory: neither its
    trigger shifted (part of the muting) nor muted, its times the file's, its shot at
    shot_time_s. What its signal checks measure: a muting zeroes their noise window after the
    slowest arrival, a trigger's shift drops the one before the trigger."""
    stream: Stream = build_preprocessing_pipeline(preset, record, profile, None, muted=False).run(
        show_log=False
    )[0]
    return stream
