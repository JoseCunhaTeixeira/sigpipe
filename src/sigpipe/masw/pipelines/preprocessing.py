"""The first steps of the pipelines, on a whole record: trace editing, mute and filter, saved as
a stream for the windows to read."""

from pathlib import Path

from sigpipe.base import Pipeline
from sigpipe.masw.pipelines.common import load_record, stage_kwargs
from sigpipe.masw.presets import ActivePreset, PassivePreset
from sigpipe.masw.profiles import Profile, Record
from sigpipe.transformers import Detrend, Filter, Mute, Save, Shift


def build_preprocessing_pipeline(
    preset: ActivePreset | PassivePreset, record: Record, profile: Profile, output_folder: Path
) -> Pipeline:
    """The preprocessing of one record, written to `output_folder`: the same for every window
    that uses the record, since each step works trace by trace. In the modes that process shots,
    the shot's time origin is corrected first: the trigger is part of the muting (the user,
    2026-09-28), so with the muting on only, by the trigger's t0 or, left to None, by the
    record's own trigger from its file; off, the record is left as recorded (t0 = 0)."""
    load = load_record(record, profile)
    muting = stage_kwargs(preset, "muting")
    on = muting["method"] != "none"
    # The presets with a trigger stage: active and passive-active.
    if "trigger" in type(preset).model_fields:
        t0 = stage_kwargs(preset, "trigger")["t0"]
        shift = (t0 if t0 is not None else record.trigger_s or 0.0) if on else 0.0
        head = load >> Shift(t0=shift)
    else:
        head = Pipeline([load])
    # A muting of the trigger alone (no bound): the shift, and nothing muted.
    bounded = any(muting.get(bound) is not None for bound in ("tmin", "tmax", "vmin", "vmax"))
    return (
        head
        >> Detrend(method="constant")
        >> Detrend(method="linear")
        >> (Mute(**muting) if bounded else Mute(method="none"))
        >> Filter(**stage_kwargs(preset, "filtering"))
        >> Save(folder_path=output_folder)
    )
