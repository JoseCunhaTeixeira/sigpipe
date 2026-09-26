"""The first steps of PAC's pipelines, on a whole record: trace editing, mute and filter, saved
as a stream for the windows to read."""

from pathlib import Path

from sigpipe.base import Pipeline
from sigpipe.masw.pipelines.common import load_record, stage_kwargs
from sigpipe.masw.presets import ActivePreset, PassivePreset
from sigpipe.masw.profiles import Profile, Record
from sigpipe.transformers import Detrend, Filter, Mute, Plot, Save, Shift


def build_preprocessing_pipeline(
    preset: ActivePreset | PassivePreset, record: Record, profile: Profile, output_folder: Path
) -> Pipeline:
    """The preprocessing of one record, written to `output_folder`: the same for every window
    that uses the record, since each step works trace by trace. A shot's trigger is corrected
    first, in the modes that process shots (t0 = 0 by default: no change)."""
    load = load_record(record, profile)
    # The presets with a trigger stage: active and passive-active (whose correction PACo's signal
    # QC asked for and never got, rejecting the demo's two shots, 2026-09-26).
    head = (
        load >> Shift(**stage_kwargs(preset, "trigger"))
        if "trigger" in type(preset).model_fields
        else Pipeline([load])
    )
    return (
        head
        >> Detrend(method="constant")
        >> Detrend(method="linear")
        >> Mute(**stage_kwargs(preset, "muting"))
        >> Filter(**stage_kwargs(preset, "filtering"))
        >> Plot(folder_path=output_folder)
        >> Save(folder_path=output_folder)
    )
