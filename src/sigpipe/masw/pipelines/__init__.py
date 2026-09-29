"""sigpipe pipelines built from a resolved preset, in two stages: each record preprocessed once
(trace editing, mute, filter), then one image pipeline per MASW window on the preprocessed
records.

The preset's stages give the tunable values; the others are fixed.
"""

from .active import build_active_pipeline
from .common import PREPROCESSED, load_preprocessed, record_folder
from .passive import build_passive_pipeline
from .passive_active import build_passive_active_pipeline, window_correlations, window_records
from .preprocessing import (
    build_preprocessing_pipeline,
    shot_time_s,
    trigger_shift_s,
    unmuted_record,
)
from .registry import PIPELINE_BUILDERS, build_image_pipeline

__all__ = [
    "PIPELINE_BUILDERS",
    "PREPROCESSED",
    "build_active_pipeline",
    "build_image_pipeline",
    "build_passive_active_pipeline",
    "build_passive_pipeline",
    "build_preprocessing_pipeline",
    "load_preprocessed",
    "record_folder",
    "shot_time_s",
    "trigger_shift_s",
    "unmuted_record",
    "window_correlations",
    "window_records",
]
