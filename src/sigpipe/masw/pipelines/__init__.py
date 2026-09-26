"""sigpipe pipelines built from a resolved preset, in two stages: each record preprocessed once
(trace editing, mute, filter), then one image pipeline per MASW window on the preprocessed
records.

Port of PAC's adapters: the same steps in the same order, with the same fixed values, cut at the
window. The preset's stages give the tunable ones.
"""

from .active import build_active_pipeline
from .common import PREPROCESSED, load_preprocessed, record_folder
from .passive import build_passive_pipeline
from .passive_active import build_passive_active_pipeline
from .preprocessing import build_preprocessing_pipeline
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
]
