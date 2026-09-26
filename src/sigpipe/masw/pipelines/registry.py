"""Which image pipeline each preset builds."""

from collections.abc import Callable
from pathlib import Path

from sigpipe.base import Pipeline
from sigpipe.masw.pipelines.active import build_active_pipeline
from sigpipe.masw.pipelines.passive import build_passive_pipeline
from sigpipe.masw.pipelines.passive_active import build_passive_active_pipeline
from sigpipe.masw.presets import ActivePreset, PassivePreset
from sigpipe.masw.profiles import ProcessingMode
from sigpipe.masw.windows import MASWWindow

PIPELINE_BUILDERS: dict[str, Callable[..., Pipeline]] = {
    ProcessingMode.ACTIVE: build_active_pipeline,
    ProcessingMode.PASSIVE: build_passive_pipeline,
    ProcessingMode.PASSIVE_ACTIVE: build_passive_active_pipeline,
}


def build_image_pipeline(
    preset: ActivePreset | PassivePreset,
    window: MASWWindow,
    records_folder: Path,
    output_folder: Path,
) -> Pipeline:
    """The pipeline of `preset` for one window, from the preprocessed records in
    `records_folder`, writing its figures and results to `output_folder`."""
    builder = PIPELINE_BUILDERS[preset.mode]
    return builder(
        preset=preset, window=window, records_folder=records_folder, output_folder=output_folder
    )
