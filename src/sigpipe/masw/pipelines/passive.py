"""The passive pipeline, from the preprocessed records: noise segments cross-correlated and
stacked, then a dispersion image."""

from pathlib import Path

from sigpipe.base import Pipeline
from sigpipe.masw.pipelines.common import load_preprocessed, stage_kwargs
from sigpipe.masw.presets import PassivePreset
from sigpipe.masw.windows import MASWWindow
from sigpipe.transformers import (
    Apodize,
    Correlate,
    Dispersion,
    Normalize,
    Plot,
    Save,
    Selection,
    Slice,
    Stack,
    Whiten,
)


def build_passive_pipeline(
    preset: PassivePreset, window: MASWWindow, records_folder: Path, output_folder: Path
) -> Pipeline:
    return (
        load_preprocessed(window, records_folder)
        >> Slice(**stage_kwargs(preset, "slicing"))
        >> Selection(**stage_kwargs(preset, "selection"), flip_negatives=True)
        >> Whiten(**stage_kwargs(preset, "whitening"))
        >> Normalize(**stage_kwargs(preset, "normalization"))
        >> Apodize(method="hanning", frac=0.1)
        >> Correlate(method="cross", virtual_source_index=0, part="causal")
        >> Stack(**stage_kwargs(preset, "stacking"))
        >> Plot(folder_path=output_folder)
        >> Save(folder_path=output_folder)
        >> Dispersion(method="phase", **stage_kwargs(preset, "dispersion"))
        >> Plot(folder_path=output_folder, normalize=True)
        >> Save(folder_path=output_folder)
    )
