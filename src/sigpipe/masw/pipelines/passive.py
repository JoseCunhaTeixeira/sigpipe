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
    PlotSelection,
    Save,
    Selection,
    Slice,
    Stack,
    Whiten,
)


def build_passive_pipeline(
    preset: PassivePreset, window: MASWWindow, records_folder: Path, output_folder: Path
) -> Pipeline:
    selection = Selection(**stage_kwargs(preset, "selection"), flip_negatives=True)
    return (
        load_preprocessed(window, records_folder)
        >> Slice(**stage_kwargs(preset, "slicing"))
        >> selection
        # Every segment's score, kept or not, as a figure.
        >> PlotSelection(selection, output_folder)
        >> Whiten(**stage_kwargs(preset, "whitening"))
        >> Normalize(**stage_kwargs(preset, "normalization"))
        >> Apodize(method="hanning", frac=0.1)
        >> Correlate(method="cross", virtual_source_index=0, part="causal")
        >> Stack(**stage_kwargs(preset, "stacking"))
        >> Save(folder_path=output_folder)
        # The stacked correlations the image is made of, as PAC's gather view draws them.
        >> Plot(folder_path=output_folder)
        >> Dispersion(method="phase", **stage_kwargs(preset, "dispersion"))
        >> Plot(folder_path=output_folder, normalize=True)
        >> Save(folder_path=output_folder)
    )
