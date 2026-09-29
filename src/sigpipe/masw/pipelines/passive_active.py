"""The passive-active pipeline, from the preprocessed records: interferometry on the window's
shots (each gather cross-correlated with the receiver nearest its shot, and flipped when the shot
is past the window's far end), the correlations stacked, then a dispersion image. To correlate
the surface waves alone, the preprocessing's muting keeps them (its velocities)."""

from collections.abc import Mapping
from pathlib import Path

from sigpipe.base import Pipeline
from sigpipe.base.stream import Stream
from sigpipe.masw.pipelines.common import load_preprocessed, stage_kwargs
from sigpipe.masw.presets import PassiveActivePreset
from sigpipe.masw.windows import MASWWindow
from sigpipe.transformers import (
    ActiveShotCorrelation,
    Apodize,
    Dispersion,
    Plot,
    PlotSpectra,
    Save,
    Stack,
)


def build_passive_active_pipeline(
    preset: PassiveActivePreset, window: MASWWindow, records_folder: Path, output_folder: Path
) -> Pipeline:
    return (
        load_preprocessed(window, records_folder)
        >> Apodize(method="hanning", frac=0.1)
        >> ActiveShotCorrelation(method="cross")
        >> Stack(**stage_kwargs(preset, "stacking"))
        >> Save(folder_path=output_folder)
        # The stacked correlations the image is made of, as PAC's gather view draws them, and
        # their spectra.
        >> Plot(folder_path=output_folder)
        >> PlotSpectra(folder_path=output_folder)
        >> Dispersion(method="phase", **stage_kwargs(preset, "dispersion"))
        >> Plot(folder_path=output_folder, normalize=True)
        >> Save(folder_path=output_folder)
    )


def window_records(window: MASWWindow, streams: Mapping[str, Stream]) -> list[Stream]:
    """The window's records from `streams` (each record's, by its file's name; its records
    before their muting, say), cut to the window's receivers as load_preprocessed cuts the saved
    ones."""
    records: list[Stream] = []
    for i, path in enumerate(window.selected_files):
        stream = streams[path.name]
        indices = window.record_receivers[i] if window.record_receivers else window.receiver_indices
        records.append(
            Stream(
                xt=stream.xt[indices, :],
                ts=stream.ts,
                sampling_freq=stream.sampling_freq,
                acquisition=window.acquisitions[i],
            )
        )
    return records


def window_correlations(
    preset: PassiveActivePreset, window: MASWWindow, streams: Mapping[str, Stream]
) -> Stream:
    """The window's stacked correlations made from `streams` as its pipeline makes them from
    the saved records: what G2 measures a muted line's correlations on, its records before their
    muting (a muting zeroes their noise, and the correlations' after the slowest arrival)."""
    records = Apodize(method="hanning", frac=0.1).transform(window_records(window, streams))
    correlated = ActiveShotCorrelation(method="cross").transform(records)
    return Stack(**stage_kwargs(preset, "stacking")).transform(correlated)[0]
