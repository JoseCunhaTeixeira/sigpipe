"""PAC's passive-active pipeline, from the preprocessed records: interferometry on the window's
shots (each gather cut to its surface-wave window, PACo's addition, then cross-correlated with
the receiver nearest its shot, and flipped when the shot is past the window's far end), the
correlations stacked, then a dispersion image."""

from pathlib import Path

from sigpipe.base import Pipeline
from sigpipe.masw.pipelines.common import load_preprocessed, stage_kwargs
from sigpipe.masw.presets import PassiveActivePreset
from sigpipe.masw.windows import MASWWindow
from sigpipe.transformers import ActiveShotCorrelation, Apodize, Dispersion, Mute, Plot, Save, Stack


def build_passive_active_pipeline(
    preset: PassiveActivePreset, window: MASWWindow, records_folder: Path, output_folder: Path
) -> Pipeline:
    return (
        load_preprocessed(window, records_folder)
        >> Mute(**stage_kwargs(preset, "correlation_window"))
        >> Apodize(method="hanning", frac=0.1)
        >> ActiveShotCorrelation(method="cross")
        >> Stack(**stage_kwargs(preset, "stacking"))
        >> Plot(folder_path=output_folder)
        >> Save(folder_path=output_folder)
        >> Dispersion(method="phase", **stage_kwargs(preset, "dispersion"))
        >> Plot(folder_path=output_folder, normalize=True)
        >> Save(folder_path=output_folder)
    )
