from collections.abc import Sequence
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from sigpipe.base.stream import Stream
from sigpipe.base.transformer import Transformer
from sigpipe.dataio.plot_config import SAVING_DPI
from sigpipe.dataio.registry import PLOT_HANDLERS, resolve_handler
from sigpipe.dataio.signal_plotting import save_spectra


class Plot[T](Transformer[T, T]):
    """
    Plotting transformer.
    """

    def __init__(
        self,
        folder_path: Path,
        file_name: str = "",
        **params: object,
    ) -> None:
        self.folder_path = folder_path
        self.file_name = file_name
        self.params = params

    def transform(self, data: Sequence[T]) -> Sequence[T]:

        self.validate_homogeneous_sequence(data)

        self.folder_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        first = data[0]
        handler = resolve_handler(PLOT_HANDLERS, first)
        if handler is None:
            raise TypeError(f"No plot handler for {type(first).__name__}")

        for i, obj in enumerate(data):
            figure = handler(
                obj,
                **self.params,
            )
            file_name = (
                f"{self.file_name}_{i:04d}.png"
                if self.file_name
                else f"{type(obj).__name__}_{i:04d}.png"
            )
            self.savefig(
                path=self.folder_path / file_name,
                figure=figure,
            )
            plt.close(figure)
        return data

    @staticmethod
    def savefig(
        path: Path,
        figure: Figure,
    ) -> None:
        figure.savefig(
            path,
            bbox_inches="tight",
            dpi=SAVING_DPI,
        )


class PlotSpectra(Transformer[Stream, Stream]):
    """Each stream's spectra, their figure and their data (save_spectra), in `folder_path`: a
    window's stacked correlations, as Visualization draws them."""

    def __init__(self, folder_path: Path) -> None:
        self.folder_path = folder_path

    def transform(self, data: Sequence[Stream]) -> Sequence[Stream]:
        self.validate_sequence(data, Stream)
        for i, stream in enumerate(data):
            save_spectra(stream, self.folder_path, index=i)
        return data
