from collections.abc import Sequence
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from sigpipe.base.transformer import Transformer
from sigpipe.dataio.plot_config import SAVING_DPI
from sigpipe.dataio.registry import SECTION_HANDLERS, resolve_handler


class PlotSection[T](Transformer[T, T]):
    """
    Section plotting transformer.
    """

    def __init__(
        self,
        folder_path: Path,
        **params: object,
    ) -> None:
        self.folder_path = folder_path
        self.params = params

    def transform(self, data: Sequence[T]) -> Sequence[T]:

        self.validate_homogeneous_sequence(data)

        self.folder_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        first = data[0]
        handler = resolve_handler(SECTION_HANDLERS, first)
        if handler is None:
            raise TypeError(f"No section-plot handler for {type(first).__name__}")

        for i, obj in enumerate(data):
            figure = handler(
                obj,
                **self.params,
            )
            self.savefig(
                path=self.folder_path / f"{type(obj).__name__}_{i:04d}.png",
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
