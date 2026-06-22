from collections.abc import Sequence
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from sigproc.base.transformer import Transformer
from sigproc.dataio.plot_config import SAVING_DPI
from sigproc.dataio.registry import PLOT_HANDLERS, resolve_handler


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
