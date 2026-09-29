from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt

from sigpipe.algorithms.selection.registry import STREAM_SELECTION_METHODS
from sigpipe.algorithms.selection.stream.fk import fk_ratio, select_by_ratio
from sigpipe.base.stream import Stream
from sigpipe.base.transformer import Transformer
from sigpipe.dataio.selection_plotting import plot_selection
from sigpipe.transformers.plotting import Plot


class Selection(Transformer[Stream, Stream]):
    """
    Stream selection transformer. With the fk method, each stream's score is kept, in the
    order met (`scores`: its fk_ratio, and whether it was kept), for PlotSelection.
    """

    def __init__(
        self,
        method: Literal["none", "fk"],
        **params: object,
    ) -> None:
        self.method = method
        self.params = params
        self.scores: list[tuple[float, bool]] = []

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        self.validate_sequence(data, Stream)

        if self.method == "none":
            return list(data)

        algorithm = STREAM_SELECTION_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown selection method '{self.method}'. "
                f"Available methods: {list(STREAM_SELECTION_METHODS.keys())}"
            )

        first = data[0]

        if isinstance(first, Stream):
            streams_out: list[Stream] = []
            for stream in data:
                stream_out = (
                    self._select(stream)
                    if self.method == "fk"
                    else algorithm(
                        stream=stream,
                        **self.params,
                    )
                )
                if stream_out is not None:
                    streams_out.append(stream_out)
            if not streams_out and self.method == "fk":
                # Said here: the steps after it would only find nothing to work on.
                threshold = self.params.get("threshold")
                raise ValueError(
                    f"The fk selection kept none of the {len(data)} segments (|f-k ratio| > "
                    f"{threshold}): lower its threshold"
                )
            return streams_out

        raise TypeError(f"No selection handler for {type(first).__name__}")

    def _select(self, stream: Stream) -> Stream | None:
        """The fk method on `stream`, its score kept."""
        params: dict[str, Any] = dict(self.params)
        ratio = fk_ratio(stream, params.get("vmin"), params.get("vmax"))
        kept = select_by_ratio(
            stream, ratio, float(params["threshold"]), bool(params.get("flip_negatives", False))
        )
        self.scores.append((ratio, kept is not None))
        return kept


class PlotSelection(Transformer[Stream, Stream]):
    """Pass the streams on, and save `selection`'s scores as a figure in `folder_path`
    (`file_name`_0000.png): each segment's fk_ratio, kept or not, around its threshold. Nothing
    without scores (the method none)."""

    def __init__(
        self, selection: Selection, folder_path: Path, file_name: str = "Selection"
    ) -> None:
        self.selection = selection
        self.folder_path = folder_path
        self.file_name = file_name

    def transform(self, data: Sequence[Stream]) -> list[Stream]:
        if self.selection.scores:
            threshold = float(self.selection.params.get("threshold", 0.0))  # pyright: ignore[reportArgumentType]
            figure = plot_selection(self.selection.scores, threshold)
            self.folder_path.mkdir(parents=True, exist_ok=True)
            Plot.savefig(path=self.folder_path / f"{self.file_name}_0000.png", figure=figure)
            plt.close(figure)
        return list(data)
