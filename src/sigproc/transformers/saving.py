from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from sigproc.base.transformer import Transformer
from sigproc.dataio.registry import SAVE_HANDLERS

T = TypeVar("T")


class Save(Transformer):
    """
    Saving transformer.
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

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence, got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(x, type(data[0])) for x in data):
            raise TypeError("All elements must have the same type")

        self.folder_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        first = data[0]
        handler = SAVE_HANDLERS.get(type(first))
        if handler is None:
            raise TypeError(f"No save handler for {type(first).__name__}")

        for i, obj in enumerate(data):
            file_name = (
                f"{self.file_name}_{i:04d}" if self.file_name else f"{type(obj).__name__}_{i:04d}"
            )
            handler(
                obj,
                path=self.folder_path / file_name,
                **self.params,
            )

        return data
