from collections.abc import Sequence
from pathlib import Path

from sigpipe.base.transformer import Transformer
from sigpipe.dataio.registry import SAVE_HANDLERS, resolve_handler


class Save[T](Transformer[T, T]):
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

        self.validate_homogeneous_sequence(data)

        self.folder_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        first = data[0]
        handler = resolve_handler(SAVE_HANDLERS, first)
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
