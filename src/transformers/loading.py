from collections.abc import Sequence
from pathlib import Path

from src.base.transformer import Transformer
from src.dataio.registry import LOAD_HANDLERS


class Load(Transformer):
    """
    Saving transformer.
    """

    def __init__(
        self,
        file_paths: list[Path],
        data_type: type | str,
        **params,
    ):
        self.file_paths = file_paths
        self.data_type = data_type
        self.params = params

    def transform(self, data=None):

        handler = LOAD_HANDLERS.get(self.data_type)

        if handler is None:
            raise TypeError(
                f"No load handler for {self.data_type.__name__ if isinstance(self.data_type, type) else self.data_type}"
            )

        objects = []
        for file_path in self.file_paths:
            obj = handler(
                path=file_path,
                **self.params,
            )
            if isinstance(obj, Sequence):
                objects.extend(obj)
            else:
                objects.append(obj)

        return objects
