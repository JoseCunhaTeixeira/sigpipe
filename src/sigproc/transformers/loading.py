from collections.abc import Sequence
from pathlib import Path

from typing_extensions import Literal

from sigproc.base.transformer import Transformer
from sigproc.dataio.registry import LOAD_HANDLERS


class Load(Transformer):
    """
    Saving transformer.
    """

    def __init__(
        self,
        file_paths: Sequence[Path],
        data_type: Literal[
            "stream",
            "dispersion_image",
            "dispersion_curves",
            "segy",
            "gero_active",
            "gero_passive",
        ],
        **params,
    ):
        if not isinstance(file_paths, Sequence) or isinstance(file_paths, (str, bytes)):
            raise TypeError(
                f"Expected Sequence for file_paths, got {type(file_paths).__name__}"
            )

        if len(file_paths) == 0:
            raise ValueError("file_paths is empty")

        if not all(isinstance(s, Path) for s in file_paths):
            raise TypeError("All elements in file_paths must be Path")

        self.file_paths = file_paths
        self.data_type = data_type.lower()
        self.params = params

    def transform(self, data=None):

        handler = LOAD_HANDLERS.get(self.data_type)

        if handler is None:
            raise TypeError(
                f"No load handler for {self.data_type.__name__ if isinstance(self.data_type, type) else self.data_type}"
            )

        return handler(
            path=self.file_paths,
            **self.params,
        )
