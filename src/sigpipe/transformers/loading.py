from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from sigpipe.base.beamforming import Beam
from sigpipe.base.dispersion_curve import DispersionCurves
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.base.stream import Stream
from sigpipe.base.transformer import Transformer
from sigpipe.base.velocity_model import VelocityModels
from sigpipe.dataio.registry import LOAD_HANDLERS


class Load(
    Transformer[
        object,
        Stream | DispersionImage | DispersionCurves | VelocityModels | Beam,
    ]
):
    """
    Loading transformer.
    """

    def __init__(
        self,
        file_paths: Sequence[Path],
        data_type: Literal[
            "stream",
            "dispersion_image",
            "dispersion_curves",
            "seismic",
            "gero_active",
            "gero_passive",
            "velocity_models",
            "beam",
        ],
        **params: object,
    ) -> None:
        if not isinstance(file_paths, Sequence) or isinstance(file_paths, (str, bytes)):
            raise TypeError(f"Expected Sequence for file_paths, got {type(file_paths).__name__}")

        if len(file_paths) == 0:
            raise ValueError("file_paths is empty")

        if not all(isinstance(s, Path) for s in file_paths):
            raise TypeError("All elements in file_paths must be Path")

        self.file_paths = file_paths
        self.data_type = data_type.lower()
        self.params = params

    def transform(
        self,
        data: object = None,  # noqa: ARG002
    ) -> (
        list[Stream]
        | list[DispersionImage]
        | list[DispersionCurves]
        | list[VelocityModels]
        | list[Beam]
    ):

        handler = LOAD_HANDLERS.get(self.data_type)

        if handler is None:
            raise TypeError(f"No load handler for {self.data_type!r}")

        return handler(
            file_paths=self.file_paths,
            **self.params,
        )
