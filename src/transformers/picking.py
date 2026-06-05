from collections.abc import Sequence

from src.algorithms.picking.picking import pick
from src.base.dispersion import DispersionCurves, DispersionImage
from src.base.transformer import Transformer


class Pick(Transformer):
    """
    Picking transformer.
    """

    def __init__(
        self,
        return_image: bool = False,
        **params,
    ):
        self.return_image = return_image
        self.params = params

    def transform(
        self, data: Sequence[DispersionImage]
    ) -> Sequence[DispersionImage] | Sequence[DispersionCurves]:

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(
                f"Expected Sequence[DispersionImage], got {type(data).__name__}"
            )

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(s, DispersionImage) for s in data):
            raise TypeError("All elements must be DispersionImage")

        if self.return_image:
            dispersion_images_out = []
            for dispersion_image in data:
                dispersion_curves = pick(
                    dispersion_image,
                    **self.params,
                )
                curves = list(dispersion_image.dispersion_curves)
                curves.extend(dispersion_curves)
                dispersion_image_out = DispersionImage(
                    fv_map=dispersion_image.fv_map,
                    fs=dispersion_image.fs,
                    vs=dispersion_image.vs,
                    type=dispersion_image.type,
                    acquisitions=dispersion_image.acquisitions,
                    dispersion_curves=DispersionCurves(
                        curves=tuple(curves),
                    ),
                )
                dispersion_images_out.append(dispersion_image_out)
            return dispersion_images_out

        dispersion_curves_out = []
        for dispersion_image in data:
            dispersion_curves = pick(
                dispersion_image,
                **self.params,
            )
            dispersion_curves_out.append(dispersion_curves)
        return dispersion_curves_out
