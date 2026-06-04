import numpy as np

from src.base.dispersion import DispersionImage


def stack_linear(
    dispersion_images: list[DispersionImage],
    *,
    stack_acquisitions: bool = True,
) -> DispersionImage:
    if not dispersion_images:
        raise ValueError("list cannot be empty.")
    reference = dispersion_images[0]

    for disp in dispersion_images[1:]:
        if disp.fv_map.shape != reference.fv_map.shape:
            raise ValueError("All fv_maps must have the same shape.")

        if not np.allclose(disp.fs, reference.fs):
            raise ValueError("All frequency axes must match.")

        if not np.allclose(disp.vs, reference.vs):
            raise ValueError("All velocity axes must match.")

    fv_stack = np.mean(
        np.stack(
            [disp.fv_map for disp in dispersion_images],
            axis=0,
        ),
        axis=0,
    )

    acquisitions = (
        tuple(
            acquisition
            for disp in dispersion_images
            for acquisition in disp.acquisitions
        )
        if stack_acquisitions
        else reference.acquisitions
    )

    return DispersionImage(
        fv_map=fv_stack,
        fs=reference.fs,
        vs=reference.vs,
        type=reference.type,
        acquisitions=acquisitions,
    )
