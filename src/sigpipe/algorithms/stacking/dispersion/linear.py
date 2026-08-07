import warnings

import numpy as np

from sigpipe.base.dispersion_image import DispersionImage


def stack_linear(
    dispersion_images: list[DispersionImage],
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

        if disp.acquisition.receivers != reference.acquisition.receivers:
            warnings.warn(
                "Dispersion images have different acquisitions. "
                "Resulting image will only include acquisition from the first image "
                "with first receiver as source.",
                UserWarning,
                stacklevel=2,
            )

    fv_stack = np.mean(
        np.stack(
            [disp.fv_map for disp in dispersion_images],
            axis=0,
        ),
        axis=0,
    )

    acquisition = type(reference.acquisition)(
        source=reference.acquisition.receivers[0], receivers=reference.acquisition.receivers
    )

    return DispersionImage(
        fv_map=fv_stack,
        fs=reference.fs,
        vs=reference.vs,
        type=reference.type,
        acquisition=acquisition,
    )
