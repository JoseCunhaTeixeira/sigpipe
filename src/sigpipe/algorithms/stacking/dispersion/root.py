import warnings

import numpy as np

from sigpipe.base.dispersion_image import DispersionImage


def stack_root(
    dispersion_images: list[DispersionImage],
    *,
    n: int = 2,
) -> DispersionImage:
    """Nth-root stack of dispersion images' fv_maps: each is raised to the
    power 1/n, averaged, then the average is raised back to the nth power.

    This is the same family as `stream.root.stack_root` (Schimmel and
    Paulssen, 1997), but *not* the same mechanism: that version signs the
    root (`sign(xt) * |xt|**(1/n)`) because its noise-suppression relies
    on randomly-signed noise partially cancelling under averaging, which
    only makes sense for a signed time-domain waveform. `fv_map` is a
    non-negative amplitude/energy map (no sign to exploit), so raising it
    straight to a fractional power instead pulls the average toward a
    power/geometric mean of the stacked values, which is much more
    sensitive to a value being *inconsistently* present than the
    arithmetic mean is. In practice that suppresses whatever energy only
    shows up in a minority of the stacked images -- incoherent noise, or
    a shot-specific artifact -- more than it suppresses energy that's
    consistently present across (most of) them, i.e. the true dispersion
    ridge shared by every shot.

    The corollary: this assumes the stacked images are genuinely
    independent looks at the *same* underlying dispersion characteristic.
    If some contributing images have a real but weak version of the
    signal (e.g. a low-SNR shot, not just noise), a high `n` will
    attenuate that real signal along with the noise -- `n` should be
    picked with the actual spread of input quality in mind, not maxed
    out by default.
    """
    if n < 1:
        raise ValueError(f"n ({n}) must be >= 1")
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

    powered_mean = np.mean(
        np.stack(
            [disp.fv_map ** (1.0 / n) for disp in dispersion_images],
            axis=0,
        ),
        axis=0,
    )
    fv_stack = powered_mean**n

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
