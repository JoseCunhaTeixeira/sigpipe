import warnings

import numpy as np

from sigpipe.base.dispersion_image import DispersionImage

_LOG_EPS = 1e-12  # inside log(p + eps), just avoids log(0) for an exactly-zero bin
_REL_ENERGY_EPS = 1e-9  # column considered "empty" below this fraction of the image's own max


def stack_ridge_entropy(
    dispersion_images: list[DispersionImage],
    *,
    alpha: float = 5.0,
) -> tuple[DispersionImage, np.ndarray, np.ndarray]:
    r"""Stack `fv_map`s with weights favoring images whose energy, at each
    frequency, sits in a sharp, well-defined ridge over velocity rather
    than smeared broadly across it -- a proxy for "how confidently does
    this image see a dispersion curve" that doesn't need to know the
    curve itself. Companion to `stack_linear`/`stack_root` in this same
    package: those combine fv_maps with a fixed formula, this estimates a
    per-image quality first and stacks with weights favoring the better
    ones.

    Per image, per frequency column f (after clipping negatives to 0):

        p(v|f) = I(v,f) / sum_v I(v,f)
        H(f)   = -sum_v p(v|f) * log(p(v|f) + eps)
        H_norm(f) = H(f) / log(Nv)                    (Nv = len(vs))

    A column with no energy at all has no defined p(v|f) (0/0) and is
    *excluded* from the per-image average rather than folded in at
    maximum entropy -- see "Deviations from the literal spec" below.
    The per-image ridge entropy is the mean of H_norm(f) over the
    remaining (non-empty) columns:

        H_ridge = mean_{f: column f non-empty} H_norm(f)

    Weights are then a softmax-like function of H_ridge -- see
    "Deviations" for why this isn't literally `exp(-alpha * H_ridge)` --
    normalized to sum to 1, and the stack is their weighted sum of the
    *original* (unclipped) fv_maps:

        stack = sum_i w_i * fv_map_i

    Returns `(stacked_image, weights, ridge_entropies)`: `weights` and
    `ridge_entropies` are 1D arrays parallel to `dispersion_images`, for
    diagnostics (which images the stack effectively leaned on, and why).
    `stack_ridge_entropy_image` below discards these two and returns just
    the `DispersionImage`, for use through the `method="entropy"` /
    `Stack` transformer registry, which -- like `stack_linear`/
    `stack_root` -- expects a single `DispersionImage` return.

    Deviations from the literal ridge-entropy-weighting spec, and why
    ----------------------------------------------------------------
    1. Empty frequency columns are *excluded* from H_ridge's average,
       not forced to H_norm=1 (maximum entropy). A column can be
       genuinely empty just because it sits outside the excitation
       bandwidth of a particular acquisition -- that says nothing about
       how *sharp* the image's real ridge is within the band it does
       have energy in. Forcing empty columns to maximum entropy would
       penalize an image for having a narrower excited band than another
       image in the same stack, which has nothing to do with the ridge
       quality this is meant to weight. "Empty" is judged relative to the
       image's own peak amplitude (`_REL_ENERGY_EPS`), not an absolute
       threshold, since `fv_map` amplitude scale isn't fixed across
       acquisitions.

    2. Weights are computed from H_ridge *standardized against the
       batch being stacked* -- `z_i = (H_ridge_i - median) / (MAD + eps)`
       (median/MAD, not mean/std, for robustness to one unusually
       good-or-bad image skewing the reference scale) -- then
       `w_i = exp(-alpha * z_i)`, instead of `exp(-alpha * H_ridge_i)` on
       the raw, un-standardized value. The raw version makes `alpha`'s
       effect depend on where H_norm's absolute values happen to fall for
       a particular rig/geometry/noise floor -- the same alpha could be
       nearly a no-op for one dataset and nearly winner-take-all for
       another, with no way to tell from alpha alone. Standardizing first
       makes alpha control *relative* separation between better- and
       worse-than-typical images in the current batch specifically, which
       is the only thing it can be sensibly tuned against without
       re-deriving a good alpha per dataset.

    Critical review -- biases and weaknesses of ridge entropy itself
    -----------------------------------------------------------------
    - Multi-mode bias: a frequency where two guided-wave modes (e.g. S0
      and A0) both carry real energy has its energy legitimately split
      across two velocities, which *raises* H(f) relative to a
      single-mode column -- even though the multi-mode image may be the
      more physically complete one. Ridge entropy alone can't distinguish
      "confidently split between two real modes" from "confused/noisy,"
      and will systematically down-weight images that resolve more modes.
    - Spike sensitivity: Shannon entropy of a *single* dominant bin is 0
      regardless of whether that bin is a genuine, well-supported ridge
      or an isolated single-pixel artifact (e.g. a sensor glitch). A
      one-bin spike looks maximally "sharp" and gets the highest possible
      weight, which is exactly backwards if it's spurious. A real ridge,
      by contrast, is smeared across a few adjacent velocity bins by
      finite frequency/velocity resolution -- a numerical improvement
      worth considering (not implemented here, to avoid adding a tunable
      nobody asked for) is smoothing each column slightly along velocity
      before computing p(v|f), so an isolated spike registers as *less*
      confident than a resolution-consistent ridge, not more.
    - Batch-size sensitivity: median/MAD standardization is well-defined
      but uninformative for very few images (at n=2, MAD is just half
      the gap between the two H_ridge values, so z is always +/-1
      regardless of how similar or different they actually are) --
      quality weighting is most meaningful with enough images that
      "typical" is a real distribution, not a description of 2-3 points.
    """
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

    n_v = reference.fv_map.shape[1]
    maps = np.stack(
        [np.clip(np.asarray(disp.fv_map, dtype=np.float64), 0, None) for disp in dispersion_images],
        axis=0,
    )  # (K, Nf, Nv)

    img_max = maps.max(axis=(1, 2), keepdims=True)  # (K, 1, 1)
    col_sum = maps.sum(axis=2, keepdims=True)  # (K, Nf, 1)
    valid = col_sum[..., 0] > (_REL_ENERGY_EPS * img_max[..., 0])  # (K, Nf)

    p = np.divide(maps, col_sum, out=np.zeros_like(maps), where=col_sum > 0)
    h = -np.sum(p * np.log(p + _LOG_EPS), axis=2)  # (K, Nf)
    h_norm = h / np.log(n_v)

    valid_counts = valid.sum(axis=1)  # (K,)
    h_norm_masked = np.where(valid, h_norm, 0.0)
    ridge_entropies = np.divide(
        h_norm_masked.sum(axis=1),
        valid_counts,
        out=np.ones(len(dispersion_images)),  # no energy anywhere: treat as maximally uncertain
        where=valid_counts > 0,
    )

    median = np.median(ridge_entropies)
    mad = np.median(np.abs(ridge_entropies - median))
    z = (ridge_entropies - median) / (mad + _LOG_EPS)
    raw_weights = np.exp(-alpha * z)
    weights = raw_weights / raw_weights.sum()

    fv_stack = np.tensordot(
        weights,
        np.stack([np.asarray(disp.fv_map, dtype=np.float64) for disp in dispersion_images], axis=0),
        axes=(0, 0),
    )

    acquisition = type(reference.acquisition)(
        source=reference.acquisition.receivers[0], receivers=reference.acquisition.receivers
    )

    stacked = DispersionImage(
        fv_map=fv_stack,
        fs=reference.fs,
        vs=reference.vs,
        type=reference.type,
        acquisition=acquisition,
    )
    return stacked, weights, ridge_entropies


def stack_ridge_entropy_image(
    dispersion_images: list[DispersionImage],
    *,
    alpha: float = 5.0,
) -> DispersionImage:
    """`stack_ridge_entropy`, discarding its weights/ridge-entropies --
    the signature the `method="entropy"` `Stack` transformer registry
    entry needs (matching `stack_linear`/`stack_root`'s single-return
    signature). Call `stack_ridge_entropy` directly for the diagnostics.
    """
    stacked, _, _ = stack_ridge_entropy(dispersion_images, alpha=alpha)
    return stacked
