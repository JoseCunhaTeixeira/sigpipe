from collections.abc import Callable

from sigproc.base.stream import Stream

from .onebit import normalize_onebit

NORMALIZATION_METHODS: dict[str, Callable[..., Stream]] = {
    "onebit": normalize_onebit,
}
