from collections.abc import Callable

from sigproc.base.stream import Stream

from .hanning import apodize_hanning

APODIZATION_METHODS: dict[str, Callable[..., Stream]] = {
    "hanning": apodize_hanning,
}
