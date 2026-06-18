from collections.abc import Callable

from sigproc.base.stream import Stream

from .cross import correlate_cross

CORRELATION_METHODS: dict[str, Callable[..., tuple[Stream] | tuple[Stream, Stream]]] = {
    "cross": correlate_cross,
}
