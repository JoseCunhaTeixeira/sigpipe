from collections.abc import Callable

from sigpipe.base.stream import Stream

from .slice import slice_segments

SEGMENTATION_METHODS: dict[str, Callable[..., list[Stream]]] = {
    "slice": slice_segments,
}
