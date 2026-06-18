from collections.abc import Callable

from sigproc.base.stream import Stream

from .iir import filter_iir

FILTERING_METHODS: dict[str, Callable[..., Stream]] = {
    "iir": filter_iir,
}
