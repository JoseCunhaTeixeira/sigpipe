"""Text that people write by hand, read whatever editor saved it."""

from pathlib import Path


def read_hand_written(path: Path) -> str:
    """The text of `path`, a file a person wrote: UTF-8, with or without the byte-order mark some
    Windows editors put first, or else a Windows code page (cp1252), as editors there may save."""
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")
