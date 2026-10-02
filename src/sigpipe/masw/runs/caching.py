"""A cache of the windows' images (S8 of PACo's agent guidelines): a window's image costs seconds
to make, and a line makes the same ones again (the trials of its windows' length and muting,
the same profile run again, a comparison of settings).

Each image is kept under a key made of all that decides it: the window, the preset, the content
of the preprocessed records it reads, and the code's version (sigpipe's sources and the versions
of the libraries it computes with). An image is taken from the cache only when the same code
made it from the same records with the same settings: the same files, byte for byte. The
records are not kept: made again in a fraction of a second, they weigh ten times an image.

The cache is used where `CACHE` holds one (`using`): its caller's choice, none by default. It
holds at most its `max_bytes`, the entries used longest ago removed first."""

import hashlib
import json
import os
import secrets
import shutil
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import sigpipe
from sigpipe.masw.pipelines.common import PREPROCESSED
from sigpipe.masw.presets import ActivePreset, PassivePreset
from sigpipe.masw.windows import MASWWindow

# The libraries an image is computed and drawn with, besides sigpipe.
LIBRARIES = ("numpy", "scipy", "h5py", "matplotlib", "obspy")
# The format of the keys: a change of what goes into them changes it.
KEY_VERSION = 1


@dataclass(frozen=True)
class Cache:
    folder: Path
    max_bytes: int

    def take(self, key: str, into: Path) -> bool:
        """The files kept under `key` copied into folder `into`; False, nothing copied, when
        none are kept."""
        entry = self._entry(key)
        copied: list[Path] = []
        try:
            for path in entry.iterdir():
                copied.append(into / path.name)
                shutil.copyfile(path, copied[-1])
            os.utime(entry)  # used now: the last to go
        except FileNotFoundError:  # never kept, or removed meanwhile by another trim
            for path in copied:
                path.unlink(missing_ok=True)
            return False
        return True

    def keep(self, key: str, folder: Path, names: Iterable[str]) -> None:
        """Files `names` of `folder` kept under `key`; kept already (by another process), left
        as they are."""
        entry = self._entry(key)
        if entry.exists():
            return
        partial = entry.parent / f".{key}.{secrets.token_hex(4)}"
        partial.mkdir(parents=True)
        for name in names:
            shutil.copyfile(folder / name, partial / name)
        try:
            partial.rename(entry)
        except OSError:
            shutil.rmtree(partial, ignore_errors=True)
            if not entry.exists():
                raise
            # Kept meanwhile by another process: the same files.

    def trim(self) -> None:
        """The entries used longest ago removed, until the cache holds at most `max_bytes`."""
        entries = [
            (entry.stat().st_mtime, _size(entry), entry)
            for entry in self.folder.glob("*/*")
            if entry.is_dir() and not entry.name.startswith(".")
        ]
        total = sum(size for _, size, _ in entries)
        for _, size, entry in sorted(entries):
            if total <= self.max_bytes:
                break
            shutil.rmtree(entry, ignore_errors=True)
            total -= size

    def _entry(self, key: str) -> Path:
        return self.folder / key[:2] / key


CACHE: ContextVar[Cache | None] = ContextVar("CACHE", default=None)


@contextmanager
def using(cache: Cache | None) -> Generator[None]:
    """The processing within the block takes and keeps its images in `cache` (None: none)."""
    token = CACHE.set(cache)
    try:
        yield
    finally:
        CACHE.reset(token)


def image_key(
    preset: ActivePreset | PassivePreset,
    window: MASWWindow,
    records_folder: Path,
    hashes: dict[Path, str],
) -> str:
    """The key of `window`'s image made with `preset` on the preprocessed records of
    `records_folder`; `hashes`, the records' hashes by path, those hashed already (a record read
    by several windows hashed once), the others added."""
    streams: dict[str, str] = {}
    for path in window.selected_files:
        stream = records_folder / path.stem / PREPROCESSED
        if stream not in hashes:
            hashes[stream] = file_hash(stream)
        streams[path.stem] = hashes[stream]
    parts = {
        "version": KEY_VERSION,
        "code": code_version(),
        "kind": type(preset).__name__,
        "preset": preset.model_dump(mode="json"),
        "window": window.model_dump(mode="json"),
        "records": streams,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


@cache
def code_version() -> str:
    """sigpipe's sources and the versions of the libraries it computes with, hashed: any change
    of the code makes other keys."""
    digest = hashlib.sha256()
    root = Path(sigpipe.__file__).parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    for name in LIBRARIES:
        try:
            digest.update(f"{name}=={version(name)}".encode())
        except PackageNotFoundError:
            digest.update(f"{name} absent".encode())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    """The SHA-256 of `path`'s content."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _size(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.iterdir() if path.is_file())
