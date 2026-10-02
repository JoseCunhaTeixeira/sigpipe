"""The windows' images taken from the cache when the same code made them from the same records
with the same settings, byte for byte; made again when any of those differs; the cache kept
within its size, the entries used longest ago removed first."""

import os
from pathlib import Path

from sigpipe.masw.runs import find_run, run_processing
from sigpipe.masw.runs.caching import Cache, image_key, using
from sigpipe.masw.runs.processing import RECORDS_FOLDER
from sigpipe.masw.windows import MASWWindow
from sigpipe.masw.workspace import Folders

WINDOWS = {"masw": {"length": 6, "step": 3}}
IMAGE = "DispersionImage_0000.hdf5"


def _images(run_folder: Path) -> dict[str, bytes]:
    return {path.parent.name: path.read_bytes() for path in run_folder.glob(f"xmid_*/{IMAGE}")}


def test_a_run_made_again_takes_its_images_from_the_cache(
    workspace: Folders, tmp_path: Path
) -> None:
    cache = Cache(tmp_path / "cache", max_bytes=1 << 30)

    with using(cache):
        first = run_processing("shots", "active", WINDOWS, workspace)
        again = run_processing("shots", "active", WINDOWS, workspace)
    other = {"masw": WINDOWS["masw"], "dispersion": {"fmax": 40.0}}
    with using(cache):
        changed = run_processing("shots", "active", other, workspace)
    outside = run_processing("shots", "active", WINDOWS, workspace)

    assert not any(window.cached for window in first.windows)
    assert all(window.cached for window in again.windows) and again.windows
    made = _images(find_run(first.run_id, workspace))
    assert _images(find_run(again.run_id, workspace)) == made
    # Other settings, other images; without a cache, none taken.
    assert not any(window.cached for window in changed.windows)
    assert not any(window.cached for window in outside.windows)
    assert _images(find_run(outside.run_id, workspace)) == made


def test_the_key_holds_the_records_content(workspace: Folders) -> None:
    manifest = run_processing("shots", "active", WINDOWS, workspace)
    run_folder = find_run(manifest.run_id, workspace)
    window = MASWWindow.model_validate_json(
        (run_folder / manifest.windows[0].folder / "window.json").read_text()
    )
    records = run_folder / RECORDS_FOLDER
    key = image_key(manifest.preset, window, records, {})
    stream = records / window.selected_files[0].stem / "Stream_0000.hdf5"

    assert image_key(manifest.preset, window, records, {}) == key
    stream.write_bytes(stream.read_bytes() + b"\0")
    assert image_key(manifest.preset, window, records, {}) != key


def test_the_cache_keeps_to_its_size_the_oldest_used_going_first(tmp_path: Path) -> None:
    source = tmp_path / "made"
    source.mkdir()
    (source / IMAGE).write_bytes(b"x" * 100)
    cache = Cache(tmp_path / "cache", max_bytes=250)
    for key in ("aa1", "bb2", "cc3"):
        cache.keep(key, source, [IMAGE])
    entries = {key: cache.folder / key[:2] / key for key in ("aa1", "bb2", "cc3")}
    for age, key in enumerate(("bb2", "aa1", "cc3")):
        os.utime(entries[key], (1_000 + age, 1_000 + age))

    cache.trim()

    assert [key for key, entry in entries.items() if entry.exists()] == ["aa1", "cc3"]
    into = tmp_path / "into"
    into.mkdir()
    assert cache.take("cc3", into) and (into / IMAGE).read_bytes() == b"x" * 100
    # An entry trimmed is not there to take, and the folder keeps what it held.
    assert not cache.take("bb2", into)
    assert [path.name for path in into.iterdir()] == [IMAGE]
