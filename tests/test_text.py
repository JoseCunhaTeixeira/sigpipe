"""Text written by hand: read as its editor saved it, UTF-8 with or without a byte-order mark, or a
Windows code page."""

from pathlib import Path

from sigpipe.text import read_hand_written


def test_utf8_is_read_with_or_without_a_byte_order_mark(tmp_path: Path) -> None:
    text = "# récepteurs, λ = 12 m\n- {x: 0.0, z: 0.0}\n"
    plain, marked = tmp_path / "plain.yaml", tmp_path / "marked.yaml"
    plain.write_bytes(text.encode("utf-8"))
    marked.write_bytes(text.encode("utf-8-sig"))

    assert read_hand_written(plain) == read_hand_written(marked) == text


def test_a_windows_code_page_is_read_where_the_text_is_not_utf8(tmp_path: Path) -> None:
    text = "# récepteurs à 1,5 m\n- {x: 0.0, z: 0.0}\n"
    path = tmp_path / "windows.yaml"
    path.write_bytes(text.encode("cp1252"))

    assert read_hand_written(path) == text
