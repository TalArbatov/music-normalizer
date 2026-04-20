"""Tag write abstraction per audio format.

Verifies that write_title_artist dispatches to a handler that accepts the
'title'/'artist' keys for every format we claim to support. We don't do a full
disk round-trip for every format — FLAC is covered by the smoke test; the
others are validated via a fake that records the calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from music_normalizer.tags import (
    WRITABLE_EXTENSIONS,
    write_title_artist,
)


class _FakeHandler(dict):
    """Dict-based handler matching the minimal surface write_title_artist uses."""

    def __init__(self) -> None:
        super().__init__()
        self.saved = False

    def save(self) -> None:
        self.saved = True


def test_writable_extensions_set():
    # WAV must be excluded by design.
    assert ".flac" in WRITABLE_EXTENSIONS
    assert ".mp3" in WRITABLE_EXTENSIONS
    assert ".m4a" in WRITABLE_EXTENSIONS
    assert ".mp4" in WRITABLE_EXTENSIONS
    assert ".ogg" in WRITABLE_EXTENSIONS
    assert ".opus" in WRITABLE_EXTENSIONS
    assert ".wav" not in WRITABLE_EXTENSIONS


def test_wav_raises(tmp_path):
    with pytest.raises(ValueError, match="unsupported format"):
        write_title_artist(tmp_path / "x.wav", title="T", artist="A", dry_run=False)


def test_unknown_extension_raises(tmp_path):
    with pytest.raises(ValueError, match="unsupported format"):
        write_title_artist(tmp_path / "x.xyz", title="T", artist="A", dry_run=False)


def test_empty_values_refused(tmp_path):
    with pytest.raises(ValueError, match="empty title"):
        write_title_artist(tmp_path / "x.flac", title="", artist=None, dry_run=True)
    with pytest.raises(ValueError, match="empty artist"):
        write_title_artist(tmp_path / "x.flac", title=None, artist="", dry_run=True)


def test_dry_run_noop(tmp_path):
    # dry_run must never try to open a file that doesn't exist.
    write_title_artist(
        tmp_path / "missing.flac", title="T", artist="A", dry_run=True
    )


@pytest.mark.parametrize("ext", [".flac", ".mp3", ".m4a", ".mp4", ".ogg", ".opus"])
def test_handler_dispatch_per_extension(ext, monkeypatch, tmp_path):
    """Every supported extension dispatches to a handler that accepts title/artist."""
    captured: dict[str, _FakeHandler] = {}

    def fake_open(path: Path, e: str) -> _FakeHandler:
        h = _FakeHandler()
        captured["handler"] = h
        captured["ext"] = e
        return h

    monkeypatch.setattr("music_normalizer.tags._open_for_write", fake_open)

    target = tmp_path / f"file{ext}"
    target.touch()
    write_title_artist(target, title="New Title", artist="New Artist", dry_run=False)

    h = captured["handler"]
    assert captured["ext"] == ext
    assert h["title"] == ["New Title"]
    assert h["artist"] == ["New Artist"]
    assert h.saved is True
