"""Mutagen-backed tag read/write.

We never touch anything except TITLE and ARTIST. All other tags are preserved
exactly as found on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mutagen import File as MutagenFile
from mutagen.easymp4 import EasyMP4
from mutagen.flac import FLAC
from mutagen.id3 import ID3NoHeaderError
from mutagen.mp3 import EasyMP3
from mutagen.oggopus import OggOpus
from mutagen.oggvorbis import OggVorbis

from .utils import get_logger

log = get_logger(__name__)

# Formats we can safely write TITLE and ARTIST to using mutagen's easy/Vorbis
# interfaces. WAV is deliberately excluded: mutagen's WAV support does not expose
# a consistent title/artist surface, and many players ignore ID3-in-WAV anyway.
# WAV files are still scanned and surface in suggestions; `apply` will report a
# clear error for them rather than silently no-op.
WRITABLE_EXTENSIONS: frozenset[str] = frozenset({".flac", ".mp3", ".m4a", ".mp4", ".ogg", ".opus"})


@dataclass
class TagReadResult:
    """Flat structure returned from :func:`read_tags`. Values are already stringified.

    ``raw`` preserves the first-value-per-key view of whatever the tag library
    surfaced, for audit purposes.
    """

    title: str | None
    artist: str | None
    album: str | None
    album_artist: str | None
    track_number: int | None
    disc_number: int | None
    year: int | None
    duration_s: float | None
    raw: dict[str, str]


def _first(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        v = value[0]
    else:
        v = value
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _parse_intish(s: str | None) -> int | None:
    if not s:
        return None
    # Track numbers are often "3/12".
    head = s.split("/")[0].strip()
    try:
        return int(head)
    except ValueError:
        # Year tags sometimes contain "2021-05-03".
        try:
            return int(head[:4])
        except ValueError:
            return None


def read_tags(path: Path) -> TagReadResult | None:
    """Read TITLE/ARTIST/etc. from ``path`` using Mutagen's easy interfaces.

    Returns ``None`` if the file cannot be parsed at all (not a known audio file).
    """
    try:
        mf = MutagenFile(path, easy=True)
    except (ID3NoHeaderError, Exception) as e:  # noqa: BLE001 — mutagen raises many types
        log.warning("mutagen could not open %s: %s", path, e)
        return None
    if mf is None:
        log.warning("unrecognized audio file (no tag handler): %s", path)
        return None

    # `tags` can be None for files with no tag frame yet; treat as empty.
    raw: dict[str, str] = {}
    if mf.tags is not None:
        for key, value in dict(mf.tags).items():
            v = _first(value)
            if v is not None:
                raw[str(key).lower()] = v

    duration = None
    try:
        if mf.info is not None and getattr(mf.info, "length", None):
            duration = float(mf.info.length)
    except Exception:  # noqa: BLE001
        duration = None

    return TagReadResult(
        title=raw.get("title"),
        artist=raw.get("artist"),
        album=raw.get("album"),
        album_artist=raw.get("albumartist") or raw.get("album_artist"),
        track_number=_parse_intish(raw.get("tracknumber")),
        disc_number=_parse_intish(raw.get("discnumber")),
        year=_parse_intish(raw.get("date") or raw.get("year")),
        duration_s=duration,
        raw=raw,
    )


def write_title_artist(
    path: Path, *, title: str | None, artist: str | None, dry_run: bool = False
) -> None:
    """Write TITLE and ARTIST (only) back to the file.

    - ``None`` means "do not change this field".
    - Empty string is refused to avoid accidental wipes.
    - Unknown formats are rejected.

    :raises ValueError: on empty proposed value or unknown format.
    """
    if title is not None and title.strip() == "":
        raise ValueError("refusing to write empty title")
    if artist is not None and artist.strip() == "":
        raise ValueError("refusing to write empty artist")

    if dry_run:
        log.info("DRY-RUN would write %s: title=%r artist=%r", path, title, artist)
        return

    ext = path.suffix.lower()
    if ext not in WRITABLE_EXTENSIONS:
        raise ValueError(f"unsupported format for write: {ext}")

    handler = _open_for_write(path, ext)
    if handler is None:
        raise ValueError(f"could not open for write: {path}")

    # All four handlers below accept 'title' / 'artist' as dict keys because they
    # are all Vorbis-comment-like surfaces (FLAC, Ogg, Opus) or mutagen Easy*
    # wrappers (EasyMP3 → ID3, EasyMP4 → iTunes atoms).
    if title is not None:
        handler["title"] = [title]
    if artist is not None:
        handler["artist"] = [artist]
    handler.save()


def _open_for_write(path: Path, ext: str) -> Any:
    """Return a format-specific mutable tag object.

    Every branch returns an object whose ``__setitem__`` accepts the keys
    ``title`` and ``artist`` and whose ``save()`` persists those changes:

    - FLAC uses native Vorbis comments (case-insensitive keys).
    - EasyMP3 maps ``title``/``artist`` → ``TIT2``/``TPE1`` ID3 frames.
    - EasyMP4 maps ``title``/``artist`` → ``\\xa9nam``/``\\xa9ART`` iTunes atoms.
    - OggVorbis / OggOpus use Vorbis comments directly.
    """
    if ext == ".flac":
        return FLAC(path)
    if ext == ".mp3":
        try:
            return EasyMP3(path)
        except ID3NoHeaderError:
            # File has no ID3 header yet; add an empty one and re-open.
            m = EasyMP3(path)
            m.add_tags()
            return m
    if ext in (".m4a", ".mp4"):
        # EasyMP4 translates 'title'/'artist' to iTunes atoms. Raw MP4 would
        # require '\xa9nam' / '\xa9ART' keys, which our write path does not use.
        return EasyMP4(path)
    if ext == ".ogg":
        return OggVorbis(path)
    if ext == ".opus":
        return OggOpus(path)
    return None
