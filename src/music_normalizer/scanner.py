"""Filesystem traversal. Produces raw TrackInfo records for the grouper."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .models import TrackInfo
from .tags import read_tags
from .utils import get_logger

log = get_logger(__name__)


def scan_tracks(root: Path, supported_extensions: Iterable[str]) -> list[TrackInfo]:
    """Walk ``root`` and return a TrackInfo for every audio file found.

    Hidden files are skipped. Tag read failures are logged and skipped silently so
    that one bad file does not abort a large library scan.
    """
    exts = {e.lower() for e in supported_extensions}
    out: list[TrackInfo] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in exts:
            continue
        info = _build_track_info(p)
        if info is not None:
            out.append(info)
    log.info("scanned %d audio files under %s", len(out), root)
    return out


def _build_track_info(path: Path) -> TrackInfo | None:
    try:
        st = path.stat()
    except OSError as e:
        log.warning("stat failed for %s: %s", path, e)
        return None

    tags = read_tags(path)
    if tags is None:
        # Still emit a TrackInfo so the rest of the pipeline can flag it as unreadable.
        return TrackInfo(
            path=path,
            filename=path.name,
            ext=path.suffix.lower(),
            mtime=st.st_mtime,
            size_bytes=st.st_size,
        )

    return TrackInfo(
        path=path,
        filename=path.name,
        ext=path.suffix.lower(),
        title=tags.title,
        artist=tags.artist,
        album=tags.album,
        album_artist=tags.album_artist,
        track_number=tags.track_number,
        disc_number=tags.disc_number,
        year=tags.year,
        duration_s=tags.duration_s,
        raw_tags=tags.raw,
        mtime=st.st_mtime,
        size_bytes=st.st_size,
    )
