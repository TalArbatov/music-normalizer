"""Stable identifiers for albums.

Album ID is a short hash of the folder path relative to the staging root — stable
across runs as long as the folder isn't moved.

Fingerprint folds in per-file (path, mtime, size). Any retag on disk changes the
mtime and therefore invalidates the cache. That's the intent.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .models import TrackInfo


def album_id_for(root: Path, staging_root: Path | None = None) -> str:
    """Deterministic short id derived from the album's location."""
    try:
        rel = root.relative_to(staging_root) if staging_root is not None else root
    except ValueError:
        rel = root
    return hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:12]


def album_fingerprint(tracks: list[TrackInfo]) -> str:
    """Hash over sorted (path, mtime, size) triples.

    Using mtime and size is cheap and good enough: any tag rewrite bumps mtime on
    every OS we care about. We don't hash file contents — that would be far too slow
    on a large library.
    """
    h = hashlib.sha1()
    for t in sorted(tracks, key=lambda x: str(x.path)):
        h.update(str(t.path).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(t.mtime or 0).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(t.size_bytes or 0).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()
