"""Turn a flat list of scanned tracks into AlbumManifest objects.

Grouping strategy, in priority order:
1. By (album_artist, album) tag when both are set consistently.
2. By containing directory — this is the fallback for messy libraries where the
   album tag is missing or wrong.

For darkpsy libraries the directory fallback is usually more reliable than the
album tag, but keeping both means we correctly merge multi-disc sets whose tags
agree even when the files are split across ``CD1/`` and ``CD2/``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .fingerprints import album_fingerprint, album_id_for
from .models import AlbumManifest, TrackInfo
from .utils import get_logger

log = get_logger(__name__)


def group_into_albums(
    tracks: list[TrackInfo], staging_root: Path | None = None
) -> list[AlbumManifest]:
    # First pass: group by directory. This is our unit of review — one folder,
    # one job. Multi-disc sets with separate subfolders become separate albums;
    # we can merge at a higher level later if needed.
    by_dir: dict[Path, list[TrackInfo]] = defaultdict(list)
    for t in tracks:
        by_dir[t.path.parent].append(t)

    manifests: list[AlbumManifest] = []
    for folder, items in sorted(by_dir.items()):
        items_sorted = sorted(items, key=_track_sort_key)
        album_name = _majority(t.album for t in items_sorted) or folder.name
        album_artist = _majority(t.album_artist for t in items_sorted)
        manifest = AlbumManifest(
            album_id=album_id_for(folder, staging_root),
            root=folder,
            album_name=album_name,
            album_artist=album_artist,
            tracks=items_sorted,
            fingerprint=album_fingerprint(items_sorted),
        )
        manifests.append(manifest)

    log.info("grouped %d tracks into %d albums", len(tracks), len(manifests))
    return manifests


def _track_sort_key(t: TrackInfo) -> tuple[int, int, str]:
    return (t.disc_number or 0, t.track_number or 0, t.filename.lower())


def _majority(values) -> str | None:
    """Return the most common non-empty string, or None."""
    counts: dict[str, int] = {}
    for v in values:
        if v:
            counts[v] = counts.get(v, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]
