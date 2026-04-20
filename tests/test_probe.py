from __future__ import annotations

from datetime import datetime
from pathlib import Path

from music_normalizer.models import AlbumManifest, TrackInfo
from music_normalizer.probe import probe_album


def _mk_manifest(titles: list[str]) -> AlbumManifest:
    tracks = [
        TrackInfo(path=Path(f"/{i}.flac"), filename=f"{i}.flac", ext=".flac",
                  title=t, artist="Artist", track_number=i + 1)
        for i, t in enumerate(titles)
    ]
    return AlbumManifest(
        album_id="a", root=Path("/"), tracks=tracks,
        fingerprint="f", scanned_at=datetime.utcnow(),
    )


def test_clean_album_probes_high(config):
    titles = ["Nocturnal Visions", "Dark Crystal", "Mind Overload", "Event Horizon", "Aphotic"]
    probe = probe_album(_mk_manifest(titles), config)
    assert probe.score >= 0.9
    assert probe.hard_flag_count == 0


def test_broken_album_probes_low(config):
    titles = ["Track 01", "Track 02", "___", "Untitled", "  "]
    probe = probe_album(_mk_manifest(titles), config)
    assert probe.score < 0.6
    assert probe.hard_flag_count > 0


def test_soft_issues_reduce_score_moderately(config):
    titles = ["nocturnal_visions", "dark_crystal", "mind_overload", "event_horizon"]
    probe = probe_album(_mk_manifest(titles), config)
    # Soft issues exist but no hard ones — should still sit in the deterministic zone.
    assert 0.5 <= probe.score < 0.95


def test_sample_positions_picked(config):
    titles = [f"Track {i}" for i in range(10)]
    probe = probe_album(_mk_manifest(titles), config)
    # first, second, middle (idx 4), last (idx 9) → 4 samples
    assert len(probe.sampled_track_paths) == 4
