"""Artist extraction from title requires album-level corroboration."""

from __future__ import annotations

from pathlib import Path

from music_normalizer.models import TrackInfo
from music_normalizer.rules import album_supports_artist_split, propose_for_album


def _t(title: str, artist: str = "") -> TrackInfo:
    # Path must be unique so dict-by-path doesn't collapse entries.
    idx = abs(hash(title)) % 10_000_000
    return TrackInfo(path=Path(f"/{idx}.flac"), filename=f"{idx}.flac",
                    ext=".flac", title=title, artist=artist)


def test_album_supports_split_when_many_tracks_match(config):
    tracks = [
        _t("Kindzadza - Nocturnal Visions", ""),
        _t("Kindzadza - Dark Crystal", ""),
        _t("Kindzadza - Event Horizon", ""),
    ]
    assert album_supports_artist_split(tracks, config) is True


def test_album_rejects_split_when_only_one_track_matches(config):
    tracks = [
        _t("Kindzadza - Nocturnal Visions", ""),
        _t("Random Clean Title", "Kindzadza"),
        _t("Another Clean Title", "Kindzadza"),
        _t("Yet Another", "Kindzadza"),
    ]
    # Only one track exhibits the pattern — not enough to generalize.
    assert album_supports_artist_split(tracks, config) is False


def test_album_rejects_ambiguous_titles(config):
    tracks = [
        _t("Artist - Title - Remix", ""),
        _t("Artist - Title - Extended", ""),
    ]
    # Two dashes each → ambiguous, not a supporting pattern.
    assert album_supports_artist_split(tracks, config) is False


def test_deterministic_skips_split_without_corroboration(config):
    tracks = [
        _t("Kindzadza - Nocturnal Visions", ""),
        _t("Random Title", "Kindzadza"),
        _t("Another Title", "Kindzadza"),
        _t("Something Else", "Kindzadza"),
    ]
    cands = propose_for_album(tracks, config)
    # The one track that looks like "Artist - Title" must NOT have its artist
    # deterministically inferred when the rest of the album doesn't corroborate.
    odd = next(c for c in cands.values() if c.track_path == tracks[0].path)
    assert odd.proposed_artist is None  # no split applied


def test_deterministic_applies_split_with_corroboration(config):
    tracks = [
        _t("Kindzadza - Nocturnal Visions", ""),
        _t("Kindzadza - Dark Crystal", ""),
        _t("Kindzadza - Event Horizon", ""),
        _t("Kindzadza - Aphotic", ""),
    ]
    cands = propose_for_album(tracks, config)
    for t in tracks:
        c = cands[t.path]
        assert c.proposed_artist == "Kindzadza"
        assert c.proposed_title is not None and " - " not in c.proposed_title
