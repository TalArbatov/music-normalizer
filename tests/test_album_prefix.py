"""Album-wide repeated prefix detection and stripping."""

from __future__ import annotations

from pathlib import Path

from music_normalizer.models import TrackInfo
from music_normalizer.rules import (
    _leftmost_segment_of_prefix,
    detect_common_prefix,
    propose_for_album,
)


def _t(title: str, artist: str = "", album_artist: str | None = None) -> TrackInfo:
    idx = abs(hash(title + artist)) % 10_000_000
    return TrackInfo(
        path=Path(f"/{idx}.flac"),
        filename=f"{idx}.flac",
        ext=".flac",
        title=title,
        artist=artist,
        album_artist=album_artist,
    )


def test_detect_common_prefix_artist_only():
    titles = [
        "Kindzadza - Nocturnal Visions",
        "Kindzadza - Dark Crystal",
        "Kindzadza - Event Horizon",
        "Kindzadza - Aphotic",
    ]
    assert detect_common_prefix(titles) == "Kindzadza - "


def test_detect_common_prefix_artist_and_album():
    titles = [
        "Dazkull Eghyrth - Chaos Of Redemption - Intro",
        "Dazkull Eghyrth - Chaos Of Redemption - Comet",
        "Dazkull Eghyrth - Chaos Of Redemption - Track3",
    ]
    # Longer prefix wins.
    assert detect_common_prefix(titles) == "Dazkull Eghyrth - Chaos Of Redemption - "


def test_detect_returns_none_below_threshold():
    titles = [
        "Kindzadza - A",
        "Kindzadza - B",
        "Totally Different",
        "Also Different",
        "Yet Another",
    ]
    # Only 2/5 = 40% share the prefix; below the 60% default threshold.
    assert detect_common_prefix(titles) is None


def test_detect_requires_minimum_track_count():
    titles = ["Artist - A", "Artist - B"]
    # Too few tracks to generalize from.
    assert detect_common_prefix(titles) is None


def test_detect_prefers_longer_when_all_remainders_are_non_empty():
    titles = [
        "Artist - Album - Intro",
        "Artist - Album - Comet",
        "Artist - Album - Shadows",
    ]
    # Longer prefix wins because all tracks have non-empty remainders.
    assert detect_common_prefix(titles) == "Artist - Album - "


def test_detect_falls_back_when_longer_prefix_fails_threshold():
    # Only one track carries the "Artist - Album - " prefix, but every track
    # starts with "Artist - ". The detector should fall back to the shorter
    # prefix that actually clears the threshold.
    titles = [
        "Artist - Album - Intro",
        "Artist - Comet",
        "Artist - Shadows",
        "Artist - Aftermath",
    ]
    assert detect_common_prefix(titles) == "Artist - "


def test_leftmost_segment_single_level():
    assert _leftmost_segment_of_prefix("Kindzadza - ") == "Kindzadza"


def test_leftmost_segment_two_levels():
    assert (
        _leftmost_segment_of_prefix("Dazkull Eghyrth - Chaos Of Redemption - ")
        == "Dazkull Eghyrth"
    )


def test_propose_for_album_strips_repeated_prefix(config):
    tracks = [
        _t("Dazkull Eghyrth - Chaos Of Redemption - Intro"),
        _t("Dazkull Eghyrth - Chaos Of Redemption - Comet"),
        _t("Dazkull Eghyrth - Chaos Of Redemption - Shadows"),
        _t("Dazkull Eghyrth - Chaos Of Redemption - Aftermath"),
    ]
    cands = propose_for_album(tracks, config)
    titles = [c.proposed_title for c in cands.values()]
    assert "Intro" in titles
    assert "Comet" in titles
    assert all(t is not None and " - " not in t for t in titles)
    # Since artist tag was empty, the leftmost segment should backfill it.
    assert all(c.proposed_artist == "Dazkull Eghyrth" for c in cands.values())


def test_propose_for_album_prefix_with_track_numbers(config):
    # Leading track numbers must not fool the detector — they're stripped first.
    tracks = [
        _t("01 - Kindzadza - Nocturnal Visions"),
        _t("02 - Kindzadza - Dark Crystal"),
        _t("03 - Kindzadza - Event Horizon"),
        _t("04 - Kindzadza - Aphotic"),
    ]
    cands = propose_for_album(tracks, config)
    titles = [c.proposed_title for c in cands.values()]
    for t in titles:
        assert t is not None
        assert "Kindzadza" not in t
        assert not t[0].isdigit()


def test_propose_for_album_no_prefix_when_tracks_are_varied(config):
    tracks = [
        _t("Nocturnal Visions", artist="Kindzadza"),
        _t("Dark Crystal", artist="Kindzadza"),
        _t("Event Horizon", artist="Kindzadza"),
        _t("Aphotic", artist="Kindzadza"),
    ]
    cands = propose_for_album(tracks, config)
    # Nothing to strip; either unchanged or only case-fixed.
    for t, c in zip(tracks, cands.values()):
        assert c.proposed_artist is None
        # Title either stays or case-normalizes, never gets a word removed.
        assert c.proposed_title in (None, t.title)
