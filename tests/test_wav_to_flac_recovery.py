"""WAV->FLAC albums: only BPM tag, but filenames encode artist/title.

The motivating album is ``2LAVE RAC3 - Occult Acid Techno Trance`` (job id
``651d28cdb2a9``): every FLAC exports only ``BPM=NNN`` as its tag block, but
filenames carry the full ``N. 2LAVE RAC3 - Song Title.flac`` structure. The
pipeline must recognize this, propose deterministic fixes, and NOT escalate to
the LLM.
"""

from __future__ import annotations

from pathlib import Path

from music_normalizer.classifier import classify_album
from music_normalizer.models import AlbumClassification, AlbumManifest, TrackInfo
from music_normalizer.probe import probe_album
from music_normalizer.rules import (
    album_supports_filename_recovery,
    parse_filename_fallback,
    propose_for_album,
)

# Filenames taken verbatim from the real 2LAVE RAC3 album manifest.
_FILENAMES = [
    "1. 2LAVE RAC3 - Bloodbath.flac",
    "2. 2LAVE RAC3 - The Magi (We Have The Power).flac",
    "3. 2LAVE RAC3 - We Are The Shamans.flac",
    "4. 2LAVE RAC3 - New Generation (Rock The Earth!).flac",
    "5. 2LAVE RAC3 - Climax Vibrations.flac",
    "6. 2LAVE RAC3 - Acid Reality.flac",
    "10. 2LAVE RAC3 - Talkin Revolution.flac",
]


def _wav_to_flac_track(filename: str) -> TrackInfo:
    """Simulate a FLAC where only BPM was preserved from the source WAV."""
    return TrackInfo(
        path=Path("/music") / filename,
        filename=filename,
        ext=".flac",
        title=None,
        artist=None,
        raw_tags={"bpm": "141"},
    )


def _wav_to_flac_manifest(filenames: list[str]) -> AlbumManifest:
    return AlbumManifest(
        album_id="651d28cdb2a9",
        root=Path("/music/2LAVE RAC3 - Occult Acid Techno Trance"),
        album_name="2LAVE RAC3 - Occult Acid Techno Trance",
        tracks=[_wav_to_flac_track(f) for f in filenames],
        fingerprint="f" * 40,
    )


# ---------------------------------------------------------------------------
# parse_filename_fallback — new "N. Artist - Title" anchor
# ---------------------------------------------------------------------------


def test_two_digit_num_dot_artist_title():
    assert parse_filename_fallback("10. 2LAVE RAC3 - Talkin Revolution.flac") == (
        "2LAVE RAC3", "Talkin Revolution",
    )


def test_single_digit_num_dot_artist_title():
    assert parse_filename_fallback("3. 2LAVE RAC3 - We Are The Shamans.flac") == (
        "2LAVE RAC3", "We Are The Shamans",
    )


def test_leading_punctuation_without_digit():
    # E.g. a filename that lost its track number but kept the leading dot.
    assert parse_filename_fallback(". 2LAVE RAC3 - Bloodbath.flac") == (
        "2LAVE RAC3", "Bloodbath",
    )


def test_num_dash_artist_title_still_works():
    # The original "03 - Artist - Title" form must remain supported.
    assert parse_filename_fallback("03 - Artist - Title.flac") == (
        "Artist", "Title",
    )


def test_bare_two_part_still_requires_corroboration():
    # "Artist - Title.flac" has no leading anchor — still gated.
    assert parse_filename_fallback("Artist - Title.flac") is None


# ---------------------------------------------------------------------------
# album_supports_filename_recovery
# ---------------------------------------------------------------------------


def test_album_supported_when_many_strong_filenames():
    tracks = [_wav_to_flac_track(f) for f in _FILENAMES]
    assert album_supports_filename_recovery(tracks) is True


def test_album_not_supported_when_filenames_are_noise():
    tracks = [
        _wav_to_flac_track("track1.flac"),
        _wav_to_flac_track("track2.flac"),
        _wav_to_flac_track("track3.flac"),
    ]
    assert album_supports_filename_recovery(tracks) is False


def test_album_supported_when_exactly_two_strong_filenames():
    # Threshold is 2 — the task spec's minimum.
    tracks = [
        _wav_to_flac_track("1. Artist One - Song A.flac"),
        _wav_to_flac_track("2. Artist One - Song B.flac"),
        _wav_to_flac_track("track_whatever.flac"),
    ]
    assert album_supports_filename_recovery(tracks) is True


# ---------------------------------------------------------------------------
# End-to-end: the 2LAVE album flips BROKEN -> DETERMINISTIC
# ---------------------------------------------------------------------------


def test_propose_for_album_recovers_every_track(config):
    tracks = [_wav_to_flac_track(f) for f in _FILENAMES]
    cands = propose_for_album(tracks, config)
    for t in tracks:
        c = cands[t.path]
        assert c.proposed_title is not None, f"no title recovered for {t.filename}"
        assert c.proposed_artist is not None, f"no artist recovered for {t.filename}"
        # Case-normalization may soften "2LAVE RAC3" but the core tokens remain.
        assert "2lave" in c.proposed_artist.lower()
        assert "rac3" in c.proposed_artist.lower()


def test_album_classifies_deterministic_not_broken(config):
    manifest = _wav_to_flac_manifest(_FILENAMES)
    probe = probe_album(manifest, config)
    cands = propose_for_album(manifest.tracks, config)
    result = classify_album(
        probe, {str(k): v for k, v in cands.items()}, config,
    )
    assert result.classification == AlbumClassification.DETERMINISTIC, (
        f"expected DETERMINISTIC, got {result.classification.value}: {result.reason}"
    )


def test_candidates_have_no_blocking_flags(config):
    # Guard: the classifier's DETERMINISTIC path refuses to fire when any
    # candidate still carries a blocking flag — make sure none do.
    blocking = {
        "unreadable_title", "generic_track_name",
        "embedded_album_name", "suspicious_swap_artist_title",
    }
    tracks = [_wav_to_flac_track(f) for f in _FILENAMES]
    cands = propose_for_album(tracks, config)
    for c in cands.values():
        present = {f.value for f in c.remaining_issue_flags}
        assert present.isdisjoint(blocking), (
            f"blocking flags present on {c.track_path}: {present & blocking}"
        )
