"""Filename-based deterministic fallback when tag TITLE is missing or generic.

The motivating bug: "Aardvarkk - 9 - Bugpowder (Braingineers remix).wav" has no
usable tags, so the pipeline used to escalate to the LLM, which mis-kept "9 - "
in the title and spuriously flagged the result as generic_track_name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from music_normalizer.issues import GENERIC_TITLE_RE, detect_issues
from music_normalizer.models import IssueFlag, TrackInfo
from music_normalizer.rules import normalize_track, parse_filename_fallback


# ---------------------------------------------------------------------------
# parse_filename_fallback — unit tests
# ---------------------------------------------------------------------------


def test_three_part_artist_num_title():
    out = parse_filename_fallback("Aardvarkk - 9 - Bugpowder (Braingineers remix).wav")
    assert out == ("Aardvarkk", "Bugpowder (Braingineers remix)")


def test_three_part_padded_track_number():
    out = parse_filename_fallback("Aardvarkk - 09 - Bugpowder.wav")
    assert out == ("Aardvarkk", "Bugpowder")


def test_three_part_num_artist_title():
    out = parse_filename_fallback("01 - Artist Name - Title Goes Here.flac")
    assert out == ("Artist Name", "Title Goes Here")


def test_two_part_only_with_corroboration():
    # Without corroboration we refuse to split — too easy to misparse.
    assert parse_filename_fallback("Artist - Title.flac") is None
    # With corroboration it works.
    out = parse_filename_fallback("Artist - Title.flac", allow_two_part_split=True)
    assert out == ("Artist", "Title")


def test_leading_number_only_returns_title_only():
    out = parse_filename_fallback("05 - Some Track.flac")
    assert out == (None, "Some Track")


def test_plain_filename_returns_none():
    assert parse_filename_fallback("just_a_filename.flac") is None


def test_underscore_separators_are_folded():
    out = parse_filename_fallback("Artist_-_9_-_Title.wav")
    assert out == ("Artist", "Title")


def test_short_sides_rejected():
    # Single-character artist or title is almost always a false positive.
    assert parse_filename_fallback("A - 1 - B.flac") is None


# ---------------------------------------------------------------------------
# normalize_track integration
# ---------------------------------------------------------------------------


def _track(filename: str, *, title=None, artist=None) -> TrackInfo:
    return TrackInfo(
        path=Path(f"/music/{filename}"),
        filename=filename,
        ext=Path(filename).suffix.lower(),
        title=title,
        artist=artist,
    )


def test_the_motivating_case(config):
    """Aardvarkk WAV with no tags: deterministic parse must recover both fields.

    Note: the mixed-case "Bugpowder (Braingineers remix)" is preserved verbatim
    by design — the capitalization rule only fixes all-lowercase or all-uppercase
    strings, never mixed case (soft-capitalization policy).
    """
    t = _track("Aardvarkk - 9 - Bugpowder (Braingineers remix).wav")
    r = normalize_track(t, config)
    assert r.proposed_artist == "Aardvarkk"
    assert r.proposed_title == "Bugpowder (Braingineers remix)"
    assert IssueFlag.GENERIC_TRACK_NAME not in r.remaining_issue_flags


def test_the_motivating_case_with_artist_tag_already_set(config):
    # Artist tag already correct — we just need to derive the title.
    t = _track(
        "Aardvarkk - 9 - Bugpowder (Braingineers remix).wav",
        artist="Aardvarkk",
    )
    r = normalize_track(t, config)
    assert r.proposed_title == "Bugpowder (Braingineers remix)"
    # Artist unchanged, so no proposed_artist change.
    assert r.proposed_artist is None
    assert IssueFlag.GENERIC_TRACK_NAME not in r.remaining_issue_flags


def test_num_artist_title_filename(config):
    t = _track("01 - Artist Name - Title.flac")
    r = normalize_track(t, config)
    assert r.proposed_artist == "Artist Name"
    assert r.proposed_title == "Title"


def test_filename_fallback_does_not_override_existing_good_artist(config):
    # If the tag says one thing and the filename says another, trust the tag.
    t = _track(
        "Aardvarkk - 9 - Bugpowder.wav",
        artist="Different Artist",
    )
    r = normalize_track(t, config)
    # Title comes from filename, artist preserved from tag.
    assert r.proposed_title == "Bugpowder"
    assert r.proposed_artist is None  # no change to artist


def test_filename_fallback_triggers_on_placeholder_title(config):
    # Tag says "Track 01" — a generic placeholder — so filename fallback runs.
    t = _track(
        "Aardvarkk - 9 - Bugpowder.wav",
        title="Track 01",
    )
    r = normalize_track(t, config)
    assert r.proposed_title == "Bugpowder"
    assert r.proposed_artist == "Aardvarkk"


def test_filename_fallback_skipped_when_title_is_descriptive(config):
    # Tag already has a good title — do NOT override from the filename.
    t = _track(
        "Aardvarkk - 9 - Bugpowder.wav",
        title="Real Title",
        artist="Aardvarkk",
    )
    r = normalize_track(t, config)
    # Either unchanged or only case-normalized — must still be "Real Title".
    assert r.proposed_title in (None, "Real Title")


def test_confidence_not_penalized_when_original_tags_were_missing(config):
    t = _track("Aardvarkk - 9 - Bugpowder (Braingineers remix).wav")
    r = normalize_track(t, config)
    # Confidence must be high enough to be picked up as a real propose_fix,
    # not dumped into manual review just because the original tags were empty.
    assert r.confidence >= config.confidence.low_confidence_flag


# ---------------------------------------------------------------------------
# GENERIC_TITLE_RE coverage — placeholders still flagged, real titles aren't
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title",
    ["Track 01", "Track 5", "Untitled", "Untitled 7", "Audio Track 03", "T9", "12"],
)
def test_placeholders_still_match(title):
    assert GENERIC_TITLE_RE.match(title)


@pytest.mark.parametrize(
    "title",
    ["Bugpowder", "Pure Energy", "Nocturnal Visions", "9mm", "Track Record"],
)
def test_descriptive_titles_do_not_match(title):
    assert GENERIC_TITLE_RE.match(title) is None


def test_descriptive_title_does_not_raise_generic_flag(config):
    flags = detect_issues("Bugpowder (Braingineers remix)", "Aardvarkk", config)
    assert IssueFlag.GENERIC_TRACK_NAME not in flags


def test_placeholder_title_still_raises_generic_flag(config):
    flags = detect_issues("Track 01", "Aardvarkk", config)
    assert IssueFlag.GENERIC_TRACK_NAME in flags
