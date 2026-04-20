"""Deterministic rules engine tests — the most safety-critical layer."""

from __future__ import annotations

from pathlib import Path

from music_normalizer.models import IssueFlag, TrackInfo
from music_normalizer.rules import normalize_track


def _track(title: str | None, artist: str | None = "") -> TrackInfo:
    return TrackInfo(path=Path("/t.flac"), filename="t.flac", ext=".flac",
                    title=title, artist=artist)


def test_strips_leading_track_number(config):
    t = _track("03 - some darkpsy banger", artist="xikwri")
    r = normalize_track(t, config)
    assert r.proposed_title == "Some Darkpsy Banger"


def test_removes_original_mix_paren(config):
    t = _track("Nocturnal Visions (Original Mix)", artist="Kindzadza")
    r = normalize_track(t, config)
    assert r.proposed_title == "Nocturnal Visions"
    # The flag should be raised against the *original* input and cleared from
    # the remaining set once the rule fires.
    assert IssueFlag.ORIGINAL_MIX_REDUNDANT in r.original_issue_flags
    assert IssueFlag.ORIGINAL_MIX_REDUNDANT not in r.remaining_issue_flags


def test_removes_original_mix_dash(config):
    t = _track("Mind Overload - Original Mix", artist="Para Halu")
    r = normalize_track(t, config)
    assert r.proposed_title == "Mind Overload"


def test_strips_bpm_suffix(config):
    t = _track("Dark Crystal (160 bpm)", artist="Xikwri")
    r = normalize_track(t, config)
    assert r.proposed_title == "Dark Crystal"
    assert IssueFlag.BPM_SUFFIX in r.original_issue_flags
    assert IssueFlag.BPM_SUFFIX not in r.remaining_issue_flags


def test_normalizes_underscores(config):
    t = _track("some_broken_title_string", artist="Artist")
    r = normalize_track(t, config)
    assert r.proposed_title == "Some Broken Title String"
    assert IssueFlag.WEIRD_SEPARATORS in r.original_issue_flags
    assert IssueFlag.WEIRD_SEPARATORS not in r.remaining_issue_flags


def test_normalizes_em_dash(config):
    t = _track("title — with em dash", artist="a")
    r = normalize_track(t, config)
    assert r.proposed_title == "Title - With Em Dash"


def test_infers_artist_from_title_when_missing(config):
    # Single-track call with the split explicitly allowed. Album-level
    # corroboration is exercised separately in test_album_artist_corroboration.
    t = _track("Kindzadza - Nocturnal Visions", artist="")
    r = normalize_track(t, config, allow_artist_split=True)
    assert r.proposed_title == "Nocturnal Visions"
    assert r.proposed_artist == "Kindzadza"
    assert IssueFlag.MISSING_ARTIST_BUT_PRESENT_IN_TITLE in r.original_issue_flags


def test_split_ambiguous_not_applied(config):
    # Two " - " separators should bail on the split.
    t = _track("xikwri - the fall - remix", artist="")
    r = normalize_track(t, config)
    assert r.proposed_artist is None  # didn't invent anything


def test_duplicate_artist_title_triggers_split(config):
    t = _track("Kindzadza - Nocturnal Visions", artist="Kindzadza - Nocturnal Visions")
    r = normalize_track(t, config)
    assert r.proposed_title == "Nocturnal Visions"
    assert r.proposed_artist == "Kindzadza"


def test_title_case_preserves_stopwords(config):
    t = _track("the tree of life", artist="a")
    r = normalize_track(t, config)
    assert r.proposed_title == "The Tree of Life"


def test_preserves_acronyms(config):
    t = _track("va - darkpsy compilation", artist="")
    r = normalize_track(t, config)
    # Junk prefix "va -" is stripped.
    assert r.proposed_title == "Darkpsy Compilation"


def test_strips_junk_suffix(config):
    t = _track("track name (free download)", artist="artist")
    r = normalize_track(t, config)
    assert r.proposed_title == "Track Name"


def test_generic_title_is_flagged_and_not_confidently_changed(config):
    t = _track("Track 01", artist="artist")
    r = normalize_track(t, config)
    assert IssueFlag.GENERIC_TRACK_NAME in r.issue_flags
    assert r.confidence <= 0.5


def test_unreadable_title(config):
    t = _track("___---", artist="artist")
    r = normalize_track(t, config)
    assert IssueFlag.UNREADABLE_TITLE in r.issue_flags


def test_empty_title_untouched_if_nothing_to_change(config):
    t = _track(None, artist="Kindzadza")
    r = normalize_track(t, config)
    assert r.proposed_title is None
    assert r.proposed_artist is None


def test_collapses_multiple_spaces(config):
    t = _track("too   much    space", artist="a")
    r = normalize_track(t, config)
    assert r.proposed_title == "Too Much Space"


# ---------------------------------------------------------------------------
# Stronger BPM / bracket stripping
# ---------------------------------------------------------------------------


def test_strips_bracketed_bare_number(config):
    t = _track("Dark Crystal [160]", artist="Kindzadza")
    r = normalize_track(t, config)
    assert r.proposed_title == "Dark Crystal"


def test_strips_bracketed_bpm(config):
    t = _track("Dark Crystal [160bpm]", artist="Kindzadza")
    r = normalize_track(t, config)
    assert r.proposed_title == "Dark Crystal"


def test_strips_bpm_range_bare(config):
    t = _track("Mind Overload 178-200bpm", artist="Para Halu")
    r = normalize_track(t, config)
    assert r.proposed_title == "Mind Overload"


def test_strips_bpm_range_parens(config):
    t = _track("Mind Overload (178-200bpm)", artist="Para Halu")
    r = normalize_track(t, config)
    assert r.proposed_title == "Mind Overload"


def test_strips_bpm_range_brackets_no_suffix(config):
    t = _track("Mind Overload [190-220]", artist="Para Halu")
    r = normalize_track(t, config)
    assert r.proposed_title == "Mind Overload"


def test_strips_bpm_unicode_dash_in_range(config):
    t = _track("Mind Overload [178–220bpm]", artist="Para Halu")
    r = normalize_track(t, config)
    assert r.proposed_title == "Mind Overload"


def test_does_not_strip_non_bpm_brackets(config):
    # Bracketed text without a leading number is not BPM and must be preserved.
    t = _track("Track (Live)", artist="a")
    r = normalize_track(t, config)
    # Either unchanged or case-normalized, but "(Live)" stays.
    assert r.proposed_title is None or "Live" in r.proposed_title


def test_bpm_only_title_not_stripped_to_empty(config):
    # If the entire title is tempo junk, leave it for manual review rather
    # than wipe it out.
    t = _track("[160bpm]", artist="a")
    r = normalize_track(t, config)
    assert r.proposed_title != ""
    assert r.proposed_title is None or "160" in r.proposed_title or "[" in r.proposed_title


# ---------------------------------------------------------------------------
# Artist prefix strip (even when artist tag is populated)
# ---------------------------------------------------------------------------


def test_strips_artist_prefix_when_artist_tag_matches(config):
    t = _track("Mertis - Pure Energy", artist="Mertis")
    r = normalize_track(t, config)
    assert r.proposed_title == "Pure Energy"
    # Artist field must not be mutated — it was already correct.
    assert r.proposed_artist is None


def test_strips_artist_prefix_then_bpm(config):
    t = _track("Psy-VitaMinD - Beginnings - 165 Bpm", artist="Psy-VitaMinD")
    r = normalize_track(t, config)
    assert r.proposed_title == "Beginnings"


def test_does_not_strip_unrelated_prefix(config):
    # Single-word artist happens to be first word of title — we only strip
    # when followed by a recognized separator, so "Mertis Pure Energy"
    # (no separator) is NOT stripped.
    t = _track("Mertis Pure Energy", artist="Mertis")
    r = normalize_track(t, config)
    # Title should be unchanged (or only cased).
    assert r.proposed_title in (None, "Mertis Pure Energy")


def test_strips_artist_prefix_case_insensitive(config):
    t = _track("KINDZADZA - nocturnal visions", artist="Kindzadza")
    r = normalize_track(t, config)
    assert r.proposed_title == "Nocturnal Visions"
