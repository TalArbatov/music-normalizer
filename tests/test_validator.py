from __future__ import annotations

from pathlib import Path

from music_normalizer.models import (
    IssueFlag,
    Suggestion,
    SuggestionSource,
    TrackAction,
)
from music_normalizer.validator import validate


def _sug(**kw) -> Suggestion:
    defaults = dict(
        track_path=Path("/t.flac"),
        current_title="a",
        current_artist="b",
        proposed_title="a",
        proposed_artist="b",
        action=TrackAction.KEEP,
        confidence=1.0,
        reason="",
        source=SuggestionSource.DETERMINISTIC,
    )
    defaults.update(kw)
    return Suggestion(**defaults)


def test_no_change_downgraded_to_keep(config):
    s = _sug(action=TrackAction.PROPOSE_FIX)
    v = validate(s, config)
    assert v.action == TrackAction.KEEP


def test_empty_proposed_title_rejected(config):
    s = _sug(
        current_title="Nocturnal Visions",
        proposed_title="",
        action=TrackAction.PROPOSE_FIX,
    )
    v = validate(s, config)
    assert v.proposed_title == "Nocturnal Visions"
    assert v.action == TrackAction.UNREADABLE_NEEDS_MANUAL_REVIEW


def test_swap_suspected_and_downgraded(config):
    s = _sug(
        current_title="Kindzadza",
        current_artist="Nocturnal Visions",
        proposed_title="Nocturnal Visions",
        proposed_artist="Kindzadza",
        action=TrackAction.PROPOSE_FIX,
    )
    v = validate(s, config)
    assert IssueFlag.SUSPICIOUS_SWAP_ARTIST_TITLE in v.issue_flags
    assert v.action == TrackAction.UNREADABLE_NEEDS_MANUAL_REVIEW


def test_llm_full_rewrite_downgraded(config):
    s = _sug(
        current_title="zzzzzzz",
        current_artist="qqqqqq",
        proposed_title="Completely Different Title",
        proposed_artist="Entirely Other Artist",
        action=TrackAction.PROPOSE_FIX,
        source=SuggestionSource.LLM,
        confidence=0.95,
    )
    v = validate(s, config)
    assert v.action == TrackAction.UNREADABLE_NEEDS_MANUAL_REVIEW


def test_low_confidence_stays_propose_fix_but_notes_added(config):
    s = _sug(
        current_title="x",
        proposed_title="Y Title",
        action=TrackAction.PROPOSE_FIX,
        confidence=0.3,
    )
    v = validate(s, config)
    # Low confidence alone doesn't downgrade, but it should show a note.
    assert any("confidence" in n for n in v.validator_notes)


def test_generic_proposed_still_flagged(config):
    s = _sug(
        current_title="garbage",
        proposed_title="Track 5",
        action=TrackAction.PROPOSE_FIX,
        confidence=0.9,
    )
    v = validate(s, config)
    assert v.action == TrackAction.UNREADABLE_NEEDS_MANUAL_REVIEW
