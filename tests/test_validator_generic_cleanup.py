"""Validator strips LLM-induced spurious generic_track_name flags."""

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
        current_title="9 - Bugpowder",
        current_artist=None,
        proposed_title="Bugpowder",
        proposed_artist="Aardvarkk",
        action=TrackAction.PROPOSE_FIX,
        confidence=0.85,
        reason="",
        source=SuggestionSource.LLM,
        issue_flags=[IssueFlag.GENERIC_TRACK_NAME],
        remaining_issue_flags=[IssueFlag.GENERIC_TRACK_NAME],
    )
    defaults.update(kw)
    return Suggestion(**defaults)


def test_spurious_generic_flag_stripped_from_descriptive_title(config):
    s = _sug()
    v = validate(s, config)
    assert IssueFlag.GENERIC_TRACK_NAME not in v.remaining_issue_flags
    assert IssueFlag.GENERIC_TRACK_NAME not in v.issue_flags
    assert any("spurious generic_track_name" in n for n in v.validator_notes)
    # Crucially, the action is NOT downgraded — this is still a real propose_fix.
    assert v.action == TrackAction.PROPOSE_FIX


def test_truly_generic_proposed_title_still_downgrades(config):
    s = _sug(
        proposed_title="Track 01",
        proposed_artist="Aardvarkk",
    )
    v = validate(s, config)
    assert v.action == TrackAction.UNREADABLE_NEEDS_MANUAL_REVIEW
    # The flag stays because the title really is generic.
    # (It's fine if it's still present — the key behavior is the downgrade.)


def test_bare_number_proposed_title_downgrades(config):
    s = _sug(proposed_title="9", proposed_artist="Aardvarkk")
    v = validate(s, config)
    assert v.action == TrackAction.UNREADABLE_NEEDS_MANUAL_REVIEW


def test_single_word_descriptive_title_is_not_downgraded(config):
    s = _sug(proposed_title="Bugpowder", proposed_artist="Aardvarkk")
    v = validate(s, config)
    assert v.action == TrackAction.PROPOSE_FIX
    assert IssueFlag.GENERIC_TRACK_NAME not in v.remaining_issue_flags
