"""Strict validation of Ollama responses."""

from __future__ import annotations

from music_normalizer.prompts import validate_response


def _ok_entry(**overrides):
    base = {
        "track_path": "/a.flac",
        "action": "keep",
        "proposed_title": "T",
        "proposed_artist": "A",
        "confidence": 0.9,
        "issue_flags": [],
        "reason": "",
    }
    base.update(overrides)
    return base


def test_accepts_well_formed_response():
    raw = {"album_id": "x", "tracks": [_ok_entry()]}
    ok, reason = validate_response(raw)
    assert ok, reason


def test_rejects_non_object():
    ok, reason = validate_response([])
    assert not ok and "object" in reason


def test_rejects_missing_tracks_key():
    ok, reason = validate_response({"album_id": "x"})
    assert not ok and "tracks" in reason


def test_rejects_invalid_action():
    raw = {"tracks": [_ok_entry(action="delete_file")]}
    ok, reason = validate_response(raw)
    assert not ok and "action" in reason


def test_rejects_out_of_range_confidence():
    raw = {"tracks": [_ok_entry(confidence=1.5)]}
    ok, reason = validate_response(raw)
    assert not ok and "confidence" in reason


def test_rejects_non_string_proposed_title():
    raw = {"tracks": [_ok_entry(proposed_title=123)]}
    ok, reason = validate_response(raw)
    assert not ok and "proposed_title" in reason


def test_accepts_null_proposed_fields():
    raw = {"tracks": [_ok_entry(proposed_title=None, proposed_artist=None)]}
    ok, reason = validate_response(raw)
    assert ok, reason
