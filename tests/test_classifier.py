from __future__ import annotations

from pathlib import Path

from music_normalizer.classifier import classify_album
from music_normalizer.models import AlbumClassification, DeterministicCandidate, IssueFlag
from music_normalizer.probe import ProbeResult


def _probe(score: float) -> ProbeResult:
    return ProbeResult(score=score, sampled_track_paths=[], hard_flag_count=0,
                      soft_flag_count=0, notes=[])


def _cand(confidence: float = 1.0, flags=None, proposed=False) -> DeterministicCandidate:
    return DeterministicCandidate(
        track_path=Path("/t.flac"),
        proposed_title="New" if proposed else None,
        confidence=confidence,
        issue_flags=flags or [],
    )


def test_clean_when_probe_high_and_no_changes(config):
    cands = {"/t.flac": _cand(confidence=1.0, proposed=False)}
    r = classify_album(_probe(0.95), cands, config)
    assert r.classification == AlbumClassification.CLEAN


def test_deterministic_when_rules_cover_all(config):
    cands = {"/t.flac": _cand(confidence=0.9, proposed=True)}
    r = classify_album(_probe(0.7), cands, config)
    assert r.classification == AlbumClassification.DETERMINISTIC


def test_mixed_when_low_confidence(config):
    cands = {"/t.flac": _cand(confidence=0.4, proposed=True)}
    r = classify_album(_probe(0.7), cands, config)
    assert r.classification == AlbumClassification.MIXED


def test_broken_when_majority_unreadable(config):
    cands = {
        "/a.flac": _cand(confidence=0.3, flags=[IssueFlag.UNREADABLE_TITLE]),
        "/b.flac": _cand(confidence=0.3, flags=[IssueFlag.UNREADABLE_TITLE]),
        "/c.flac": _cand(confidence=0.9),
    }
    r = classify_album(_probe(0.1), cands, config)
    assert r.classification == AlbumClassification.BROKEN


def test_blocking_flag_prevents_deterministic(config):
    cands = {"/t.flac": _cand(confidence=0.9, flags=[IssueFlag.EMBEDDED_ALBUM_NAME],
                              proposed=True)}
    r = classify_album(_probe(0.7), cands, config)
    assert r.classification == AlbumClassification.MIXED


def test_probe_below_deterministic_threshold_is_mixed(config):
    # Probe below deterministic_threshold (0.6 by default) should never be
    # classified deterministic even if the candidates look fine.
    cands = {"/t.flac": _cand(confidence=0.9, proposed=True)}
    r = classify_album(_probe(0.55), cands, config)
    assert r.classification == AlbumClassification.MIXED
    assert "deterministic_threshold" in r.reason


def test_probe_at_deterministic_threshold_allowed(config):
    cands = {"/t.flac": _cand(confidence=0.9, proposed=True)}
    r = classify_album(_probe(config.probe.deterministic_threshold), cands, config)
    assert r.classification == AlbumClassification.DETERMINISTIC


def test_clean_still_requires_clean_threshold(config):
    # Probe at deterministic_threshold but no changes proposed → still not CLEAN
    # because the probe didn't clear the higher clean bar.
    cands = {"/t.flac": _cand(confidence=1.0, proposed=False)}
    r = classify_album(_probe(config.probe.deterministic_threshold), cands, config)
    assert r.classification != AlbumClassification.CLEAN


def test_classifier_uses_remaining_flags_not_original(config):
    # A candidate whose *original* flags included unreadable but whose cleanup
    # resolved them should not be labelled broken.
    from music_normalizer.models import DeterministicCandidate
    from pathlib import Path
    cand = DeterministicCandidate(
        track_path=Path("/t.flac"),
        proposed_title="Fixed",
        original_issue_flags=[IssueFlag.UNREADABLE_TITLE],
        remaining_issue_flags=[],
        issue_flags=[],
        confidence=0.9,
    )
    r = classify_album(_probe(0.8), {"/t.flac": cand}, config)
    assert r.classification == AlbumClassification.DETERMINISTIC
