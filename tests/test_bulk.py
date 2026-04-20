"""Bulk workflow: export-pending, auto-approve, review-all, apply-all, report counts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rich.console import Console

from music_normalizer.bulk import (
    AutoApprovePolicy,
    apply_all,
    auto_approve,
    export_pending,
    iter_album_jobs,
    review_all,
)
from music_normalizer.models import (
    AUTO_APPROVE_NOTE,
    AlbumClassification,
    AlbumSuggestions,
    IssueFlag,
    Suggestion,
    SuggestionSource,
    TrackAction,
)
from music_normalizer.report import build_report
from music_normalizer.utils import dump_json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sug(
    track_path: Path,
    *,
    action: TrackAction = TrackAction.PROPOSE_FIX,
    source: SuggestionSource = SuggestionSource.DETERMINISTIC,
    confidence: float = 0.98,
    remaining_flags: list[IssueFlag] | None = None,
    current_title: str = "old",
    proposed_title: str = "New",
    current_artist: str = "olda",
    proposed_artist: str = "New A",
) -> Suggestion:
    return Suggestion(
        track_path=track_path,
        current_title=current_title,
        current_artist=current_artist,
        proposed_title=proposed_title,
        proposed_artist=proposed_artist,
        action=action,
        confidence=confidence,
        source=source,
        remaining_issue_flags=remaining_flags or [],
        issue_flags=remaining_flags or [],
    )


def _make_album(
    jobs_dir: Path, album_id: str, suggestions: list[Suggestion],
    classification: AlbumClassification = AlbumClassification.DETERMINISTIC,
) -> Path:
    album_dir = jobs_dir / album_id
    album_dir.mkdir(parents=True, exist_ok=True)
    album = AlbumSuggestions(
        album_id=album_id,
        classification=classification,
        probe_score=0.85,
        classification_reason="test",
        used_llm=False,
        suggestions=suggestions,
    )
    dump_json(album.model_dump(mode="json"), album_dir / "suggestions.json")
    return album_dir


def _config_with_jobs(tmp_path: Path, config):
    # Point the fixture config at a tmp jobs dir so tests are isolated.
    config.jobs_dir = tmp_path / "jobs"
    config.jobs_dir.mkdir(parents=True, exist_ok=True)
    return config


# ---------------------------------------------------------------------------
# iter_album_jobs
# ---------------------------------------------------------------------------


def test_iter_album_jobs_sorted_and_skips_dirs_without_suggestions(tmp_path, config):
    config = _config_with_jobs(tmp_path, config)
    _make_album(config.jobs_dir, "b_album", [_sug(tmp_path / "b.flac")])
    _make_album(config.jobs_dir, "a_album", [_sug(tmp_path / "a.flac")])
    # A dir with no suggestions.json must be skipped.
    (config.jobs_dir / "empty").mkdir()

    jobs = iter_album_jobs(config)
    assert [j.album_id for j in jobs] == ["a_album", "b_album"]


# ---------------------------------------------------------------------------
# export-pending
# ---------------------------------------------------------------------------


def test_export_pending_writes_only_pending_rows(tmp_path, config):
    config = _config_with_jobs(tmp_path, config)
    _make_album(config.jobs_dir, "a1", [
        _sug(tmp_path / "a1_1.flac", action=TrackAction.PROPOSE_FIX),
        _sug(tmp_path / "a1_2.flac", action=TrackAction.KEEP),
        _sug(tmp_path / "a1_3.flac", action=TrackAction.UNREADABLE_NEEDS_MANUAL_REVIEW),
    ])
    _make_album(config.jobs_dir, "a2", [
        _sug(tmp_path / "a2_1.flac", action=TrackAction.KEEP),
    ])

    out = tmp_path / "pending.csv"
    rows = export_pending(config, out)
    assert rows == 2

    with out.open() as fh:
        lines = list(csv.reader(fh))
    assert lines[0] == [
        "album_id", "track_path", "source", "confidence",
        "current_title", "proposed_title",
        "current_artist", "proposed_artist",
        "issue_flags", "reason",
    ]
    paths = {row[1] for row in lines[1:]}
    assert paths == {str(tmp_path / "a1_1.flac"), str(tmp_path / "a1_3.flac")}


def test_export_pending_empty_when_no_albums(tmp_path, config):
    config = _config_with_jobs(tmp_path, config)
    out = tmp_path / "empty.csv"
    assert export_pending(config, out) == 0
    assert out.exists()
    # Header-only file.
    with out.open() as fh:
        assert len(list(csv.reader(fh))) == 1


# ---------------------------------------------------------------------------
# auto-approve
# ---------------------------------------------------------------------------


def test_auto_approve_default_policy(tmp_path, config):
    config = _config_with_jobs(tmp_path, config)
    safe = _sug(tmp_path / "safe.flac", confidence=0.98)
    low = _sug(tmp_path / "low.flac", confidence=0.7)
    llm = _sug(tmp_path / "llm.flac", source=SuggestionSource.LLM)
    blocked = _sug(
        tmp_path / "blocked.flac",
        remaining_flags=[IssueFlag.UNREADABLE_TITLE],
    )
    unreadable = _sug(
        tmp_path / "unread.flac",
        action=TrackAction.UNREADABLE_NEEDS_MANUAL_REVIEW,
    )

    album_dir = _make_album(
        config.jobs_dir, "album1", [safe, low, llm, blocked, unreadable]
    )

    result = auto_approve(config)

    assert result.newly_approved == 1
    assert result.already_decided == 0
    assert result.tracks_considered == 5

    decisions_path = album_dir / "review_decisions.json"
    decisions = json.loads(decisions_path.read_text())
    assert len(decisions) == 1
    d = decisions[0]
    assert d["track_path"] == str(tmp_path / "safe.flac")
    assert d["action"] == "approve"
    assert d["note"] == AUTO_APPROVE_NOTE


def test_auto_approve_respects_min_confidence_flag(tmp_path, config):
    config = _config_with_jobs(tmp_path, config)
    _make_album(config.jobs_dir, "a", [
        _sug(tmp_path / "x.flac", confidence=0.8),
    ])
    # Default policy (>=0.95) rejects it.
    assert auto_approve(config).newly_approved == 0
    # Lower threshold accepts it.
    result = auto_approve(config, AutoApprovePolicy(min_confidence=0.7))
    assert result.newly_approved == 1


def test_auto_approve_allow_llm_flag(tmp_path, config):
    config = _config_with_jobs(tmp_path, config)
    _make_album(config.jobs_dir, "a", [
        _sug(tmp_path / "x.flac", source=SuggestionSource.LLM, confidence=0.99),
    ])
    assert auto_approve(config).newly_approved == 0
    result = auto_approve(config, AutoApprovePolicy(allow_llm=True))
    assert result.newly_approved == 1


def test_auto_approve_does_not_overwrite_existing_decisions(tmp_path, config):
    config = _config_with_jobs(tmp_path, config)
    safe_path = tmp_path / "safe.flac"
    album_dir = _make_album(config.jobs_dir, "a", [_sug(safe_path, confidence=0.99)])

    # Pre-existing skip decision that a human made.
    prior = [{
        "track_path": str(safe_path),
        "action": "skip",
        "final_title": None, "final_artist": None,
        "decided_at": "2026-04-20T00:00:00",
        "note": "reviewed by hand",
    }]
    (album_dir / "review_decisions.json").write_text(json.dumps(prior))

    result = auto_approve(config)
    assert result.newly_approved == 0
    assert result.already_decided == 1

    # The human skip must survive intact.
    decisions = json.loads((album_dir / "review_decisions.json").read_text())
    assert len(decisions) == 1
    assert decisions[0]["action"] == "skip"
    assert decisions[0]["note"] == "reviewed by hand"


# ---------------------------------------------------------------------------
# review-all
# ---------------------------------------------------------------------------


def _null_console() -> Console:
    return Console(file=open("/dev/null", "w"))


def test_review_all_spans_multiple_albums(tmp_path, config, monkeypatch):
    config = _config_with_jobs(tmp_path, config)
    a1_dir = _make_album(config.jobs_dir, "a1", [
        _sug(tmp_path / "a1_1.flac"),
        _sug(tmp_path / "a1_2.flac"),
    ])
    a2_dir = _make_album(config.jobs_dir, "a2", [
        _sug(tmp_path / "a2_1.flac"),
    ])
    calls = iter(["a", "s", "a"])
    monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **k: next(calls))

    total = review_all(config, console=_null_console())
    assert total == 3

    d1 = json.loads((a1_dir / "review_decisions.json").read_text())
    d2 = json.loads((a2_dir / "review_decisions.json").read_text())
    actions1 = [d["action"] for d in d1]
    actions2 = [d["action"] for d in d2]
    assert actions1 == ["approve", "skip"]
    assert actions2 == ["approve"]


def test_review_all_approve_rest_of_album_cascades(tmp_path, config, monkeypatch):
    config = _config_with_jobs(tmp_path, config)
    album_dir = _make_album(config.jobs_dir, "a", [
        _sug(tmp_path / f"t{i}.flac") for i in range(5)
    ])
    # First track: "A" = approve rest of album. No further prompts expected.
    calls = iter(["A"])
    prompted = {"n": 0}

    def fake(*args, **kwargs):
        prompted["n"] += 1
        return next(calls)

    monkeypatch.setattr("rich.prompt.Prompt.ask", fake)

    total = review_all(config, console=_null_console())
    assert prompted["n"] == 1
    assert total == 5
    decisions = json.loads((album_dir / "review_decisions.json").read_text())
    assert len(decisions) == 5
    assert all(d["action"] == "approve" for d in decisions)
    # Cascade decisions should carry a bulk note.
    assert any("rest of album" in (d.get("note") or "") for d in decisions)


def test_review_all_skip_rest_of_album_cascades(tmp_path, config, monkeypatch):
    config = _config_with_jobs(tmp_path, config)
    album_dir = _make_album(config.jobs_dir, "a", [
        _sug(tmp_path / f"t{i}.flac") for i in range(3)
    ])
    calls = iter(["S"])
    monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **k: next(calls))

    total = review_all(config, console=_null_console())
    assert total == 3
    decisions = json.loads((album_dir / "review_decisions.json").read_text())
    assert [d["action"] for d in decisions] == ["skip", "skip", "skip"]


def test_review_all_quit_preserves_progress(tmp_path, config, monkeypatch):
    config = _config_with_jobs(tmp_path, config)
    a1_dir = _make_album(config.jobs_dir, "a1", [
        _sug(tmp_path / "a.flac"),
        _sug(tmp_path / "b.flac"),
    ])
    _make_album(config.jobs_dir, "a2", [_sug(tmp_path / "c.flac")])
    calls = iter(["a", "q"])
    monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **k: next(calls))

    total = review_all(config, console=_null_console())
    assert total == 1

    d1 = json.loads((a1_dir / "review_decisions.json").read_text())
    assert len(d1) == 1
    assert d1[0]["action"] == "approve"
    # a2 never got started, no decisions file.
    assert not (config.jobs_dir / "a2" / "review_decisions.json").exists()


def test_review_all_skips_already_decided(tmp_path, config, monkeypatch):
    config = _config_with_jobs(tmp_path, config)
    p1 = tmp_path / "p1.flac"
    p2 = tmp_path / "p2.flac"
    album_dir = _make_album(config.jobs_dir, "a", [_sug(p1), _sug(p2)])

    (album_dir / "review_decisions.json").write_text(json.dumps([{
        "track_path": str(p1), "action": "approve",
        "final_title": None, "final_artist": None,
        "decided_at": "2026-04-20T00:00:00", "note": None,
    }]))

    calls = iter(["s"])  # only one prompt expected
    prompted = {"n": 0}

    def fake(*a, **k):
        prompted["n"] += 1
        return next(calls)

    monkeypatch.setattr("rich.prompt.Prompt.ask", fake)
    review_all(config, console=_null_console())
    assert prompted["n"] == 1


# ---------------------------------------------------------------------------
# apply-all
# ---------------------------------------------------------------------------


def test_apply_all_dry_run_counts(tmp_path, config):
    config = _config_with_jobs(tmp_path, config)
    # Create real FLAC-like stub paths. apply uses dry_run so files don't need to exist.
    p1 = tmp_path / "t1.flac"
    p2 = tmp_path / "t2.flac"

    a_dir = _make_album(config.jobs_dir, "a", [_sug(p1), _sug(p2)])
    (a_dir / "review_decisions.json").write_text(json.dumps([
        {"track_path": str(p1), "action": "approve",
         "final_title": None, "final_artist": None,
         "decided_at": "2026-04-20T00:00:00", "note": None},
        {"track_path": str(p2), "action": "skip",
         "final_title": None, "final_artist": None,
         "decided_at": "2026-04-20T00:00:00", "note": None},
    ]))

    # Second album has no decisions — must be counted as albums_no_decisions.
    _make_album(config.jobs_dir, "b", [_sug(tmp_path / "t3.flac")])

    result = apply_all(config, dry_run=True)
    assert result.albums_applied == 1
    assert result.albums_no_decisions == 1
    assert result.tracks_ok == 2  # approve + skip both count as success in dry-run
    assert result.tracks_failed == 0

    # apply.log.json is written for the applied album.
    assert (a_dir / "apply.log.json").exists()


# ---------------------------------------------------------------------------
# report with decision counts
# ---------------------------------------------------------------------------


def test_report_counts_decisions(tmp_path, config):
    config = _config_with_jobs(tmp_path, config)
    p1, p2, p3, p4 = [tmp_path / f"t{i}.flac" for i in range(1, 5)]
    a_dir = _make_album(config.jobs_dir, "a", [
        _sug(p1, action=TrackAction.PROPOSE_FIX),
        _sug(p2, action=TrackAction.PROPOSE_FIX),
        _sug(p3, action=TrackAction.PROPOSE_FIX),
        _sug(p4, action=TrackAction.PROPOSE_FIX),
    ])
    (a_dir / "review_decisions.json").write_text(json.dumps([
        {"track_path": str(p1), "action": "approve",
         "final_title": None, "final_artist": None,
         "decided_at": "2026-04-20T00:00:00", "note": None},
        {"track_path": str(p2), "action": "approve",
         "final_title": None, "final_artist": None,
         "decided_at": "2026-04-20T00:00:00", "note": AUTO_APPROVE_NOTE},
        {"track_path": str(p3), "action": "skip",
         "final_title": None, "final_artist": None,
         "decided_at": "2026-04-20T00:00:00", "note": None},
        # p4 deliberately left undecided → pending_review should count 1.
    ]))

    report = build_report(config)
    assert report.approved == 2
    assert report.auto_approved == 1
    assert report.skipped == 1
    assert report.edited == 0
    assert report.pending_review == 1
