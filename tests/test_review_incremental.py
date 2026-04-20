"""Review saves after each decision and resumes where it left off."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from music_normalizer.models import (
    AlbumClassification,
    AlbumSuggestions,
    IssueFlag,
    Suggestion,
    SuggestionSource,
    TrackAction,
)
from music_normalizer.review import review_album
from music_normalizer.utils import dump_json


def _mk_album(tmp_path: Path) -> Path:
    sug = AlbumSuggestions(
        album_id="abc123",
        classification=AlbumClassification.DETERMINISTIC,
        probe_score=0.8,
        classification_reason="test",
        used_llm=False,
        suggestions=[
            Suggestion(
                track_path=tmp_path / f"t{i}.flac",
                current_title=f"t{i}", current_artist="a",
                proposed_title=f"T{i}", proposed_artist="A",
                action=TrackAction.PROPOSE_FIX,
                confidence=0.9, source=SuggestionSource.DETERMINISTIC,
                remaining_issue_flags=[IssueFlag.BAD_CAPITALIZATION],
            )
            for i in range(3)
        ],
    )
    sug_path = tmp_path / "suggestions.json"
    dump_json(sug.model_dump(mode="json"), sug_path)
    return sug_path


def test_incremental_save_persists_on_each_track(tmp_path, monkeypatch):
    sug_path = _mk_album(tmp_path)
    out_path = tmp_path / "review_decisions.json"

    # Capture decisions file contents after each Rich prompt call.
    snapshots: list[int] = []
    calls = iter(["a", "a", "a"])

    def fake_ask(*_args, **_kwargs):
        # Record how many decisions are persisted BEFORE this prompt.
        if out_path.exists():
            snapshots.append(len(json.loads(out_path.read_text())))
        else:
            snapshots.append(0)
        return next(calls)

    monkeypatch.setattr("rich.prompt.Prompt.ask", fake_ask)

    review_album(sug_path, out_path, console=Console(file=open("/dev/null", "w")))

    # Before each of the 3 prompts, N-1 decisions should already be on disk.
    assert snapshots == [0, 1, 2]
    # And at the end the file has all 3.
    assert len(json.loads(out_path.read_text())) == 3


def test_quit_preserves_completed_work(tmp_path, monkeypatch):
    sug_path = _mk_album(tmp_path)
    out_path = tmp_path / "review_decisions.json"

    calls = iter(["a", "q"])
    monkeypatch.setattr("rich.prompt.Prompt.ask",
                        lambda *a, **kw: next(calls))

    review_album(sug_path, out_path, console=Console(file=open("/dev/null", "w")))

    # First decision got saved before the quit on the second prompt.
    assert out_path.exists()
    decisions = json.loads(out_path.read_text())
    assert len(decisions) == 1
    assert decisions[0]["action"] == "approve"


def test_resume_skips_already_decided(tmp_path, monkeypatch):
    sug_path = _mk_album(tmp_path)
    out_path = tmp_path / "review_decisions.json"

    # Pre-populate decisions for the first two tracks.
    prior = [
        {"track_path": str(tmp_path / "t0.flac"), "action": "approve",
         "final_title": None, "final_artist": None,
         "decided_at": "2026-04-20T00:00:00", "note": None},
        {"track_path": str(tmp_path / "t1.flac"), "action": "skip",
         "final_title": None, "final_artist": None,
         "decided_at": "2026-04-20T00:00:00", "note": None},
    ]
    out_path.write_text(json.dumps(prior))

    calls = iter(["a"])  # only one prompt expected for the last undecided track
    prompted = {"count": 0}

    def fake_ask(*_a, **_kw):
        prompted["count"] += 1
        return next(calls)

    monkeypatch.setattr("rich.prompt.Prompt.ask", fake_ask)

    review_album(sug_path, out_path, console=Console(file=open("/dev/null", "w")))

    assert prompted["count"] == 1
    final = json.loads(out_path.read_text())
    assert {d["track_path"] for d in final} == {
        str(tmp_path / f"t{i}.flac") for i in range(3)
    }
