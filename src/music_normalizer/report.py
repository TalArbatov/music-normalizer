"""Aggregate job artifacts into a single :class:`ReportSummary`.

This is a pure read operation over the jobs directory — it does not re-scan the
library or call any network services.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .config import Config
from .models import (
    AUTO_APPROVE_NOTE,
    PENDING_TRACK_ACTIONS,
    AlbumClassification,
    AlbumSuggestions,
    ReportSummary,
    TrackAction,
)
from .utils import get_logger, load_json

log = get_logger(__name__)


def build_report(config: Config) -> ReportSummary:
    summary = ReportSummary()
    flag_counter: Counter[str] = Counter()

    jobs_dir = config.jobs_dir
    if not jobs_dir.exists():
        return summary

    for album_dir in sorted(jobs_dir.iterdir()):
        if not album_dir.is_dir():
            continue
        sug_path = album_dir / "suggestions.json"
        if not sug_path.exists():
            # Manifest exists but no suggestions yet → treat as skipped-unchanged.
            manifest_path = album_dir / "manifest.json"
            if manifest_path.exists():
                summary.skipped_unchanged += 1
                summary.total_albums += 1
            continue

        try:
            album = AlbumSuggestions.model_validate(load_json(sug_path))
        except Exception as e:  # noqa: BLE001
            log.warning("could not parse %s: %s", sug_path, e)
            continue

        summary.total_albums += 1
        summary.total_tracks += len(album.suggestions)

        match album.classification:
            case AlbumClassification.CLEAN:
                summary.clean_albums += 1
            case AlbumClassification.DETERMINISTIC:
                summary.deterministic_albums += 1
            case AlbumClassification.MIXED:
                summary.llm_albums += 1 if album.used_llm else 0
            case AlbumClassification.BROKEN:
                summary.broken_albums += 1

        for s in album.suggestions:
            for f in s.issue_flags:
                flag_counter[f.value] += 1
            if s.action == TrackAction.PROPOSE_FIX:
                summary.proposed_changes += 1
            if s.action == TrackAction.UNREADABLE_NEEDS_MANUAL_REVIEW:
                summary.unreadable += 1
            if s.confidence < config.confidence.low_confidence_flag and s.action == TrackAction.PROPOSE_FIX:
                summary.low_confidence += 1

        # Decision-level counts from review_decisions.json if present.
        decisions_path = album_dir / "review_decisions.json"
        decided_paths: set[str] = set()
        if decisions_path.exists():
            try:
                for d in load_json(decisions_path) or []:
                    decided_paths.add(str(d.get("track_path") or ""))
                    action = d.get("action")
                    note = d.get("note") or ""
                    if action == "approve":
                        summary.approved += 1
                        if note == AUTO_APPROVE_NOTE:
                            summary.auto_approved += 1
                    elif action == "skip":
                        summary.skipped += 1
                    elif action == "edit":
                        summary.edited += 1
            except Exception as e:  # noqa: BLE001
                log.warning("could not parse %s: %s", decisions_path, e)

        # Pending review = pending-action suggestions that have no decision yet.
        for s in album.suggestions:
            if (
                s.action in PENDING_TRACK_ACTIONS
                and str(s.track_path) not in decided_paths
            ):
                summary.pending_review += 1

    summary.top_issues = dict(flag_counter.most_common(15))
    return summary
