"""Apply approved review decisions to the underlying audio files.

For each decision:
- ``approve`` → write the suggested title/artist.
- ``edit``    → write the reviewer's edited values.
- ``skip``    → nothing written, but logged for the audit trail.

The full audit is always written, even in dry-run, so a reviewer can inspect
exactly what would have changed.
"""

from __future__ import annotations

from pathlib import Path

from .models import (
    AlbumSuggestions,
    ApplyResult,
    ReviewAction,
    ReviewDecision,
    Suggestion,
    TrackAction,
)
from .tags import write_title_artist
from .utils import dump_json, get_logger, load_json

log = get_logger(__name__)


def apply_decisions(
    suggestions_path: Path,
    decisions_path: Path,
    log_path: Path,
    *,
    dry_run: bool = True,
) -> list[ApplyResult]:
    """Execute approved changes. Returns one :class:`ApplyResult` per decision."""
    album = AlbumSuggestions.model_validate(load_json(suggestions_path))
    decisions_raw = load_json(decisions_path)
    decisions = [ReviewDecision.model_validate(d) for d in decisions_raw]

    sug_by_path: dict[str, Suggestion] = {str(s.track_path): s for s in album.suggestions}
    results: list[ApplyResult] = []

    for decision in decisions:
        sug = sug_by_path.get(str(decision.track_path))
        if sug is None:
            log.warning("no suggestion for %s, skipping", decision.track_path)
            continue

        if decision.action == ReviewAction.SKIP:
            results.append(ApplyResult(
                track_path=decision.track_path,
                applied_title=None,
                applied_artist=None,
                previous_title=sug.current_title,
                previous_artist=sug.current_artist,
                success=True,
                dry_run=dry_run,
                error="skipped by reviewer",
            ))
            continue

        final_title, final_artist = _resolve_final(decision, sug)

        # Only emit the field if it actually changed — this is what lets the audit log
        # show a single-field edit as exactly that, not a double-write.
        title_to_write = final_title if final_title != sug.current_title else None
        artist_to_write = final_artist if final_artist != sug.current_artist else None

        if title_to_write is None and artist_to_write is None:
            results.append(ApplyResult(
                track_path=decision.track_path,
                applied_title=None,
                applied_artist=None,
                previous_title=sug.current_title,
                previous_artist=sug.current_artist,
                success=True,
                dry_run=dry_run,
                error="no change needed after edit",
            ))
            continue

        try:
            write_title_artist(
                decision.track_path,
                title=title_to_write,
                artist=artist_to_write,
                dry_run=dry_run,
            )
            results.append(ApplyResult(
                track_path=decision.track_path,
                applied_title=title_to_write,
                applied_artist=artist_to_write,
                previous_title=sug.current_title,
                previous_artist=sug.current_artist,
                success=True,
                dry_run=dry_run,
            ))
        except Exception as e:  # noqa: BLE001 — we want every failure recorded in the audit
            log.error("apply failed for %s: %s", decision.track_path, e)
            results.append(ApplyResult(
                track_path=decision.track_path,
                applied_title=title_to_write,
                applied_artist=artist_to_write,
                previous_title=sug.current_title,
                previous_artist=sug.current_artist,
                success=False,
                error=str(e),
                dry_run=dry_run,
            ))

    dump_json([r.model_dump(mode="json") for r in results], log_path)
    return results


def _resolve_final(d: ReviewDecision, s: Suggestion) -> tuple[str | None, str | None]:
    if d.action == ReviewAction.EDIT:
        return d.final_title, d.final_artist
    # APPROVE: use the suggestion's proposed values. If the suggestion's action
    # is UNREADABLE_NEEDS_MANUAL_REVIEW, approving it is a no-op (we keep current).
    if s.action == TrackAction.UNREADABLE_NEEDS_MANUAL_REVIEW:
        return s.current_title, s.current_artist
    return s.proposed_title, s.proposed_artist
