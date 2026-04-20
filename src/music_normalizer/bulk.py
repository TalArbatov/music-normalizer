"""Bulk operations across every album under ``jobs_dir``.

Sits on top of the per-album commands (``suggest``, ``review``, ``apply``) and
provides higher-throughput flows for reviewing large libraries:

- :func:`export_pending` — one global CSV of everything still awaiting review.
- :func:`auto_approve`   — turn safe rows into approve-decisions without a human.
- :func:`review_all`     — one interactive queue spanning every album, with
  "approve/skip rest of album" shortcuts.
- :func:`apply_all`      — run the per-album apply across every album's
  approved decisions.

Decisions are always persisted into each album's own ``review_decisions.json``
so the existing per-album ``apply`` command continues to work unchanged.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .apply import apply_decisions
from .config import Config
from .models import (
    AUTO_APPROVE_NOTE,
    PENDING_TRACK_ACTIONS,
    AlbumSuggestions,
    IssueFlag,
    ReviewAction,
    ReviewDecision,
    Suggestion,
    SuggestionSource,
    TrackAction,
)
from .review import _load_prior_decisions, _render_suggestion
from .utils import dump_json, get_logger, load_json

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared iteration
# ---------------------------------------------------------------------------


@dataclass
class AlbumJob:
    """An album dir that has a suggestions.json ready for bulk processing."""

    album_dir: Path
    album_id: str
    suggestions: AlbumSuggestions


def iter_album_jobs(config: Config) -> list[AlbumJob]:
    """Return every album under ``jobs_dir`` that has a suggestions.json.

    Sorted by directory name for deterministic review order across sessions.
    """
    jobs_dir = config.jobs_dir
    if not jobs_dir.exists():
        return []
    out: list[AlbumJob] = []
    for album_dir in sorted(p for p in jobs_dir.iterdir() if p.is_dir()):
        sug_path = album_dir / "suggestions.json"
        if not sug_path.exists():
            continue
        try:
            album = AlbumSuggestions.model_validate(load_json(sug_path))
        except Exception as e:  # noqa: BLE001 — corrupt job dirs shouldn't abort the loop
            log.warning("could not parse %s: %s", sug_path, e)
            continue
        out.append(AlbumJob(album_dir=album_dir, album_id=album.album_id, suggestions=album))
    return out


# ---------------------------------------------------------------------------
# export-pending
# ---------------------------------------------------------------------------


_CSV_HEADER = (
    "album_id", "track_path", "source", "confidence",
    "current_title", "proposed_title",
    "current_artist", "proposed_artist",
    "issue_flags", "reason",
)


def export_pending(config: Config, out_path: Path) -> int:
    """Write a global CSV of every track still pending review. Returns row count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_HEADER)
        for job in iter_album_jobs(config):
            for s in job.suggestions.suggestions:
                if s.action not in PENDING_TRACK_ACTIONS:
                    continue
                writer.writerow((
                    job.album_id,
                    str(s.track_path),
                    s.source.value,
                    f"{s.confidence:.2f}",
                    s.current_title or "",
                    s.proposed_title or "",
                    s.current_artist or "",
                    s.proposed_artist or "",
                    ";".join(f.value for f in s.issue_flags),
                    s.reason,
                ))
                rows += 1
    return rows


# ---------------------------------------------------------------------------
# auto-approve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AutoApprovePolicy:
    """Configurable thresholds for the auto-approve pass.

    Defaults match the conservative policy from the brief:
    - only deterministic-sourced suggestions
    - confidence ≥ 0.95
    - action is propose_fix
    - none of the blocking flags are present
    """

    min_confidence: float = 0.95
    allow_llm: bool = False
    blocking_flags: frozenset[IssueFlag] = frozenset(
        {
            IssueFlag.UNREADABLE_TITLE,
            IssueFlag.GENERIC_TRACK_NAME,
            IssueFlag.SUSPICIOUS_SWAP_ARTIST_TITLE,
        }
    )


def _is_safe_for_auto_approve(s: Suggestion, policy: AutoApprovePolicy) -> bool:
    if s.action != TrackAction.PROPOSE_FIX:
        return False
    if s.source != SuggestionSource.DETERMINISTIC and not policy.allow_llm:
        return False
    if s.confidence < policy.min_confidence:
        return False
    # Prefer remaining flags (what's still wrong) but fall back to issue_flags
    # for older suggestions that predate the split.
    flags = set(s.remaining_issue_flags or s.issue_flags)
    if flags & policy.blocking_flags:
        return False
    return True


@dataclass
class AutoApproveResult:
    total_albums: int = 0
    tracks_considered: int = 0
    newly_approved: int = 0
    already_decided: int = 0


def auto_approve(
    config: Config, policy: AutoApprovePolicy | None = None
) -> AutoApproveResult:
    """Walk every album and auto-approve the rows that pass ``policy``.

    Never overwrites a pre-existing decision — rows already present in
    ``review_decisions.json`` are left alone so manual edits aren't clobbered.
    """
    policy = policy or AutoApprovePolicy()
    result = AutoApproveResult()

    for job in iter_album_jobs(config):
        result.total_albums += 1
        decisions_path = job.album_dir / "review_decisions.json"
        prior = _load_prior_decisions(decisions_path)
        new_count_before = result.newly_approved

        for s in job.suggestions.suggestions:
            if s.action not in PENDING_TRACK_ACTIONS:
                continue
            result.tracks_considered += 1
            key = str(s.track_path)
            if key in prior:
                result.already_decided += 1
                continue
            if not _is_safe_for_auto_approve(s, policy):
                continue
            prior[key] = ReviewDecision(
                track_path=s.track_path,
                action=ReviewAction.APPROVE,
                note=AUTO_APPROVE_NOTE,
            )
            result.newly_approved += 1

        # Only rewrite the decisions file when this album actually gained new
        # approvals. Avoids touching files we didn't change.
        if result.newly_approved > new_count_before:
            dump_json(
                [d.model_dump(mode="json") for d in prior.values()], decisions_path
            )

    return result


# ---------------------------------------------------------------------------
# review-all
# ---------------------------------------------------------------------------


class BulkReviewChoice(str, Enum):
    APPROVE = "a"
    SKIP = "s"
    EDIT = "e"
    APPROVE_REST_ALBUM = "A"
    SKIP_REST_ALBUM = "S"
    QUIT = "q"


_BULK_PROMPT = (
    "[bold]Action[/] ([green]a[/]pprove / [yellow]s[/]kip / [cyan]e[/]dit / "
    "[green bold]A[/]=approve rest / [yellow bold]S[/]=skip rest / "
    "[magenta]q[/]uit)"
)


def _prompt_bulk(console: Console) -> BulkReviewChoice:  # noqa: ARG001 — console reserved for future use
    choice = Prompt.ask(
        _BULK_PROMPT,
        choices=[c.value for c in BulkReviewChoice],
        default="s",
    )
    return BulkReviewChoice(choice)


def _prompt_edit_values(s: Suggestion) -> ReviewDecision:
    new_title = Prompt.ask("title", default=s.proposed_title or "")
    new_artist = Prompt.ask("artist", default=s.proposed_artist or "")
    note = Prompt.ask("note (optional)", default="")
    return ReviewDecision(
        track_path=s.track_path,
        action=ReviewAction.EDIT,
        final_title=new_title,
        final_artist=new_artist,
        note=note or None,
    )


def review_all(config: Config, *, console: Console | None = None) -> int:
    """Interactive queue across every album. Returns total decisions written.

    Decisions are persisted to each album's ``review_decisions.json`` after
    every track, so quitting never loses completed work. Already-decided tracks
    are skipped on resume.
    """
    console = console or Console()
    total = 0

    for job in iter_album_jobs(config):
        decisions_path = job.album_dir / "review_decisions.json"
        prior = _load_prior_decisions(decisions_path)
        already = set(prior.keys())
        pending = [
            s
            for s in job.suggestions.suggestions
            if s.action in PENDING_TRACK_ACTIONS and str(s.track_path) not in already
        ]
        if not pending:
            continue

        console.print(
            Panel.fit(
                f"[bold]Album {job.album_id}[/]\n"
                f"pending: {len(pending)}  "
                f"classification: {job.suggestions.classification.value}  "
                f"probe: {job.suggestions.probe_score:.2f}",
                title="Bulk review",
            )
        )
        decisions = list(prior.values())
        cascade: ReviewAction | None = None  # set when user chooses "rest of album"

        def _save() -> None:
            dump_json([d.model_dump(mode="json") for d in decisions], decisions_path)

        try:
            for idx, s in enumerate(pending, start=1):
                if cascade is not None:
                    decisions.append(
                        ReviewDecision(
                            track_path=s.track_path,
                            action=cascade,
                            note="bulk: rest of album",
                        )
                    )
                    total += 1
                    _save()
                    continue

                _render_suggestion(console, s, idx, len(pending))
                choice = _prompt_bulk(console)

                if choice == BulkReviewChoice.QUIT:
                    _save()
                    console.print(
                        f"[yellow]quit; {total} decisions saved across albums[/]"
                    )
                    return total

                if choice == BulkReviewChoice.APPROVE_REST_ALBUM:
                    decisions.append(
                        ReviewDecision(
                            track_path=s.track_path,
                            action=ReviewAction.APPROVE,
                            note="bulk: approve rest of album",
                        )
                    )
                    cascade = ReviewAction.APPROVE
                elif choice == BulkReviewChoice.SKIP_REST_ALBUM:
                    decisions.append(
                        ReviewDecision(
                            track_path=s.track_path,
                            action=ReviewAction.SKIP,
                            note="bulk: skip rest of album",
                        )
                    )
                    cascade = ReviewAction.SKIP
                elif choice == BulkReviewChoice.APPROVE:
                    decisions.append(
                        ReviewDecision(track_path=s.track_path, action=ReviewAction.APPROVE)
                    )
                elif choice == BulkReviewChoice.SKIP:
                    decisions.append(
                        ReviewDecision(track_path=s.track_path, action=ReviewAction.SKIP)
                    )
                elif choice == BulkReviewChoice.EDIT:
                    decisions.append(_prompt_edit_values(s))

                total += 1
                _save()
        except KeyboardInterrupt:
            _save()
            console.print(
                f"[yellow]interrupted; {total} decisions saved across albums[/]"
            )
            return total

    console.print(f"[green]bulk review done: {total} decisions saved[/]")
    return total


# ---------------------------------------------------------------------------
# apply-all
# ---------------------------------------------------------------------------


@dataclass
class ApplyAllResult:
    albums_applied: int = 0
    albums_no_decisions: int = 0
    tracks_ok: int = 0
    tracks_failed: int = 0


def apply_all(config: Config, *, dry_run: bool) -> ApplyAllResult:
    """Run the per-album apply across every album that has decisions on disk."""
    result = ApplyAllResult()
    for job in iter_album_jobs(config):
        decisions_path = job.album_dir / "review_decisions.json"
        if not decisions_path.exists():
            result.albums_no_decisions += 1
            continue
        result.albums_applied += 1
        results = apply_decisions(
            suggestions_path=job.album_dir / "suggestions.json",
            decisions_path=decisions_path,
            log_path=job.album_dir / "apply.log.json",
            dry_run=dry_run,
        )
        result.tracks_ok += sum(1 for r in results if r.success)
        result.tracks_failed += sum(1 for r in results if not r.success)
    return result
