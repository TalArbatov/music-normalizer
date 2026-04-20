"""Terminal review flow.

Loads an album's ``suggestions.json``, walks the user through each proposed change,
and writes ``review_decisions.json``. Only tracks where ``action == propose_fix``
or ``unreadable_needs_manual_review`` are shown — tracks marked ``keep`` are
auto-skipped unless ``--all`` is passed.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .models import (
    AlbumSuggestions,
    ReviewAction,
    ReviewDecision,
    Suggestion,
    TrackAction,
)
from .utils import dump_json, get_logger, load_json

log = get_logger(__name__)


def review_album(
    suggestions_path: Path,
    out_path: Path,
    *,
    show_all: bool = False,
    console: Console | None = None,
) -> list[ReviewDecision]:
    """Interactively review one album's suggestions.

    Decisions are persisted to ``out_path`` **after every track** so that if the
    reviewer quits or the process crashes, all completed work is preserved.
    Existing decisions at ``out_path`` are loaded up front and used to skip
    already-reviewed tracks (resumable review).
    """
    console = console or Console()
    album = AlbumSuggestions.model_validate(load_json(suggestions_path))

    # Resume from any prior session.
    prior = _load_prior_decisions(out_path)
    decisions: list[ReviewDecision] = list(prior.values())
    already_done = set(prior.keys())

    console.print(Panel.fit(
        f"[bold]Album {album.album_id}[/]\n"
        f"classification: {album.classification.value}  "
        f"probe: {album.probe_score:.2f}  "
        f"used_llm: {album.used_llm}\n"
        f"reason: {album.classification_reason}"
        + (f"\n[yellow]resuming: {len(prior)} prior decisions loaded[/]" if prior else ""),
        title="Review",
    ))

    queue_full = [s for s in album.suggestions if show_all or s.action != TrackAction.KEEP]
    queue = [s for s in queue_full if str(s.track_path) not in already_done]
    total = len(queue)
    if total == 0:
        console.print("[green]Nothing to review — album is clean (or fully reviewed).[/]")
        dump_json([d.model_dump(mode="json") for d in decisions], out_path)
        return decisions

    try:
        for idx, s in enumerate(queue, start=1):
            _render_suggestion(console, s, idx, total)
            decision = _prompt_for_decision(console, s)
            decisions.append(decision)
            # Persist immediately — never lose a completed verdict to a crash / quit.
            dump_json([d.model_dump(mode="json") for d in decisions], out_path)
    except KeyboardInterrupt:
        console.print(
            f"[yellow]review interrupted; {len(decisions)} decisions saved → {out_path}[/]"
        )
        return decisions

    console.print(f"[green]Saved {len(decisions)} decisions → {out_path}[/]")
    return decisions


def _load_prior_decisions(path: Path) -> dict[str, ReviewDecision]:
    if not path.exists():
        return {}
    try:
        raw = load_json(path)
    except Exception as e:  # noqa: BLE001
        log.warning("could not parse prior decisions at %s: %s", path, e)
        return {}
    out: dict[str, ReviewDecision] = {}
    for item in raw or []:
        try:
            d = ReviewDecision.model_validate(item)
            out[str(d.track_path)] = d
        except Exception as e:  # noqa: BLE001
            log.warning("skipping unparseable prior decision: %s", e)
    return out


def _render_suggestion(console: Console, s: Suggestion, idx: int, total: int) -> None:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim", width=12)
    table.add_column()

    table.add_row("track", str(s.track_path))
    table.add_row("title (cur)", _repr(s.current_title))
    table.add_row("title (new)", _repr(s.proposed_title))
    table.add_row("artist (cur)", _repr(s.current_artist))
    table.add_row("artist (new)", _repr(s.proposed_artist))
    table.add_row("action", s.action.value)
    table.add_row("source", s.source.value)
    table.add_row("confidence", f"{s.confidence:.2f}")
    if s.original_issue_flags:
        table.add_row("orig flags", ", ".join(f.value for f in s.original_issue_flags))
    if s.remaining_issue_flags:
        table.add_row("remaining", ", ".join(f.value for f in s.remaining_issue_flags))
    elif s.issue_flags and not s.original_issue_flags:
        # Back-compat for suggestions written before the split existed.
        table.add_row("flags", ", ".join(f.value for f in s.issue_flags))
    if s.reason:
        table.add_row("reason", s.reason)
    if s.validator_notes:
        table.add_row("validator", "; ".join(s.validator_notes))
    console.print(Panel(table, title=f"[{idx}/{total}] review"))


def _repr(v: str | None) -> str:
    if v is None:
        return "[dim]∅[/]"
    if v == "":
        return "[red]<empty>[/]"
    return v


def _prompt_for_decision(console: Console, s: Suggestion) -> ReviewDecision:
    while True:
        choice = Prompt.ask(
            "[bold]Action[/] ([green]a[/]pprove / [yellow]s[/]kip / [cyan]e[/]dit / [magenta]q[/]uit)",
            choices=["a", "s", "e", "q"],
            default="s",
        )
        if choice == "q":
            raise KeyboardInterrupt("user quit review")
        if choice == "a":
            return ReviewDecision(track_path=s.track_path, action=ReviewAction.APPROVE)
        if choice == "s":
            return ReviewDecision(track_path=s.track_path, action=ReviewAction.SKIP)
        if choice == "e":
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
