"""Typer CLI. Each command is a thin wrapper over a pipeline function.

Per-album commands:
  scan     — walk staging, group into albums, write manifests
  suggest  — run probe + rules + (conditionally) LLM → suggestions.json
  review   — interactive TUI over suggestions.json → review_decisions.json
  apply    — write approved changes to tags (dry-run by default)
  report   — aggregate counts over the jobs directory

Bulk commands (operate across all albums under jobs_dir):
  export-pending — CSV of every track still pending review
  auto-approve   — approve rows meeting a safe-policy threshold
  review-all     — one interactive queue across every album
  apply-all      — apply approved decisions album by album
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .apply import apply_decisions
from .bulk import (
    AutoApprovePolicy,
    apply_all,
    auto_approve,
    export_pending,
    review_all,
)
from .config import Config, default_config_path, load_config
from .grouper import group_into_albums
from .models import AlbumManifest, AlbumSuggestions
from .ollama_client import OllamaClient
from .report import build_report
from .review import review_album
from .scanner import scan_tracks
from .suggestions import suggest_for_album
from .utils import dump_json, ensure_dir, get_logger, load_json

app = typer.Typer(
    name="music-normalizer",
    help="Review-first CLI for cleaning up messy TITLE/ARTIST tags.",
    no_args_is_help=True,
    add_completion=False,
)

log = get_logger("music_normalizer.cli")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(config_path: Path) -> Config:
    if not config_path.exists():
        raise typer.BadParameter(
            f"config file not found: {config_path}\n"
            f"Copy config.example.yaml to {config_path} and edit it."
        )
    return load_config(config_path)


def _album_dir(config: Config, album_id: str) -> Path:
    return ensure_dir(config.jobs_dir / album_id)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


@app.command()
def scan(
    config_path: Path = typer.Option(default_config_path(), "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Traverse the staging root and write one manifest.json per album."""
    _configure_logging(verbose)
    config = _load(config_path)
    console = Console()

    tracks = scan_tracks(config.staging_root, config.supported_extensions)
    albums = group_into_albums(tracks, staging_root=config.staging_root)

    skipped = 0
    written = 0
    for album in albums:
        album_dir = _album_dir(config, album.album_id)
        manifest_path = album_dir / "manifest.json"

        if config.cache.skip_unchanged_albums and manifest_path.exists():
            prev = AlbumManifest.model_validate(load_json(manifest_path))
            if prev.fingerprint == album.fingerprint:
                skipped += 1
                continue

        dump_json(album.model_dump(mode="json"), manifest_path)
        written += 1

    console.print(
        f"[green]scan done[/]: {len(albums)} albums, {written} written, {skipped} unchanged"
    )


# ---------------------------------------------------------------------------
# suggest
# ---------------------------------------------------------------------------


@app.command()
def suggest(
    config_path: Path = typer.Option(default_config_path(), "--config", "-c"),
    album_id: str | None = typer.Option(None, "--album", "-a",
                                        help="Only process this album id."),
    force_llm: bool = typer.Option(False, "--force-llm",
                                   help="Always call Ollama, skip deterministic shortcuts."),
    no_llm: bool = typer.Option(False, "--no-llm",
                                help="Never call Ollama; fall back to deterministic only."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Produce suggestions.json + review.csv for each album that has a manifest."""
    _configure_logging(verbose)
    config = _load(config_path)
    console = Console()

    jobs_dir = config.jobs_dir
    if not jobs_dir.exists():
        raise typer.BadParameter(f"no jobs directory at {jobs_dir}; run 'scan' first")

    target_dirs: list[Path]
    if album_id is not None:
        target = jobs_dir / album_id
        if not target.exists():
            raise typer.BadParameter(f"no album dir at {target}")
        target_dirs = [target]
    else:
        target_dirs = sorted(p for p in jobs_dir.iterdir() if p.is_dir())

    ollama: OllamaClient | None = None
    if not no_llm:
        ollama = OllamaClient(config.ollama)
        if not ollama.health_check():
            console.print(
                f"[yellow]warning[/]: Ollama not reachable at {config.ollama.base_url}; "
                f"continuing in deterministic-only mode"
            )
            ollama.close()
            ollama = None

    try:
        for album_dir in target_dirs:
            manifest_path = album_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = AlbumManifest.model_validate(load_json(manifest_path))

            album_suggestions = suggest_for_album(
                manifest, config, ollama, force_llm=force_llm
            )

            sug_path = album_dir / "suggestions.json"
            dump_json(album_suggestions.model_dump(mode="json"), sug_path)
            _write_review_csv(album_suggestions, album_dir / "review.csv")

            console.print(
                f"[cyan]{manifest.album_id}[/] "
                f"classification={album_suggestions.classification.value} "
                f"probe={album_suggestions.probe_score:.2f} "
                f"llm={album_suggestions.used_llm} "
                f"tracks={len(album_suggestions.suggestions)}"
            )
    finally:
        if ollama is not None:
            ollama.close()


def _write_review_csv(album: AlbumSuggestions, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "track_path", "action", "confidence",
            "current_title", "proposed_title",
            "current_artist", "proposed_artist",
            "issue_flags", "reason",
        ])
        for s in album.suggestions:
            writer.writerow([
                str(s.track_path),
                s.action.value,
                f"{s.confidence:.2f}",
                s.current_title or "",
                s.proposed_title or "",
                s.current_artist or "",
                s.proposed_artist or "",
                ";".join(f.value for f in s.issue_flags),
                s.reason,
            ])


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------


@app.command()
def review(
    album_id: str = typer.Argument(..., help="Album id to review."),
    config_path: Path = typer.Option(default_config_path(), "--config", "-c"),
    show_all: bool = typer.Option(False, "--all",
                                  help="Also review tracks marked keep."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Interactive review for one album."""
    _configure_logging(verbose)
    config = _load(config_path)
    album_dir = config.jobs_dir / album_id
    sug_path = album_dir / "suggestions.json"
    if not sug_path.exists():
        raise typer.BadParameter(f"no suggestions.json at {sug_path}; run 'suggest' first")

    out_path = album_dir / "review_decisions.json"
    review_album(sug_path, out_path, show_all=show_all)


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


@app.command()
def apply(
    album_id: str = typer.Argument(..., help="Album id to apply."),
    config_path: Path = typer.Option(default_config_path(), "--config", "-c"),
    dry_run: bool | None = typer.Option(None, "--dry-run/--no-dry-run",
                                        help="Override the config's dry-run default."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Apply approved review decisions to the audio files."""
    _configure_logging(verbose)
    config = _load(config_path)
    album_dir = config.jobs_dir / album_id
    sug_path = album_dir / "suggestions.json"
    decisions_path = album_dir / "review_decisions.json"
    log_path = album_dir / "apply.log.json"

    if not sug_path.exists():
        raise typer.BadParameter(f"no suggestions.json at {sug_path}")
    if not decisions_path.exists():
        raise typer.BadParameter(f"no review_decisions.json at {decisions_path}; run 'review' first")

    effective_dry_run = config.dry_run_default if dry_run is None else dry_run
    results = apply_decisions(sug_path, decisions_path, log_path, dry_run=effective_dry_run)

    ok = sum(1 for r in results if r.success)
    fail = sum(1 for r in results if not r.success)
    mode = "DRY-RUN" if effective_dry_run else "WRITE"
    Console().print(f"[green]{mode}[/] apply done: {ok} ok, {fail} failed, log → {log_path}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@app.command()
def report(
    config_path: Path = typer.Option(default_config_path(), "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Summarize processed albums across the jobs directory."""
    _configure_logging(verbose)
    config = _load(config_path)
    summary = build_report(config)
    console = Console()

    table = Table(title="music-normalizer report")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    for field in (
        "total_albums", "clean_albums", "deterministic_albums", "llm_albums",
        "broken_albums", "skipped_unchanged", "total_tracks", "proposed_changes",
        "low_confidence", "unreadable",
        # Decision counts (populated from per-album review_decisions.json).
        "pending_review", "approved", "auto_approved", "skipped", "edited",
    ):
        table.add_row(field, str(getattr(summary, field)))
    console.print(table)

    if summary.top_issues:
        issue_table = Table(title="top issues")
        issue_table.add_column("flag")
        issue_table.add_column("count", justify="right")
        for flag, count in summary.top_issues.items():
            issue_table.add_row(flag, str(count))
        console.print(issue_table)


# ---------------------------------------------------------------------------
# Bulk commands
# ---------------------------------------------------------------------------


@app.command(name="export-pending")
def export_pending_cmd(
    config_path: Path = typer.Option(default_config_path(), "--config", "-c"),
    out: Path | None = typer.Option(
        None, "--out", "-o",
        help="CSV output path. Defaults to <jobs_dir>/pending_review.csv.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Write one global CSV of every track pending review across all albums."""
    _configure_logging(verbose)
    config = _load(config_path)
    out_path = out or (config.jobs_dir / "pending_review.csv")
    rows = export_pending(config, out_path)
    Console().print(
        f"[green]exported[/] {rows} pending rows → {out_path}"
    )


@app.command(name="auto-approve")
def auto_approve_cmd(
    config_path: Path = typer.Option(default_config_path(), "--config", "-c"),
    min_confidence: float = typer.Option(
        0.95, "--min-confidence",
        help="Minimum confidence required to auto-approve a suggestion.",
    ),
    allow_llm: bool = typer.Option(
        False, "--allow-llm",
        help="Include LLM-sourced suggestions (default: deterministic only).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Auto-approve safe rows across every album. Manual decisions are never overwritten."""
    _configure_logging(verbose)
    config = _load(config_path)
    policy = AutoApprovePolicy(min_confidence=min_confidence, allow_llm=allow_llm)
    result = auto_approve(config, policy)
    Console().print(
        f"[green]auto-approve[/]: {result.newly_approved} newly approved, "
        f"{result.already_decided} already decided, "
        f"{result.tracks_considered} pending tracks considered across "
        f"{result.total_albums} albums"
    )


@app.command(name="review-all")
def review_all_cmd(
    config_path: Path = typer.Option(default_config_path(), "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Single interactive queue across every album's pending tracks."""
    _configure_logging(verbose)
    config = _load(config_path)
    review_all(config)


@app.command(name="apply-all")
def apply_all_cmd(
    config_path: Path = typer.Option(default_config_path(), "--config", "-c"),
    dry_run: bool | None = typer.Option(
        None, "--dry-run/--no-dry-run",
        help="Override the config's dry-run default.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Apply approved decisions for every album under jobs_dir."""
    _configure_logging(verbose)
    config = _load(config_path)
    effective_dry_run = config.dry_run_default if dry_run is None else dry_run
    result = apply_all(config, dry_run=effective_dry_run)
    mode = "DRY-RUN" if effective_dry_run else "WRITE"
    Console().print(
        f"[green]{mode} apply-all[/]: {result.tracks_ok} ok, "
        f"{result.tracks_failed} failed across {result.albums_applied} albums "
        f"({result.albums_no_decisions} had no decisions)"
    )


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
