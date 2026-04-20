"""Top-level orchestration: given an album manifest, produce validated Suggestions.

Flow:
  scan → group → (this module) → review → apply

This is where the speed heuristics live — we do a probe, a deterministic pass,
classify, and only call the LLM when the classifier says we must.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .classifier import classify_album
from .config import Config
from .models import (
    AlbumClassification,
    AlbumManifest,
    AlbumSuggestions,
    DeterministicCandidate,
    IssueFlag,
    Suggestion,
    SuggestionSource,
    TrackAction,
    TrackInfo,
)
from .ollama_client import OllamaClient, OllamaError
from .probe import probe_album
from .prompts import SYSTEM_PROMPT, build_user_message, validate_response
from .rules import propose_for_album
from .utils import get_logger
from .validator import validate_suggestions

log = get_logger(__name__)


def suggest_for_album(
    manifest: AlbumManifest,
    config: Config,
    ollama: OllamaClient | None = None,
    *,
    force_llm: bool = False,
) -> AlbumSuggestions:
    """Full suggestion pipeline for a single album."""
    probe = probe_album(manifest, config)
    candidates = propose_for_album(manifest.tracks, config)
    classification = classify_album(probe, {str(k): v for k, v in candidates.items()}, config)

    used_llm = False
    suggestions: list[Suggestion]

    if classification.classification == AlbumClassification.CLEAN and not force_llm:
        suggestions = _keep_all(manifest.tracks, candidates)

    elif classification.classification == AlbumClassification.DETERMINISTIC and not force_llm:
        suggestions = _from_candidates(manifest.tracks, candidates)

    else:
        # MIXED / BROKEN / forced → escalate.
        if ollama is None:
            log.warning(
                "album %s needs LLM (class=%s) but no ollama client was provided; "
                "falling back to deterministic candidates",
                manifest.album_id,
                classification.classification,
            )
            suggestions = _from_candidates(manifest.tracks, candidates)
        else:
            try:
                suggestions = _ask_llm(manifest, ollama, candidates, config)
                used_llm = True
            except OllamaError as e:
                log.error("Ollama call failed for album %s: %s", manifest.album_id, e)
                suggestions = _from_candidates(manifest.tracks, candidates)

    validated = validate_suggestions(suggestions, config)
    return AlbumSuggestions(
        album_id=manifest.album_id,
        classification=classification.classification,
        probe_score=probe.score,
        suggestions=validated,
        classification_reason=classification.reason,
        used_llm=used_llm,
    )


# ---------------------------------------------------------------------------
# Suggestion construction from the three sources
# ---------------------------------------------------------------------------


def _keep_all(
    tracks: list[TrackInfo], candidates: dict[Path, DeterministicCandidate]
) -> list[Suggestion]:
    out: list[Suggestion] = []
    for t in tracks:
        c = candidates.get(t.path)
        out.append(Suggestion(
            track_path=t.path,
            current_title=t.title,
            current_artist=t.artist,
            proposed_title=t.title,
            proposed_artist=t.artist,
            action=TrackAction.KEEP,
            issue_flags=(c.remaining_issue_flags if c else []),
            original_issue_flags=(c.original_issue_flags if c else []),
            remaining_issue_flags=(c.remaining_issue_flags if c else []),
            confidence=1.0,
            reason="album probe clean; no change needed",
            source=SuggestionSource.NONE,
        ))
    return out


def _from_candidates(
    tracks: list[TrackInfo], candidates: dict[Path, DeterministicCandidate]
) -> list[Suggestion]:
    out: list[Suggestion] = []
    for t in tracks:
        c = candidates.get(t.path)
        if c is None:
            out.append(Suggestion(
                track_path=t.path,
                current_title=t.title,
                current_artist=t.artist,
                proposed_title=t.title,
                proposed_artist=t.artist,
                action=TrackAction.KEEP,
                confidence=1.0,
                reason="no deterministic candidate produced",
                source=SuggestionSource.NONE,
            ))
            continue
        has_change = (c.proposed_title is not None) or (c.proposed_artist is not None)
        action = TrackAction.PROPOSE_FIX if has_change else TrackAction.KEEP
        if IssueFlag.UNREADABLE_TITLE in c.remaining_issue_flags and not has_change:
            action = TrackAction.UNREADABLE_NEEDS_MANUAL_REVIEW
        reason = "; ".join(c.parse_notes) if c.parse_notes else "no changes proposed"
        out.append(Suggestion(
            track_path=t.path,
            current_title=t.title,
            current_artist=t.artist,
            proposed_title=c.proposed_title if c.proposed_title is not None else t.title,
            proposed_artist=c.proposed_artist if c.proposed_artist is not None else t.artist,
            action=action,
            issue_flags=c.remaining_issue_flags,
            original_issue_flags=c.original_issue_flags,
            remaining_issue_flags=c.remaining_issue_flags,
            confidence=c.confidence,
            reason=reason,
            source=SuggestionSource.DETERMINISTIC,
        ))
    return out


def _ask_llm(
    manifest: AlbumManifest,
    ollama: OllamaClient,
    candidates: dict[Path, DeterministicCandidate],
    config: Config,
) -> list[Suggestion]:
    """Call Ollama in chunks of at most ``max_tracks_per_call`` and merge results.

    If any chunk fails schema validation we fall back to deterministic suggestions
    for that chunk only — the rest of the album still gets LLM cleanup.
    """
    chunk_size = max(1, config.ollama.max_tracks_per_call)
    tracks = manifest.tracks
    merged: dict[str, Suggestion] = {}

    for i in range(0, len(tracks), chunk_size):
        chunk = tracks[i : i + chunk_size]
        user = build_user_message(manifest, tracks=chunk)
        raw = ollama.chat_json(SYSTEM_PROMPT, user)
        ok, reason = validate_response(raw)
        if not ok:
            log.error(
                "Ollama response failed schema validation for album %s chunk %d-%d: %s",
                manifest.album_id, i, i + len(chunk), reason,
            )
            # Deterministic fallback just for this chunk.
            for s in _from_candidates(chunk, candidates):
                merged[str(s.track_path)] = s
            continue
        for s in _parse_llm_response(raw, chunk, candidates):
            merged[str(s.track_path)] = s

    # Any track the LLM completely skipped across all chunks → deterministic fallback.
    out: list[Suggestion] = []
    for t in tracks:
        s = merged.get(str(t.path))
        if s is not None:
            out.append(s)
            continue
        c = candidates.get(t.path)
        out.append(Suggestion(
            track_path=t.path,
            current_title=t.title,
            current_artist=t.artist,
            proposed_title=(c.proposed_title if c else None) or t.title,
            proposed_artist=(c.proposed_artist if c else None) or t.artist,
            action=TrackAction.KEEP,
            issue_flags=(c.remaining_issue_flags if c else []),
            original_issue_flags=(c.original_issue_flags if c else []),
            remaining_issue_flags=(c.remaining_issue_flags if c else []),
            confidence=0.5,
            reason="LLM response missing this track; used deterministic fallback",
            source=SuggestionSource.DETERMINISTIC,
        ))
    return out


def _parse_llm_response(
    raw: dict[str, Any],
    tracks: list[TrackInfo],
    candidates: dict[Path, DeterministicCandidate],
) -> list[Suggestion]:
    tracks_by_path = {str(t.path): t for t in tracks}
    by_path: dict[str, dict[str, Any]] = {}
    for entry in raw.get("tracks", []):
        p = entry.get("track_path")
        if p in tracks_by_path:
            by_path[p] = entry

    out: list[Suggestion] = []
    for t in tracks:
        entry = by_path.get(str(t.path))
        c = candidates.get(t.path)
        if entry is None:
            out.append(Suggestion(
                track_path=t.path,
                current_title=t.title,
                current_artist=t.artist,
                proposed_title=(c.proposed_title if c else None) or t.title,
                proposed_artist=(c.proposed_artist if c else None) or t.artist,
                action=TrackAction.KEEP,
                issue_flags=(c.remaining_issue_flags if c else []),
                original_issue_flags=(c.original_issue_flags if c else []),
                remaining_issue_flags=(c.remaining_issue_flags if c else []),
                confidence=0.5,
                reason="LLM response missing this track; used deterministic fallback",
                source=SuggestionSource.DETERMINISTIC,
            ))
            continue

        action = _parse_action(entry.get("action"))
        flags = _parse_flags(entry.get("issue_flags") or [])
        out.append(Suggestion(
            track_path=t.path,
            current_title=t.title,
            current_artist=t.artist,
            proposed_title=entry.get("proposed_title") or t.title,
            proposed_artist=entry.get("proposed_artist") or t.artist,
            action=action,
            issue_flags=flags,
            # LLM sees raw tags only, so its "issue_flags" describes the current
            # state — treat that as the remaining set for downstream consistency.
            original_issue_flags=(c.original_issue_flags if c else []),
            remaining_issue_flags=flags,
            confidence=float(entry.get("confidence") or 0.0),
            reason=str(entry.get("reason") or ""),
            source=SuggestionSource.LLM,
        ))
    return out


def _parse_action(value: Any) -> TrackAction:
    try:
        return TrackAction(str(value))
    except ValueError:
        return TrackAction.UNREADABLE_NEEDS_MANUAL_REVIEW


def _parse_flags(values: list[Any]) -> list[IssueFlag]:
    out: list[IssueFlag] = []
    for v in values:
        try:
            out.append(IssueFlag(str(v)))
        except ValueError:
            continue
    return out
