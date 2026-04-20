"""Validate proposed Suggestions before they reach the review stage.

The validator is the last line of defense against the LLM (or an overly-eager
rules change). It can:

- drop a suggestion entirely (when it becomes a no-op or is clearly wrong)
- downgrade ``propose_fix`` → ``unreadable_needs_manual_review`` when the model's
  output looks like invention
- lower confidence for borderline cases so the reviewer sees a warning

It never silently modifies proposed values; it only adds notes and adjusts status.
"""

from __future__ import annotations

import re

from .config import Config
from .issues import GENERIC_TITLE_RE, is_symbolic_title, is_unreadable
from .models import IssueFlag, Suggestion, SuggestionSource, TrackAction


def validate_suggestions(suggestions: list[Suggestion], config: Config) -> list[Suggestion]:
    return [validate(s, config) for s in suggestions]


def validate(s: Suggestion, config: Config) -> Suggestion:
    notes: list[str] = list(s.validator_notes)
    out = s.model_copy(deep=True)

    # 0a. LLM mutating a short symbolic title (``i``, ``iii``, ``iiiii`` …) is
    # destructive — every character is meaningful. Revert to the current title
    # so the rest of validation sees no title change. Artist edits, if any,
    # still apply.
    if (
        out.source == SuggestionSource.LLM
        and out.proposed_title is not None
        and is_symbolic_title(out.current_title)
        and (out.proposed_title or "").strip() != (out.current_title or "").strip()
    ):
        notes.append(
            f"validator: rejected LLM edit of symbolic title "
            f"{out.current_title!r} -> {out.proposed_title!r}"
        )
        out.proposed_title = out.current_title
        out.confidence = min(out.confidence, 0.3)

    # 0. No effective change? Short-circuit to KEEP before applying any of the
    # heuristics below — otherwise a very short current title like "a" would hit
    # the "unreadable" branch even when nothing is being proposed.
    if not out.has_change:
        if out.action == TrackAction.PROPOSE_FIX:
            out.action = TrackAction.KEEP
            notes.append("validator: no effective change, marked keep")
        out.validator_notes = notes
        return out

    # 1. Empty proposed fields → drop the change, keep track for review.
    if out.proposed_title is not None and not out.proposed_title.strip():
        notes.append("validator: rejected empty proposed_title")
        out.proposed_title = out.current_title
        out.confidence = min(out.confidence, 0.3)
        _downgrade(out, TrackAction.UNREADABLE_NEEDS_MANUAL_REVIEW)
    if out.proposed_artist is not None and not out.proposed_artist.strip():
        notes.append("validator: rejected empty proposed_artist")
        out.proposed_artist = out.current_artist
        out.confidence = min(out.confidence, 0.3)

    # 2. Still generic junk? Use the same strict definition the rules engine and
    # detector use so there's one source of truth.
    if out.proposed_title and GENERIC_TITLE_RE.match(out.proposed_title):
        notes.append("validator: proposed title is still generic")
        out.confidence = min(out.confidence, 0.3)
        _downgrade(out, TrackAction.UNREADABLE_NEEDS_MANUAL_REVIEW)
    elif (
        out.proposed_title
        and IssueFlag.GENERIC_TRACK_NAME in out.remaining_issue_flags
    ):
        # LLM (or stale deterministic state) flagged generic_track_name on a
        # title that clearly isn't generic — e.g. "Bugpowder (Braingineers remix)".
        # Drop the spurious flag so it doesn't later downgrade the suggestion.
        out.remaining_issue_flags = [
            f for f in out.remaining_issue_flags if f != IssueFlag.GENERIC_TRACK_NAME
        ]
        out.issue_flags = [
            f for f in out.issue_flags if f != IssueFlag.GENERIC_TRACK_NAME
        ]
        notes.append(
            "validator: removed spurious generic_track_name flag; title is descriptive"
        )

    # 3. Looks unreadable still.
    if out.proposed_title and is_unreadable(out.proposed_title):
        notes.append("validator: proposed title still looks unreadable")
        _downgrade(out, TrackAction.UNREADABLE_NEEDS_MANUAL_REVIEW)

    # 4. LLM hallucination check: if the model's proposed text contains tokens not
    #    derivable from the input, we can't detect that in general — but we can at
    #    least flag when proposed strings share zero tokens with the original.
    if out.source == SuggestionSource.LLM and out.has_change:
        if _looks_invented(out):
            notes.append("validator: proposed strings share no tokens with current — possible invention")
            out.confidence = min(out.confidence, 0.4)

    # 5. Suspicious artist/title swap.
    if _looks_like_swap(out):
        notes.append("validator: artist/title look swapped — downgrading")
        out.issue_flags = _ensure_flag(out.issue_flags, IssueFlag.SUSPICIOUS_SWAP_ARTIST_TITLE)
        _downgrade(out, TrackAction.UNREADABLE_NEEDS_MANUAL_REVIEW)

    # 6. Low confidence threshold.
    if out.confidence < config.confidence.low_confidence_flag and out.action == TrackAction.PROPOSE_FIX:
        notes.append(
            f"validator: confidence {out.confidence:.2f} below low-confidence flag "
            f"{config.confidence.low_confidence_flag:.2f}"
        )

    # 7. Aggressive total rewrite: both title AND artist changed AND neither shares
    #    tokens with the originals. Downgrade to manual review.
    if (
        out.changes_title
        and out.changes_artist
        and _no_shared_tokens(out.current_title or "", out.proposed_title or "")
        and _no_shared_tokens(out.current_artist or "", out.proposed_artist or "")
    ):
        notes.append("validator: both fields fully rewritten — requires manual review")
        _downgrade(out, TrackAction.UNREADABLE_NEEDS_MANUAL_REVIEW)

    out.validator_notes = notes
    return out


def _downgrade(s: Suggestion, to: TrackAction) -> None:
    if s.action != to:
        s.downgraded_from = s.downgraded_from or s.action
        s.action = to


def _ensure_flag(flags: list[IssueFlag], flag: IssueFlag) -> list[IssueFlag]:
    return flags if flag in flags else [*flags, flag]


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[\s\-_/]+", s.lower()) if len(t) > 1}


def _no_shared_tokens(a: str, b: str) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return not (ta & tb)


def _looks_invented(s: Suggestion) -> bool:
    # If *both* fields share no tokens with the originals, treat as invention.
    title_shared = not _no_shared_tokens(s.current_title or "", s.proposed_title or "")
    artist_shared = not _no_shared_tokens(s.current_artist or "", s.proposed_artist or "")
    return not title_shared and not artist_shared


def _looks_like_swap(s: Suggestion) -> bool:
    c_title = (s.current_title or "").strip().lower()
    c_artist = (s.current_artist or "").strip().lower()
    p_title = (s.proposed_title or "").strip().lower()
    p_artist = (s.proposed_artist or "").strip().lower()
    if not (c_title and c_artist and p_title and p_artist):
        return False
    return c_title == p_artist and c_artist == p_title
