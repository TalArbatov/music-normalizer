"""Decide how to handle an album: skip, rules-only, or escalate to LLM.

This module consumes the probe result *and* the full deterministic candidate set,
so "deterministic" actually means "rules can fix everything without blocking flags".
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .models import AlbumClassification, DeterministicCandidate, IssueFlag
from .probe import ProbeResult

_BLOCKING_FLAGS = {
    IssueFlag.UNREADABLE_TITLE,
    IssueFlag.GENERIC_TRACK_NAME,
    IssueFlag.EMBEDDED_ALBUM_NAME,
    IssueFlag.SUSPICIOUS_SWAP_ARTIST_TITLE,
}


@dataclass
class ClassificationResult:
    classification: AlbumClassification
    reason: str


def classify_album(
    probe: ProbeResult,
    candidates: dict[str, DeterministicCandidate],
    config: Config,
) -> ClassificationResult:
    """Map (probe score, candidate state) → AlbumClassification.

    Policy:
    - CLEAN: probe ≥ ``clean_threshold`` AND no deterministic changes proposed.
    - DETERMINISTIC: probe ≥ ``deterministic_threshold`` AND every candidate is
      confident AND no remaining blocking flags.
    - BROKEN: majority of tracks have unreadable/generic titles after cleanup.
    - MIXED: everything else → escalate to LLM.
    """
    clean_t = config.probe.clean_threshold
    det_t = config.probe.deterministic_threshold

    if probe.score >= clean_t and not _has_any_change(candidates):
        return ClassificationResult(
            AlbumClassification.CLEAN,
            f"probe {probe.score:.2f} ≥ clean_threshold {clean_t:.2f} and no rule changes",
        )

    if (
        probe.score >= det_t
        and _all_candidates_confident(candidates)
        and not _has_blocking_flags(candidates)
    ):
        return ClassificationResult(
            AlbumClassification.DETERMINISTIC,
            f"probe {probe.score:.2f} ≥ deterministic_threshold {det_t:.2f}; "
            f"all candidates confident and no blocking remaining flags",
        )

    if _majority_unreadable(candidates):
        return ClassificationResult(
            AlbumClassification.BROKEN,
            "majority of tracks have unreadable/generic titles after deterministic pass",
        )

    reason_bits: list[str] = []
    if probe.score < det_t:
        reason_bits.append(f"probe {probe.score:.2f} < deterministic_threshold {det_t:.2f}")
    if not _all_candidates_confident(candidates):
        reason_bits.append("low-confidence deterministic candidates")
    if _has_blocking_flags(candidates):
        reason_bits.append("blocking remaining flags present")
    reason = "; ".join(reason_bits) or "mixed evidence"
    return ClassificationResult(AlbumClassification.MIXED, reason)


def _has_any_change(candidates: dict[str, DeterministicCandidate]) -> bool:
    return any(
        (c.proposed_title is not None) or (c.proposed_artist is not None)
        for c in candidates.values()
    )


def _all_candidates_confident(candidates: dict[str, DeterministicCandidate]) -> bool:
    return all(c.confidence >= 0.75 for c in candidates.values())


def _remaining_flags(c: DeterministicCandidate) -> list[IssueFlag]:
    # Prefer the explicit field; fall back to issue_flags for older candidates.
    return c.remaining_issue_flags or c.issue_flags


def _has_blocking_flags(candidates: dict[str, DeterministicCandidate]) -> bool:
    return any(
        any(f in _BLOCKING_FLAGS for f in _remaining_flags(c)) for c in candidates.values()
    )


def _majority_unreadable(candidates: dict[str, DeterministicCandidate]) -> bool:
    if not candidates:
        return False
    unreadable = sum(
        1 for c in candidates.values()
        if IssueFlag.UNREADABLE_TITLE in _remaining_flags(c)
        or IssueFlag.GENERIC_TRACK_NAME in _remaining_flags(c)
    )
    return unreadable * 2 > len(candidates)
