"""Pydantic data models used across the pipeline.

These types are the contract between modules. Every artifact written under ``jobs/``
serializes one of these models.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IssueFlag(str, Enum):
    """Canonical set of issues the detector / LLM can raise against a track."""

    EMBEDDED_TRACK_NUMBER = "embedded_track_number"
    EMBEDDED_ARTIST = "embedded_artist"
    EMBEDDED_ALBUM_NAME = "embedded_album_name"
    BAD_CAPITALIZATION = "bad_capitalization"
    WEIRD_SEPARATORS = "weird_separators"
    JUNK_PREFIX = "junk_prefix"
    JUNK_SUFFIX = "junk_suffix"
    ORIGINAL_MIX_REDUNDANT = "original_mix_redundant"
    BPM_SUFFIX = "bpm_suffix"
    GENERIC_TRACK_NAME = "generic_track_name"
    UNREADABLE_TITLE = "unreadable_title"
    DUPLICATE_ARTIST_TITLE = "duplicate_artist_title"
    SCENE_JUNK = "scene_junk"
    MISSING_ARTIST_BUT_PRESENT_IN_TITLE = "missing_artist_but_present_in_title"
    ARTIST_MISSING = "artist_missing"
    SUSPICIOUS_SWAP_ARTIST_TITLE = "suspicious_swap_artist_title"


class SuggestionSource(str, Enum):
    """Where a suggestion came from. Used by validator and report."""

    DETERMINISTIC = "deterministic"
    LLM = "llm"
    NONE = "none"  # track deemed clean, no change proposed


class TrackAction(str, Enum):
    """What the pipeline thinks should happen with this track."""

    KEEP = "keep"
    PROPOSE_FIX = "propose_fix"
    UNREADABLE_NEEDS_MANUAL_REVIEW = "unreadable_needs_manual_review"


class AlbumClassification(str, Enum):
    CLEAN = "clean"
    DETERMINISTIC = "deterministic"
    MIXED = "mixed"
    BROKEN = "broken"


class ReviewAction(str, Enum):
    APPROVE = "approve"
    SKIP = "skip"
    EDIT = "edit"


# ---------------------------------------------------------------------------
# Track / album
# ---------------------------------------------------------------------------


class TrackInfo(BaseModel):
    """Raw observation of a single audio file on disk.

    Immutable for the duration of a job. ``path`` is the canonical identifier — it is
    also the key used throughout suggestions and review decisions.
    """

    model_config = ConfigDict(frozen=False)

    path: Path
    filename: str
    ext: str
    track_number: int | None = None
    disc_number: int | None = None
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    album_artist: str | None = None
    year: int | None = None
    duration_s: float | None = None
    # Raw tag dict preserved for audit (first value per key, stringified).
    raw_tags: dict[str, str] = Field(default_factory=dict)
    # mtime of the file at scan time, used for fingerprinting.
    mtime: float | None = None
    size_bytes: int | None = None


class AlbumManifest(BaseModel):
    """One album worth of tracks plus a stable fingerprint.

    Fingerprint is computed from sorted (path, mtime, size) tuples — changes to any
    file invalidate the cache, which is what we want for a safety-first tool.
    """

    album_id: str  # short hash, used as subdirectory under jobs/
    root: Path  # folder on disk this album lives in
    album_name: str | None = None
    album_artist: str | None = None
    tracks: list[TrackInfo] = Field(default_factory=list)
    fingerprint: str
    scanned_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Deterministic layer
# ---------------------------------------------------------------------------


class DeterministicCandidate(BaseModel):
    """A proposed fix produced by the rules engine, before validation.

    Deliberately mirrors the shape of a Suggestion so that the validator can treat
    deterministic and LLM candidates uniformly.

    ``issue_flags`` is kept as a back-compat alias pointing to the *remaining* flags
    (what still looks wrong after deterministic cleanup). ``original_issue_flags``
    is what was wrong on the raw input. Classifier/confidence use the remaining set.
    """

    track_path: Path
    proposed_title: str | None = None
    proposed_artist: str | None = None
    issue_flags: list[IssueFlag] = Field(default_factory=list)
    original_issue_flags: list[IssueFlag] = Field(default_factory=list)
    remaining_issue_flags: list[IssueFlag] = Field(default_factory=list)
    parse_notes: list[str] = Field(default_factory=list)
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Suggestions (unified output of deterministic + LLM path)
# ---------------------------------------------------------------------------


class Suggestion(BaseModel):
    """A proposed change for one track. The canonical unit of review."""

    track_path: Path
    current_title: str | None
    current_artist: str | None
    proposed_title: str | None
    proposed_artist: str | None
    action: TrackAction
    issue_flags: list[IssueFlag] = Field(default_factory=list)
    # Flags raised against the raw (pre-cleanup) tags; kept for audit + review UI.
    original_issue_flags: list[IssueFlag] = Field(default_factory=list)
    # Flags still present after the deterministic pass. This is what the classifier
    # and downstream confidence scoring should consult.
    remaining_issue_flags: list[IssueFlag] = Field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    source: SuggestionSource = SuggestionSource.NONE
    # Filled in by the validator. ``downgraded_from`` captures the original action if
    # the validator weakened it (e.g. propose_fix -> unreadable_needs_manual_review).
    downgraded_from: TrackAction | None = None
    validator_notes: list[str] = Field(default_factory=list)

    @property
    def changes_title(self) -> bool:
        return self.proposed_title is not None and self.proposed_title != (self.current_title or "")

    @property
    def changes_artist(self) -> bool:
        return self.proposed_artist is not None and self.proposed_artist != (
            self.current_artist or ""
        )

    @property
    def has_change(self) -> bool:
        return self.changes_title or self.changes_artist


class AlbumSuggestions(BaseModel):
    """All suggestions for an album plus how the album was classified."""

    album_id: str
    classification: AlbumClassification
    probe_score: float
    suggestions: list[Suggestion]
    # Diagnostic string explaining how the classifier reached its decision.
    classification_reason: str = ""
    # True if the LLM was invoked for this album.
    used_llm: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Review + apply
# ---------------------------------------------------------------------------


class ReviewDecision(BaseModel):
    """A user's verdict on a single suggestion."""

    track_path: Path
    action: ReviewAction
    # Present when action == EDIT.
    final_title: str | None = None
    final_artist: str | None = None
    decided_at: datetime = Field(default_factory=datetime.utcnow)
    note: str | None = None


class ApplyResult(BaseModel):
    """Outcome of writing one approved change to disk."""

    track_path: Path
    applied_title: str | None
    applied_artist: str | None
    previous_title: str | None
    previous_artist: str | None
    success: bool
    error: str | None = None
    dry_run: bool = False
    applied_at: datetime = Field(default_factory=datetime.utcnow)


class ReportSummary(BaseModel):
    """Aggregated counts for the report command."""

    total_albums: int = 0
    clean_albums: int = 0
    deterministic_albums: int = 0
    llm_albums: int = 0
    broken_albums: int = 0
    skipped_unchanged: int = 0
    total_tracks: int = 0
    proposed_changes: int = 0
    low_confidence: int = 0
    unreadable: int = 0
    # Decision-level counts, filled in from review_decisions.json across jobs.
    pending_review: int = 0
    approved: int = 0
    auto_approved: int = 0
    skipped: int = 0
    edited: int = 0
    top_issues: dict[str, int] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


# Shared across bulk, report, and review layers so everyone agrees on which
# TrackActions still need a human decision.
PENDING_TRACK_ACTIONS: frozenset[TrackAction] = frozenset(
    {TrackAction.PROPOSE_FIX, TrackAction.UNREADABLE_NEEDS_MANUAL_REVIEW}
)

# Marker written into ReviewDecision.note when a decision was produced by the
# auto-approve pass rather than by a human reviewer. The report uses this to
# distinguish hand-approved from machine-approved rows.
AUTO_APPROVE_NOTE: str = "auto-approved"
