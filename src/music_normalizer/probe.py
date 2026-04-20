"""Cheap album sampling.

The probe looks at a handful of tracks (first, second, middle, last) and computes
a score in [0, 1] describing how "clean" the album appears. The classifier uses
this to decide whether to skip the LLM entirely, run deterministic rules only, or
escalate to Ollama.

This is a fast path. Anything expensive belongs in the classifier or the LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .issues import detect_issues
from .models import AlbumManifest, IssueFlag, TrackInfo

# Flags that strongly suggest the LLM will be needed. Presence in a probe sample
# drags the score down sharply.
_HARD_FLAGS = {
    IssueFlag.UNREADABLE_TITLE,
    IssueFlag.GENERIC_TRACK_NAME,
    IssueFlag.EMBEDDED_ALBUM_NAME,
    IssueFlag.SUSPICIOUS_SWAP_ARTIST_TITLE,
}

# Flags that the deterministic layer handles well on its own.
_SOFT_FLAGS = {
    IssueFlag.WEIRD_SEPARATORS,
    IssueFlag.BAD_CAPITALIZATION,
    IssueFlag.ORIGINAL_MIX_REDUNDANT,
    IssueFlag.JUNK_PREFIX,
    IssueFlag.JUNK_SUFFIX,
    IssueFlag.BPM_SUFFIX,
    IssueFlag.EMBEDDED_TRACK_NUMBER,
}


@dataclass
class ProbeResult:
    score: float  # 1.0 = pristine, 0.0 = disaster
    sampled_track_paths: list[str]
    hard_flag_count: int
    soft_flag_count: int
    notes: list[str]


def probe_album(manifest: AlbumManifest, config: Config) -> ProbeResult:
    sample = _select_sample(manifest.tracks, config.probe.sample_positions)
    if not sample:
        return ProbeResult(score=0.0, sampled_track_paths=[], hard_flag_count=0,
                           soft_flag_count=0, notes=["no tracks"])

    hard = 0
    soft = 0
    notes: list[str] = []
    for t in sample:
        flags = detect_issues(t.title, t.artist, config)
        for f in flags:
            if f in _HARD_FLAGS:
                hard += 1
            elif f in _SOFT_FLAGS:
                soft += 1
        if not (t.title or "").strip():
            notes.append(f"{t.filename}: empty title")

    n = len(sample)
    # Each hard flag costs 0.4, each soft flag 0.1, clamped at 0.
    score = max(0.0, 1.0 - (hard * 0.4 + soft * 0.1) / n)
    return ProbeResult(
        score=round(score, 3),
        sampled_track_paths=[str(t.path) for t in sample],
        hard_flag_count=hard,
        soft_flag_count=soft,
        notes=notes,
    )


def _select_sample(tracks: list[TrackInfo], positions: list[str]) -> list[TrackInfo]:
    if not tracks:
        return []
    picked: dict[int, TrackInfo] = {}
    last = len(tracks) - 1
    for pos in positions:
        idx: int | None
        match pos:
            case "first":
                idx = 0
            case "second":
                idx = 1 if last >= 1 else None
            case "middle":
                idx = last // 2
            case "last":
                idx = last
            case _:
                idx = None
        if idx is not None and 0 <= idx <= last:
            picked[idx] = tracks[idx]
    return [picked[k] for k in sorted(picked)]
