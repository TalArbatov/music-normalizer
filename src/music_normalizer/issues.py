"""Detector functions that map raw title/artist strings to IssueFlag sets.

Kept separate from :mod:`rules` so the rules engine can both consume flags (to
decide what to normalize) and emit flags (as parse notes).
"""

from __future__ import annotations

import re

from .config import Config
from .models import IssueFlag

# Tokens that constitute a "generic" title on their own.
#
# Public because both the deterministic rules and the validator need to agree on
# exactly what counts as generic. A descriptive title like "Bugpowder" must not
# be flagged just because it is short or unusual.
GENERIC_TITLE_RE: re.Pattern[str] = re.compile(
    r"""^\s*
    (?:
        track\s*\d+                 # "Track 01", "Track 1"
      | untitled(?:\s*\d+)?          # "Untitled", "Untitled 5"
      | audio\s*track\s*\d+          # "Audio Track 03"
      | t\d+                          # "T7"
      | \d{1,3}                       # bare "9" or "12"
    )
    \s*$""",
    re.IGNORECASE | re.VERBOSE,
)

_LEADING_TRACK_NUM_RE = re.compile(r"^\s*\d{1,3}\s*[-._)\]\s]")
_BPM_RE = re.compile(r"[\s\-\(\[]+\d{2,3}\s*bpm\b", re.IGNORECASE)
_MULTI_DASH_RE = re.compile(r"\s-\s.*\s-\s")


def is_symbolic_title(s: str | None) -> bool:
    """Short symbolic tokens we must never rewrite or "correct".

    Covers single-letter, repeated-letter (``i``, ``ii``, ``iiiii``) and
    roman-numeral-style (``iv``, ``viii``) titles. These look destructible to a
    case-fixer or an LLM but every character is meaningful — turning ``iiiii``
    into ``iiii`` or ``i`` into ``I`` is a destructive edit, not a cleanup.
    """
    if s is None:
        return False
    t = s.strip()
    if not t or len(t) > 6:
        return False
    if not t.isalpha():
        return False
    low = t.lower()
    # Repeated-letter form: e.g. "i", "ii", "aaa".
    if len(set(low)) == 1:
        return True
    # Roman-numeral-only letters: i, v, x.
    if set(low) <= {"i", "v", "x"}:
        return True
    return False


def detect_issues(title: str | None, artist: str | None, config: Config) -> list[IssueFlag]:
    """Return the full set of flags raised against a (title, artist) pair."""
    flags: list[IssueFlag] = []
    t = title or ""
    a = artist or ""

    if not a.strip():
        flags.append(IssueFlag.ARTIST_MISSING)

    if _LEADING_TRACK_NUM_RE.search(t):
        flags.append(IssueFlag.EMBEDDED_TRACK_NUMBER)

    # "Artist - Title" pattern in title field is the classic darkpsy embed.
    if " - " in t and t.count(" - ") <= 3:
        flags.append(IssueFlag.EMBEDDED_ARTIST)

    if _has_junk_prefix(t, config.junk_prefixes):
        flags.append(IssueFlag.JUNK_PREFIX)

    if _has_junk_suffix(t, config.junk_suffixes):
        flags.append(IssueFlag.JUNK_SUFFIX)

    if _has_original_mix(t):
        flags.append(IssueFlag.ORIGINAL_MIX_REDUNDANT)

    if _BPM_RE.search(t):
        flags.append(IssueFlag.BPM_SUFFIX)

    if _has_weird_separators(t, config):
        flags.append(IssueFlag.WEIRD_SEPARATORS)

    if _is_bad_capitalization(t):
        flags.append(IssueFlag.BAD_CAPITALIZATION)

    if GENERIC_TITLE_RE.match(t):
        flags.append(IssueFlag.GENERIC_TRACK_NAME)

    if not t.strip() or is_unreadable(t):
        flags.append(IssueFlag.UNREADABLE_TITLE)

    if a and t and a.strip().lower() == t.strip().lower():
        flags.append(IssueFlag.DUPLICATE_ARTIST_TITLE)

    for marker in config.scene_markers:
        if marker.lower() in t.lower():
            flags.append(IssueFlag.SCENE_JUNK)
            break

    if not a.strip() and " - " in t:
        flags.append(IssueFlag.MISSING_ARTIST_BUT_PRESENT_IN_TITLE)

    return _dedupe(flags)


def _dedupe(flags: list[IssueFlag]) -> list[IssueFlag]:
    seen: set[IssueFlag] = set()
    out: list[IssueFlag] = []
    for f in flags:
        if f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def _has_junk_prefix(s: str, prefixes: list[str]) -> bool:
    low = s.lower().lstrip()
    return any(low.startswith(p.lower()) for p in prefixes)


def _has_junk_suffix(s: str, suffixes: list[str]) -> bool:
    low = s.lower().rstrip()
    return any(low.endswith(p.lower()) for p in suffixes)


def _has_original_mix(s: str) -> bool:
    return bool(re.search(r"\b(original\s*mix)\b", s, re.IGNORECASE))


def _has_weird_separators(s: str, config: Config) -> bool:
    if "_" in s:
        return True
    if "  " in s:  # double space
        return True
    for token in config.separator_rules.replace_with_dash:
        if token in s:
            return True
    # Multiple " - " separators probably indicate mashed metadata.
    return bool(_MULTI_DASH_RE.search(s))


def _is_bad_capitalization(s: str) -> bool:
    """Pure lowercase or pure uppercase strings that contain letters."""
    if not any(ch.isalpha() for ch in s):
        return False
    if is_symbolic_title(s):
        return False
    stripped = s.strip()
    if stripped == stripped.lower() and stripped != stripped.upper():
        return True
    if stripped == stripped.upper() and len(stripped) > 3:
        return True
    return False


def is_unreadable(s: str) -> bool:
    """Heuristic: mostly non-letters or very short → unreadable."""
    if is_symbolic_title(s):
        return False
    stripped = s.strip()
    if len(stripped) < 2:
        return True
    letters = sum(1 for ch in stripped if ch.isalpha())
    return letters / max(1, len(stripped)) < 0.3
