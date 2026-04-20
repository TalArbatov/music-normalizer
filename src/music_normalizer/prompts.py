"""Prompt construction for the LLM path.

Design goals:
- The model only gets what's on disk. No network lookups, no outside knowledge.
- It must return strict JSON matching :data:`RESPONSE_SCHEMA`.
- Minimal edits: when in doubt, keep the original.
- Explicit ``unreadable_needs_manual_review`` action so the model has an honest
  escape hatch instead of hallucinating a title.
"""

from __future__ import annotations

import json
from typing import Any

from .models import AlbumManifest, TrackInfo

SYSTEM_PROMPT = """You are a careful music-metadata cleanup assistant. Your only job is to
clean up messy TITLE and ARTIST tags for one album of audio files.

STRICT RULES — follow all of them:
1. Use ONLY the information provided in the user message (tags + filenames + album
   context). Do NOT use outside knowledge about artists, discographies, labels, or
   releases. Do NOT invent titles or artist names that are not already present in
   the input.
2. Prefer MINIMAL edits. If a field is already acceptable, keep it unchanged.
3. Never produce an empty title or empty artist.
4. If the information is too ambiguous or unreadable to fix safely, set the track's
   action to "unreadable_needs_manual_review" and keep proposed_title / proposed_artist
   equal to the current values.
5. Return ONLY valid JSON that matches the requested schema. No prose, no markdown,
   no code fences.
6. For each track, set confidence in [0.0, 1.0] reflecting how sure you are the
   proposed fix is correct. Low confidence is fine and encouraged when unsure.
7. issue_flags must be drawn from this exact set: embedded_track_number,
   embedded_artist, embedded_album_name, bad_capitalization, weird_separators,
   junk_prefix, junk_suffix, original_mix_redundant, bpm_suffix, generic_track_name,
   unreadable_title, duplicate_artist_title, scene_junk,
   missing_artist_but_present_in_title, artist_missing, suspicious_swap_artist_title.
8. Your output `tracks` array MUST have exactly one entry per input track, in the
   same order, and each `track_path` must equal the input `path` verbatim.
9. `generic_track_name` is RESERVED for true placeholder titles only:
   "Track 01", "Track 5", "Untitled", "Untitled 3", "Audio Track 07", "T9",
   or a bare number like "9" or "12". Do NOT raise this flag for descriptive
   titles — "Bugpowder", "Pure Energy", "Nocturnal Visions", and other real
   song names are NOT generic even when they are short, unusual, or one word.
10. When a title contains a leading track number like "9 - Bugpowder" or
    "01 - Pure Energy", strip the number and keep the descriptive remainder.
    Do NOT carry the "9 -" into proposed_title and do NOT flag the result as
    generic_track_name.
"""


# JSON schema (documented for humans — we validate on parse as well).
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["album_id", "tracks"],
    "properties": {
        "album_id": {"type": "string"},
        "album_notes": {"type": "string"},
        "tracks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "track_path",
                    "action",
                    "proposed_title",
                    "proposed_artist",
                    "confidence",
                    "issue_flags",
                    "reason",
                ],
                "properties": {
                    "track_path": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": [
                            "keep",
                            "propose_fix",
                            "unreadable_needs_manual_review",
                        ],
                    },
                    "proposed_title": {"type": ["string", "null"]},
                    "proposed_artist": {"type": ["string", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "issue_flags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string"},
                },
            },
        },
    },
}


def build_user_message(
    manifest: AlbumManifest,
    tracks: list[TrackInfo] | None = None,
) -> str:
    """Serialize the album (or a subset of its tracks) into a compact JSON payload.

    ``tracks`` defaults to every track in ``manifest`` but can be a chunk for
    album-level batching when the album exceeds ``max_tracks_per_call``.
    """
    selected = tracks if tracks is not None else manifest.tracks
    payload_tracks = [
        {
            "path": str(t.path),
            "filename": t.filename,
            "track_number": t.track_number,
            "disc_number": t.disc_number,
            "current_title": t.title,
            "current_artist": t.artist,
        }
        for t in selected
    ]
    payload = {
        "album_id": manifest.album_id,
        "album_name": manifest.album_name,
        "album_artist": manifest.album_artist,
        "album_folder": str(manifest.root),
        "tracks": payload_tracks,
        "response_schema": RESPONSE_SCHEMA,
    }
    return (
        "Clean up the TITLE and ARTIST tags for the album described below. "
        "Return a single JSON object matching response_schema. "
        "Do not include any text outside the JSON.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


# Flags accepted from the model. Anything outside this set is silently dropped.
_VALID_FLAGS: set[str] = {
    "embedded_track_number", "embedded_artist", "embedded_album_name",
    "bad_capitalization", "weird_separators", "junk_prefix", "junk_suffix",
    "original_mix_redundant", "bpm_suffix", "generic_track_name",
    "unreadable_title", "duplicate_artist_title", "scene_junk",
    "missing_artist_but_present_in_title", "artist_missing",
    "suspicious_swap_artist_title",
}

_VALID_ACTIONS: set[str] = {"keep", "propose_fix", "unreadable_needs_manual_review"}


def validate_response(raw: Any) -> tuple[bool, str]:
    """Strict structural check of an Ollama response.

    Returns ``(ok, reason)``. Does not mutate or coerce — callers should only
    trust ``raw`` after a True return. Unknown flag strings are tolerated at the
    per-track level (dropped later in :mod:`suggestions`) but a malformed
    top-level shape fails validation outright.
    """
    if not isinstance(raw, dict):
        return False, f"top-level must be object, got {type(raw).__name__}"
    if "tracks" not in raw or not isinstance(raw["tracks"], list):
        return False, "missing or non-list 'tracks'"
    for i, entry in enumerate(raw["tracks"]):
        if not isinstance(entry, dict):
            return False, f"tracks[{i}] is not an object"
        for field in ("track_path", "action", "confidence", "issue_flags"):
            if field not in entry:
                return False, f"tracks[{i}] missing '{field}'"
        if not isinstance(entry["track_path"], str) or not entry["track_path"]:
            return False, f"tracks[{i}].track_path must be non-empty string"
        if entry["action"] not in _VALID_ACTIONS:
            return False, f"tracks[{i}].action {entry['action']!r} is not a known action"
        try:
            conf = float(entry["confidence"])
        except (TypeError, ValueError):
            return False, f"tracks[{i}].confidence is not a number"
        if not 0.0 <= conf <= 1.0:
            return False, f"tracks[{i}].confidence {conf} outside [0,1]"
        if not isinstance(entry["issue_flags"], list):
            return False, f"tracks[{i}].issue_flags must be a list"
        # proposed_* allowed to be missing / None / string — model may omit when action=keep.
        for key in ("proposed_title", "proposed_artist"):
            if key in entry and entry[key] is not None and not isinstance(entry[key], str):
                return False, f"tracks[{i}].{key} must be string or null"
    return True, "ok"
