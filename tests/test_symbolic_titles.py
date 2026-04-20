"""Short symbolic titles (i, ii, iii, iv, ...) must not be rewritten.

The motivating album is ``caca4b150b34`` — a 2024 EP where every track is
titled as a repeated-i token with a BPM suffix (``i (170BPM)``,
``iii (190BPM)``, ``iiiii (200BPM)``). The pipeline used to:
- title-case "i" to "I" (then flag it as unreadable),
- raise ``bad_capitalization`` against lowercase "i",
- escalate to the LLM, which proposed ``iiiii`` -> ``iiii`` (destructive).

The fix: detect symbolic tokens and route them through the deterministic
cleanup (BPM strip) without touching the token itself; reject LLM edits that
mutate a symbolic title.
"""

from __future__ import annotations

from pathlib import Path

from music_normalizer.classifier import classify_album
from music_normalizer.issues import detect_issues, is_symbolic_title
from music_normalizer.models import (
    AlbumClassification,
    AlbumManifest,
    IssueFlag,
    Suggestion,
    SuggestionSource,
    TrackAction,
    TrackInfo,
)
from music_normalizer.probe import probe_album
from music_normalizer.rules import normalize_track, propose_for_album
from music_normalizer.validator import validate


# ---------------------------------------------------------------------------
# is_symbolic_title
# ---------------------------------------------------------------------------


def test_is_symbolic_title_matches_repeated_letter():
    for t in ("i", "ii", "iii", "iiii", "iiiii", "I", "II", "III", "IIIII"):
        assert is_symbolic_title(t), t


def test_is_symbolic_title_matches_roman_numerals():
    for t in ("iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "IV", "VIII"):
        assert is_symbolic_title(t), t


def test_is_symbolic_title_rejects_regular_titles():
    for t in ("Bugpowder", "Bloodbath", "Hello", "Nocturnal Visions",
              "abcdef", None, "", "   ", "iamlong"):
        assert not is_symbolic_title(t), repr(t)


def test_is_symbolic_title_rejects_mixed_with_punct():
    # The BPM suffix form is NOT symbolic; cleanup must strip the suffix first.
    assert not is_symbolic_title("i (170BPM)")


# ---------------------------------------------------------------------------
# Deterministic: BPM strip + preserve token verbatim
# ---------------------------------------------------------------------------


def _track(filename: str, *, title: str, artist: str) -> TrackInfo:
    return TrackInfo(
        path=Path(f"/music/{filename}"),
        filename=filename,
        ext=".flac",
        title=title,
        artist=artist,
    )


def test_strip_bpm_from_i_170bpm(config):
    t = _track("01 i (170BPM).flac", title="i (170BPM)", artist="01010010")
    r = normalize_track(t, config)
    assert r.proposed_title == "i"


def test_strip_bpm_from_iii_190bpm(config):
    t = _track("02 iii (190BPM).flac", title="iii (190BPM)", artist="01010010")
    r = normalize_track(t, config)
    assert r.proposed_title == "iii"


def test_strip_bpm_from_iiiii_200bpm(config):
    t = _track("03 iiiii (200BPM).flac", title="iiiii (200BPM)", artist="01010010")
    r = normalize_track(t, config)
    # Must preserve every 'i' — five of them.
    assert r.proposed_title == "iiiii"


def test_symbolic_lowercase_not_flagged_bad_capitalization(config):
    flags = detect_issues("i", "01010010", config)
    assert IssueFlag.BAD_CAPITALIZATION not in flags


def test_symbolic_lowercase_not_flagged_unreadable(config):
    flags = detect_issues("i", "01010010", config)
    assert IssueFlag.UNREADABLE_TITLE not in flags


def test_symbolic_title_preserved_verbatim_no_uppercase(config):
    # Without the exemption, `_needs_case_fix("i")` would rewrite to "I".
    t = _track("x.flac", title="i", artist="01010010")
    r = normalize_track(t, config)
    assert r.proposed_title in (None, "i")  # either unchanged or explicitly "i"


# ---------------------------------------------------------------------------
# End-to-end: caca4b150b34 classifies DETERMINISTIC (no LLM)
# ---------------------------------------------------------------------------


_TRACKS = [
    ("01010010 - 始まり(EP) - 01 i (170BPM).flac",     "i (170BPM)"),
    ("01010010 - 始まり(EP) - 02 iii (190BPM).flac",   "iii (190BPM)"),
    ("01010010 - 始まり(EP) - 03 iiiii (200BPM).flac", "iiiii (200BPM)"),
    ("01010010 - 始まり(EP) - 04 iiii (175BPM).flac",  "iiii (175BPM)"),
]


def _manifest() -> AlbumManifest:
    tracks = [
        TrackInfo(
            path=Path("/music") / fn,
            filename=fn,
            ext=".flac",
            title=title,
            artist="01010010",
            album="始まり(EP)",
            album_artist="01010010",
        )
        for fn, title in _TRACKS
    ]
    return AlbumManifest(
        album_id="caca4b150b34",
        root=Path("/music"),
        album_name="始まり(EP)",
        album_artist="01010010",
        tracks=tracks,
        fingerprint="c" * 40,
    )


def test_caca4b150b34_classifies_deterministic(config):
    m = _manifest()
    probe = probe_album(m, config)
    cands = propose_for_album(m.tracks, config)
    result = classify_album(probe, {str(k): v for k, v in cands.items()}, config)
    assert result.classification == AlbumClassification.DETERMINISTIC, (
        f"got {result.classification.value}: {result.reason}"
    )


# ---------------------------------------------------------------------------
# Validator: reject destructive LLM edits on symbolic titles
# ---------------------------------------------------------------------------


def _llm_suggestion(current: str, proposed: str) -> Suggestion:
    return Suggestion(
        track_path=Path("/music/03 iiiii (200BPM).flac"),
        current_title=current,
        current_artist="01010010",
        proposed_title=proposed,
        proposed_artist="01010010",
        action=TrackAction.PROPOSE_FIX,
        confidence=0.8,
        reason="LLM said so",
        source=SuggestionSource.LLM,
    )


def test_validator_rejects_iiiii_to_iiii(config):
    s = _llm_suggestion("iiiii", "iiii")
    v = validate(s, config)
    assert v.proposed_title == "iiiii"
    assert v.action == TrackAction.KEEP
    assert any("symbolic title" in n for n in v.validator_notes)


def test_validator_rejects_i_to_I(config):
    s = _llm_suggestion("i", "I")
    v = validate(s, config)
    assert v.proposed_title == "i"
    assert v.action == TrackAction.KEEP


def test_validator_allows_identical_symbolic_title(config):
    # No change proposed on the title — must not be marked rejected.
    s = _llm_suggestion("iii", "iii")
    v = validate(s, config)
    assert v.proposed_title == "iii"
    assert not any("symbolic" in n for n in v.validator_notes)
