"""Deterministic normalization rules.

The rules engine only proposes a fix when it can do so with high confidence. When
in doubt it returns the input unchanged and lets the classifier / LLM take over.

Minimal edits only: we never invent words. Every transform is either a strip or a
case change.
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import Config
from .issues import GENERIC_TITLE_RE, detect_issues, is_symbolic_title
from .models import DeterministicCandidate, IssueFlag, TrackInfo

# Words that should stay lowercase in title case unless first/last.
# Kept intentionally short — music metadata conventions capitalize most prepositions
# ("With", "From", "Into"), so we only lowercase the tiny connective words.
_STOPWORDS_LOWER = {
    "a", "an", "and", "as", "at", "but", "by", "in", "of", "on", "or", "the", "to",
}

# Acronyms / stylizations that should keep their exact casing. Case-insensitive
# match, case-sensitive replacement.
_PRESERVE_TOKENS = {
    "VA": "VA",
    "EP": "EP",
    "LP": "LP",
    "DJ": "DJ",
    "MC": "MC",
    "VIP": "VIP",
    "UK": "UK",
    "USA": "USA",
    "II": "II",
    "III": "III",
    "IV": "IV",
}

_LEADING_TRACK_NUM_RE = re.compile(r"^\s*\d{1,3}\s*[-._)\]]\s*|^\s*\d{1,3}\s+")

# Filename fallback patterns, in priority order. All require explicit dash+space
# separators to avoid eating legitimate title text that merely contains a digit.
_FILENAME_ARTIST_NUM_TITLE_RE = re.compile(
    r"^(?P<artist>.+?)\s+-\s+\d{1,3}\s+-\s+(?P<title>.+)$"
)
_FILENAME_NUM_ARTIST_TITLE_RE = re.compile(
    r"^\d{1,3}\s+-\s+(?P<artist>.+?)\s+-\s+(?P<title>.+)$"
)
_FILENAME_NUM_TITLE_RE = re.compile(r"^\d{1,3}\s+-\s+(?P<title>.+)$")
_FILENAME_ARTIST_TITLE_RE = re.compile(r"^(?P<artist>.+?)\s+-\s+(?P<title>.+)$")

# "10. Artist - Title", "3. Artist - Title", ". Artist - Title", "01) Artist - Title".
# The leading (optional) number plus required punctuation (`.`, `-`, `_`, `)`, `]`)
# is a strong structural anchor: the rest of the stem can be read as
# ``Artist - Title`` without album-level corroboration. Covers WAV->FLAC rips
# where the tag block is nearly empty but the filename carries the full name.
_FILENAME_NUM_PUNCT_ARTIST_TITLE_RE = re.compile(
    r"^\s*(?:\d{1,3})?\s*[.\-_)\]]+\s*(?P<artist>.+?)\s+-\s+(?P<title>.+)$"
)

# BPM/tempo junk at the end of a title. Three cases, applied in order:
# 1. Bracketed numeric segment — `(160)`, `[160]`, `(160bpm)`, `[190-220]`,
#    `(178-200 bpm)`. Numeric content inside brackets at end of a title is
#    essentially always tempo; years would be 4 digits so \d{2,3} is safe.
# 2. Bare trailing `160bpm`, `178-200 bpm`, `165 Bpm` preceded by whitespace
#    or a dash.
# 3. Dash variants inside ranges: `-`, `–`, `—` are all accepted.
_BPM_SUFFIX_RES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\s*[\(\[]\s*\d{2,3}(?:\s*[-–—]\s*\d{2,3})?\s*(?:bpm)?\s*[\)\]]\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"[\s\-]+\d{2,3}(?:\s*[-–—]\s*\d{2,3})?\s*bpm\s*$",
        re.IGNORECASE,
    ),
)


def parse_filename_fallback(
    filename: str, *, allow_two_part_split: bool = False
) -> tuple[str | None, str] | None:
    """Best-effort ``(artist, title)`` extraction from a filename stem.

    Used only when the title tag is missing or is a generic placeholder. The
    three-part patterns (``Artist - NN - Title`` and ``NN - Artist - Title``) run
    unconditionally because the embedded track number makes the structure
    unambiguous. The bare two-part ``Artist - Title`` form is gated on album-level
    corroboration (``allow_two_part_split``) since a single dash in a filename is
    too easy to misparse.

    Returns ``None`` when no pattern matches; returns ``(None, title)`` when only
    a title can be extracted (``NN - Title``).
    """
    base = Path(filename).stem
    # Underscores are often used as separators in scene rips; fold them first.
    stem = base.replace("_", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    if not stem:
        return None

    for pat in (
        _FILENAME_ARTIST_NUM_TITLE_RE,
        _FILENAME_NUM_ARTIST_TITLE_RE,
        _FILENAME_NUM_PUNCT_ARTIST_TITLE_RE,
    ):
        m = pat.match(stem)
        if m:
            artist = m.group("artist").strip()
            title = m.group("title").strip()
            if len(artist) >= 2 and len(title) >= 2:
                return artist, title

    m = _FILENAME_NUM_TITLE_RE.match(stem)
    if m:
        title = m.group("title").strip()
        if len(title) >= 2:
            return None, title

    if allow_two_part_split:
        m = _FILENAME_ARTIST_TITLE_RE.match(stem)
        if m:
            artist = m.group("artist").strip()
            title = m.group("title").strip()
            if len(artist) >= 2 and len(title) >= 2:
                return artist, title

    return None


def normalize_track(
    track: TrackInfo,
    config: Config,
    *,
    allow_artist_split: bool = True,
    album_prefix: str | None = None,
    album_artist: str | None = None,
    artist_from_prefix: str | None = None,
) -> DeterministicCandidate:
    """Apply deterministic fixes to a single track. Returns a candidate.

    ``allow_artist_split`` is set by :func:`propose_for_album` after it has checked
    that the album as a whole exhibits the ``Artist - Title`` pattern on enough of
    its sample tracks. When the corroboration fails we still run every other rule
    — we just don't silently invent an artist from one ambiguous dash.

    ``album_prefix``, when provided, is a junk string that recurs on most tracks
    of the album (e.g. ``"Dazkull Eghyrth - Chaos Of Redemption - "``). If the
    track's title starts with it, we strip it before anything else.

    ``album_artist`` is a secondary source for the per-track artist-prefix strip,
    used when the track-level artist tag is missing.

    ``artist_from_prefix`` is the leftmost segment of a detected album-wide
    prefix (e.g. ``"Kindzadza"`` from ``"Kindzadza - Album - "``). If the track's
    artist tag is empty and we actually stripped the album prefix, we backfill
    this value so the removed information isn't lost.
    """
    notes: list[str] = []
    title = track.title or ""
    artist = track.artist or ""

    original_title = title
    original_artist = artist
    original_flags = detect_issues(track.title, track.artist, config)

    # 0. Filename fallback when the title tag is missing or a bare placeholder.
    #    Typical case: WAV files that carry no tag, or scene rips where the
    #    "title" was dumped as "Track 01". The 3-part filename pattern carries
    #    enough structure (it includes a track number) to be trusted without
    #    corroboration; the 2-part form waits for album-wide support.
    if not title.strip() or GENERIC_TITLE_RE.match(title):
        fb = parse_filename_fallback(
            track.filename, allow_two_part_split=allow_artist_split
        )
        if fb is not None:
            fb_artist, fb_title = fb
            title = fb_title
            notes.append(f"derived title from filename: {fb_title!r}")
            # Never override an already-present artist tag — if the user's tag
            # says "Aardvarkk" and the filename says something else, trust the
            # tag. Only fill in when the tag is empty.
            if not artist.strip() and fb_artist:
                artist = fb_artist
                notes.append(f"derived artist from filename: {fb_artist!r}")

    # 1. Strip leading track numbers that leaked into the title.
    title, note = _strip_leading_track_number(title)
    if note:
        notes.append(note)

    # 2. Separator normalization (before any prefix match — lets "artist_-_title"
    #    become "artist - title" so prefix comparison works uniformly).
    title, note = _normalize_separators(title, config)
    if note:
        notes.append(note)

    # 3. Album-wide repeated prefix (e.g. "Artist - Album - " on every track).
    album_prefix_stripped = False
    if album_prefix:
        new_title, note = _strip_literal_prefix(title, album_prefix, "album-wide prefix")
        if note:
            notes.append(note)
            album_prefix_stripped = True
            title = new_title
        # Backfill artist from the stripped prefix when the tag is empty.
        if (
            album_prefix_stripped
            and not artist.strip()
            and artist_from_prefix
        ):
            artist = artist_from_prefix
            notes.append(f"inferred artist from album-wide prefix: {artist!r}")

    # 4. Per-track artist prefix (works even when the artist tag is already set).
    title, note = _strip_artist_prefix(title, artist, album_artist)
    if note:
        notes.append(note)

    # 5. Junk prefix/suffix.
    title, note = _strip_junk_prefixes(title, config.junk_prefixes)
    if note:
        notes.append(note)
    title, note = _strip_junk_suffixes(title, config.junk_suffixes)
    if note:
        notes.append(note)

    # 6. BPM/tempo suffix.
    if config.remove_bpm_suffix:
        title, note = _strip_bpm(title)
        if note:
            notes.append(note)

    # 7. Original Mix.
    if config.remove_original_mix:
        title, note = _strip_original_mix(title)
        if note:
            notes.append(note)

    # 8. Artist split. Only applied when caller says the album as a whole supports
    #    the pattern — a single-track split on ambiguous data is too risky.
    if allow_artist_split and _should_split_artist(title, artist):
        split = _split_artist_title(title)
        if split is not None:
            inferred_artist, inferred_title = split
            if not artist.strip():
                artist = inferred_artist
                notes.append(f"inferred artist from title: {inferred_artist!r}")
            elif artist.strip().lower() == title.strip().lower():
                artist = inferred_artist
                notes.append(f"replaced duplicate artist with split: {inferred_artist!r}")
            title = inferred_title

    # 9. Whitespace collapse + capitalization (soft).
    title = _collapse_whitespace(title)
    artist = _collapse_whitespace(artist)

    title = _smart_title_case(title) if _needs_case_fix(title) else title
    artist = _smart_title_case(artist) if _artist_needs_case_fix(artist) else artist

    # Nothing worth proposing?
    proposed_title = title if title and title != original_title else None
    proposed_artist = artist if artist and artist != original_artist else None

    # Remaining flags: run the detector against the *cleaned* strings.
    remaining_flags = detect_issues(title or None, artist or None, config)

    confidence = _score_candidate(
        original_title=original_title,
        original_artist=original_artist,
        new_title=title,
        new_artist=artist,
        flags=remaining_flags,
    )

    return DeterministicCandidate(
        track_path=track.path,
        proposed_title=proposed_title,
        proposed_artist=proposed_artist,
        # issue_flags kept as the remaining set for back-compat with callers that
        # didn't yet learn about the split.
        issue_flags=remaining_flags,
        original_issue_flags=original_flags,
        remaining_issue_flags=remaining_flags,
        parse_notes=notes,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------


def _strip_leading_track_number(s: str) -> tuple[str, str | None]:
    m = _LEADING_TRACK_NUM_RE.match(s)
    if not m:
        return s, None
    stripped = s[m.end():].lstrip()
    # Sanity: don't strip if what remains is too short (likely the whole title was numeric).
    if len(stripped) < 2:
        return s, None
    return stripped, "stripped leading track number"


def _normalize_separators(s: str, config: Config) -> tuple[str, str | None]:
    out = s
    changed = False
    for token in config.separator_rules.replace_with_dash:
        if token in out:
            out = out.replace(token, " - ")
            changed = True
    for token in config.separator_rules.collapse:
        if token in out:
            out = out.replace(token, " ")
            changed = True
    out = re.sub(r"\s+", " ", out).strip()
    return (out, "normalized separators" if changed else None)


_ARTIST_PREFIX_SEPARATORS: tuple[str, ...] = (" - ", " – ", " — ", " -- ", "_-_")


def _strip_literal_prefix(
    s: str, prefix: str, label: str
) -> tuple[str, str | None]:
    """Case-insensitive prefix strip. Returns the original ``s`` if stripping
    would leave an empty remainder — that would mean the prefix *is* the title."""
    if not prefix:
        return s, None
    if s.lower().startswith(prefix.lower()):
        rest = s[len(prefix) :].lstrip(" -_:")
        if rest.strip():
            return rest, f"stripped {label} {prefix.strip()!r}"
    return s, None


def _strip_artist_prefix(
    title: str, artist: str, album_artist: str | None
) -> tuple[str, str | None]:
    """Strip ``"<artist><sep>"`` from the start of ``title`` when it matches the
    current track's artist, or (as a fallback) the album artist.

    This runs even when the artist tag is already populated — many scene /
    bandcamp rips stamp the artist into both fields. The match is
    case-insensitive and only uses separators we recognize as explicit (dashes,
    not plain spaces) to avoid e.g. stripping a single-word artist that happens
    to be the first word of the title.
    """
    candidates: list[str] = []
    for c in (artist, album_artist or ""):
        c = c.strip()
        if c and c not in candidates:
            candidates.append(c)

    for cand in candidates:
        for sep in _ARTIST_PREFIX_SEPARATORS:
            prefix = cand + sep
            if title.lower().startswith(prefix.lower()):
                rest = title[len(prefix) :].lstrip(" -_:")
                if rest.strip():
                    return rest, f"stripped artist prefix {cand!r}"
    return title, None


def _strip_junk_prefixes(s: str, prefixes: list[str]) -> tuple[str, str | None]:
    low = s.lower().lstrip()
    for p in prefixes:
        if low.startswith(p.lower()):
            cut = len(p) + (len(s) - len(s.lstrip()))
            return s[cut:].lstrip(" -_:"), f"stripped junk prefix {p!r}"
    return s, None


def _strip_junk_suffixes(s: str, suffixes: list[str]) -> tuple[str, str | None]:
    low = s.lower().rstrip()
    for p in suffixes:
        if low.endswith(p.lower()):
            cut = len(s.rstrip()) - len(p)
            return s[:cut].rstrip(" -_:"), f"stripped junk suffix {p!r}"
    return s, None


def _strip_bpm(s: str) -> tuple[str, str | None]:
    """Strip trailing tempo junk. Runs each pattern repeatedly until stable so
    titles with layered junk (``Title [160] 178-200bpm``) collapse cleanly."""
    out = s
    stripped_any = False
    while True:
        matched = False
        for pat in _BPM_SUFFIX_RES:
            m = pat.search(out)
            if m:
                candidate = out[: m.start()].rstrip(" -_[(),")
                # Never strip all the way to empty — that would mean the whole
                # title was tempo junk, which is better left for manual review.
                if candidate.strip():
                    out = candidate
                    matched = True
                    stripped_any = True
        if not matched:
            break
    return (out, "stripped bpm/tempo suffix") if stripped_any else (s, None)


def _strip_original_mix(s: str) -> tuple[str, str | None]:
    patterns = [
        r"\s*[\(\[]\s*original\s*mix\s*[\)\]]\s*$",
        r"\s*-\s*original\s*mix\s*$",
        r"\s+original\s*mix\s*$",
    ]
    for pat in patterns:
        new = re.sub(pat, "", s, flags=re.IGNORECASE)
        if new != s:
            return new.rstrip(), "stripped 'Original Mix'"
    return s, None


def _should_split_artist(title: str, artist: str) -> bool:
    if " - " not in title:
        return False
    if not artist.strip():
        return True
    if artist.strip().lower() == title.strip().lower():
        return True
    return False


def _split_artist_title(title: str) -> tuple[str, str] | None:
    """Split "Artist - Title" on the first " - ". Reject if either side is tiny."""
    parts = title.split(" - ", 1)
    if len(parts) != 2:
        return None
    left, right = parts[0].strip(), parts[1].strip()
    if len(left) < 2 or len(right) < 2:
        return None
    # If right side also contains " - ", the split is ambiguous — bail and let LLM handle.
    if " - " in right:
        return None
    return left, right


def _collapse_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _needs_case_fix(s: str) -> bool:
    if not s or not any(c.isalpha() for c in s):
        return False
    if is_symbolic_title(s):
        return False
    if s == s.lower():
        return True
    # All-caps strings longer than a handful of chars are almost always wrong.
    if s == s.upper() and len(s) > 3 and any(c.isalpha() for c in s):
        return True
    return False


def _artist_needs_case_fix(s: str) -> bool:
    """Artist casing policy is stricter than title casing.

    Artist names are routinely stylized — ``2LAVE RAC3``, ``PILL MURRAY``,
    ``Psy-VitaMinD``, ``deadmau5`` — and blindly title-casing them is worse
    than leaving them alone. Only fix plain all-lowercase alpha text. Any
    digit, any uppercase letter, or any already-mixed casing means we keep
    the original verbatim.
    """
    if not s or not any(c.isalpha() for c in s):
        return False
    if any(c.isdigit() for c in s):
        return False
    stripped = s.strip()
    if stripped == stripped.lower() and stripped != stripped.upper():
        return True
    return False


def _smart_title_case(s: str) -> str:
    """Title case with stopword/acronym awareness. Keeps existing mixed casing alone."""
    tokens = re.split(r"(\s+|[-/])", s)
    out: list[str] = []
    word_index = 0
    word_total = sum(1 for tok in tokens if tok.strip() and tok not in {"-", "/"})
    for tok in tokens:
        if not tok.strip() or tok in {"-", "/"}:
            out.append(tok)
            continue
        low = tok.lower()
        preserve = _PRESERVE_TOKENS.get(tok.upper())
        if preserve is not None:
            out.append(preserve)
        elif low in _STOPWORDS_LOWER and 0 < word_index < word_total - 1:
            out.append(low)
        else:
            out.append(_capitalize_word(tok))
        word_index += 1
    return "".join(out)


def _capitalize_word(tok: str) -> str:
    if not tok:
        return tok
    # Preserve inner apostrophes like "don't" -> "Don't".
    return tok[0].upper() + tok[1:].lower()


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

_BLOCKING_FLAGS = {
    IssueFlag.GENERIC_TRACK_NAME,
    IssueFlag.UNREADABLE_TITLE,
    IssueFlag.SUSPICIOUS_SWAP_ARTIST_TITLE,
    IssueFlag.EMBEDDED_ALBUM_NAME,
}


def _score_candidate(
    *,
    original_title: str,
    original_artist: str,
    new_title: str,
    new_artist: str,
    flags: list[IssueFlag],
) -> float:
    # If any blocking flag is still present, require human review.
    if any(f in _BLOCKING_FLAGS for f in flags):
        return 0.3
    # No change made → nothing to score, caller will drop the candidate.
    if new_title == original_title and new_artist == original_artist:
        return 1.0
    # Structural change (split) is less safe than pure cleanup.
    if original_artist.strip() == "" and new_artist.strip() != "":
        return 0.75
    return 0.9


_PREFIX_DETECTION_SEPARATORS: tuple[str, ...] = (" - ", " – ", " — ", " -- ")
_MIN_PREFIX_LEN: int = 4
_PREFIX_SUPPORT_THRESHOLD: float = 0.6


def detect_common_prefix(
    titles: list[str],
    *,
    threshold: float = _PREFIX_SUPPORT_THRESHOLD,
    min_len: int = _MIN_PREFIX_LEN,
) -> str | None:
    """Find a repeated leading substring shared by ≥ ``threshold`` of ``titles``.

    The prefix must end at a separator boundary (`` - ``, ``_--_``, em-dash) so
    we never cut a word in half. Prefer the longest prefix that clears the
    support bar — that's what lets us strip both ``Artist - `` and the richer
    ``Artist - Album - `` forms.

    Returns ``None`` when we have too few tracks to generalize or when no
    candidate prefix meets the threshold.
    """
    cleaned = [t.strip() for t in titles if t and t.strip()]
    if len(cleaned) < 3:
        return None

    # Candidate prefixes: every "...sep" cut from every title. Using all tracks
    # (not just the first) means a single outlier won't starve us of candidates.
    candidates: set[str] = set()
    for title in cleaned:
        for sep in _PREFIX_DETECTION_SEPARATORS:
            if sep not in title:
                continue
            parts = title.split(sep)
            for i in range(1, len(parts)):
                prefix = sep.join(parts[:i]) + sep
                if len(prefix) >= min_len:
                    candidates.add(prefix)

    if not candidates:
        return None

    need = max(2, int(len(cleaned) * threshold))
    # Longest first so "Artist - Album - " wins over "Artist - " when both match.
    for cand in sorted(candidates, key=len, reverse=True):
        low = cand.lower()
        matches = sum(1 for t in cleaned if t.lower().startswith(low))
        if matches < need:
            continue
        # Every matching title must have a non-empty remainder, else the
        # "prefix" is really the whole title for some tracks.
        if any(
            not t[len(cand) :].strip()
            for t in cleaned
            if t.lower().startswith(low)
        ):
            continue
        return cand

    return None


def _leftmost_segment_of_prefix(prefix: str) -> str | None:
    """Extract the first separator-delimited segment of an album-wide prefix.

    ``"Kindzadza - "`` → ``"Kindzadza"``;
    ``"Dazkull Eghyrth - Chaos Of Redemption - "`` → ``"Dazkull Eghyrth"``.

    Returns ``None`` when the prefix is too short to be a plausible artist name.
    """
    for sep in _PREFIX_DETECTION_SEPARATORS:
        if sep in prefix:
            first = prefix.split(sep, 1)[0].strip()
            if len(first) >= 2:
                return first
            return None
    return None


def album_supports_artist_split(tracks: list[TrackInfo], config: Config) -> bool:
    """Is the ``Artist - Title`` pattern corroborated album-wide?

    Policy:
    - Look at the probe sample (first/second/middle/last) if it covers ≥2 tracks
      with artist empty or obviously-duplicated, else fall back to the full album.
    - A track "supports" the pattern when its title contains exactly one ``" - "``
      (so a single unambiguous split is possible) AND the artist field looks bad
      (empty or equals the title).
    - Require at least 2 supporting tracks before we allow deterministic splits.

    One ambiguous dash on a single track is not enough to rewrite metadata.
    """
    # Local import to avoid a circular dependency on probe.
    from .probe import _select_sample  # noqa: PLC0415

    sample = _select_sample(tracks, config.probe.sample_positions)
    pool = sample if len(sample) >= 2 else tracks

    supporting = 0
    for t in pool:
        title = (t.title or "").strip()
        artist = (t.artist or "").strip()
        # One and only one dash-separated pair.
        if title.count(" - ") != 1:
            continue
        left, right = (p.strip() for p in title.split(" - ", 1))
        if len(left) < 2 or len(right) < 2:
            continue
        artist_looks_bad = (
            not artist or artist.lower() == title.lower()
        )
        if artist_looks_bad:
            supporting += 1

    return supporting >= 2


def album_supports_filename_recovery(tracks: list[TrackInfo]) -> bool:
    """At least 2 tracks have filenames with strong, unambiguous structure.

    "Strong" means a pattern that :func:`parse_filename_fallback` decodes
    without relying on album-level corroboration — a three-part form or a
    leading number/punct + ``Artist - Title``. When two or more siblings show
    this, the album as a whole is recoverable from filenames, so we can also
    trust the bare two-part form on tracks whose filenames are weaker. This is
    the signal that flips WAV->FLAC rips (tags nearly empty, filenames rich)
    out of the BROKEN -> LLM path and back into deterministic handling.
    """
    supporting = 0
    for t in tracks:
        if parse_filename_fallback(t.filename, allow_two_part_split=False) is not None:
            supporting += 1
            if supporting >= 2:
                return True
    return False


def propose_for_album(
    tracks: list[TrackInfo], config: Config
) -> dict[Path, DeterministicCandidate]:
    """Run :func:`normalize_track` across a whole album.

    Two album-level decisions are made once up front so every track sees a
    consistent answer:
    - ``allow_split``: is ``Artist - Title`` corroborated on enough tracks —
      either via tags (:func:`album_supports_artist_split`) or via strongly
      structured filenames (:func:`album_supports_filename_recovery`)?
    - ``album_prefix``: what junk prefix recurs on most tracks and should be
      stripped album-wide?

    Also fall back to the majority album-artist tag when individual tracks lack
    one, so the per-track artist-prefix strip can still catch cases where one
    outlier track is missing its tag.
    """
    allow_split = (
        album_supports_artist_split(tracks, config)
        or album_supports_filename_recovery(tracks)
    )

    # Use titles that have had leading track numbers stripped so a numeric
    # prefix doesn't fool the common-prefix detector.
    preproc_titles: list[str] = []
    for t in tracks:
        stripped_title, _ = _strip_leading_track_number(t.title or "")
        preproc_titles.append(stripped_title)
    album_prefix = detect_common_prefix(preproc_titles)
    artist_from_prefix = _leftmost_segment_of_prefix(album_prefix) if album_prefix else None

    # Majority album_artist for the artist-prefix strip fallback.
    counts: dict[str, int] = {}
    for t in tracks:
        aa = (t.album_artist or "").strip()
        if aa:
            counts[aa] = counts.get(aa, 0) + 1
    majority_album_artist = max(counts, key=counts.get) if counts else None

    return {
        t.path: normalize_track(
            t,
            config,
            allow_artist_split=allow_split,
            album_prefix=album_prefix,
            album_artist=majority_album_artist,
            artist_from_prefix=artist_from_prefix,
        )
        for t in tracks
    }
