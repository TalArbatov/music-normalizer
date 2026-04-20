"""Artist casing policy is stricter than title casing.

Artist names are routinely stylized — ``2LAVE RAC3``, ``PILL MURRAY``,
``Psy-VitaMinD`` — and rewriting them is worse than leaving them alone. The
rule: only normalize plain all-lowercase alpha text. Anything with a digit,
anything all-uppercase, and anything already mixed-case stays verbatim.
"""

from __future__ import annotations

from pathlib import Path

from music_normalizer.models import TrackInfo
from music_normalizer.rules import normalize_track


def _t(filename: str, *, title: str | None = None, artist: str | None = None) -> TrackInfo:
    return TrackInfo(
        path=Path(f"/music/{filename}"),
        filename=filename,
        ext=Path(filename).suffix.lower(),
        title=title,
        artist=artist,
    )


# ---------------------------------------------------------------------------
# Stylized artists: preserve verbatim
# ---------------------------------------------------------------------------


def test_digits_in_artist_preserved(config):
    # The motivating case: filename-recovered artist has a digit.
    t = _t("3. 2LAVE RAC3 - We Are The Shamans.flac")
    r = normalize_track(t, config)
    assert r.proposed_artist == "2LAVE RAC3"


def test_all_uppercase_artist_preserved(config):
    # No digits, but user clearly chose ALL CAPS — leave it alone.
    t = _t("01 - PILL MURRAY - Ghostbusters.flac")
    r = normalize_track(t, config)
    assert r.proposed_artist == "PILL MURRAY"


def test_mixed_case_artist_preserved(config):
    # Intentional stylization — don't rewrite to "Psy-Vitamind".
    t = _t("05 - Psy-VitaMinD - Third Eye.flac")
    r = normalize_track(t, config)
    assert r.proposed_artist == "Psy-VitaMinD"


def test_digit_suffix_stylization_preserved(config):
    # deadmau5 / 65daysofstatic / m.i.a. style digit embedding.
    t = _t("02 - deadmau5 - Strobe.flac", title="Strobe")
    r = normalize_track(t, config, allow_artist_split=False)
    # Artist tag wasn't set; filename fallback runs iff title is missing, so
    # here we feed an explicit tag-style setup: provide artist directly.
    t2 = _t("02 - whatever.flac", title="Strobe", artist="deadmau5")
    r2 = normalize_track(t2, config, allow_artist_split=False)
    # No change proposed — casing left as-is.
    assert r2.proposed_artist is None


# ---------------------------------------------------------------------------
# Plain lowercase artists: still get fixed
# ---------------------------------------------------------------------------


def test_plain_lowercase_artist_is_title_cased(config):
    t = _t("01 - pill murray - Ghostbusters.flac")
    r = normalize_track(t, config)
    assert r.proposed_artist == "Pill Murray"


def test_plain_lowercase_with_dj_token_preserves_preserve_token(config):
    # _PRESERVE_TOKENS upper-cases DJ when fixing plain lowercase input.
    t = _t("04 - dj shadow - Midnight In A Perfect World.flac")
    r = normalize_track(t, config)
    assert r.proposed_artist == "DJ Shadow"


# ---------------------------------------------------------------------------
# Title casing unchanged by this patch
# ---------------------------------------------------------------------------


def test_title_still_title_cased_when_all_lowercase(config):
    t = _t("01 - Some Artist - talkin revolution.flac",
           title="talkin revolution", artist="Some Artist")
    r = normalize_track(t, config, allow_artist_split=False)
    assert r.proposed_title == "Talkin Revolution"


def test_title_still_title_cased_when_all_uppercase(config):
    # Titles with no digits that are all-uppercase still get softened —
    # only the artist path became more conservative.
    t = _t("01 - Some Artist - TALKIN REVOLUTION.flac",
           title="TALKIN REVOLUTION", artist="Some Artist")
    r = normalize_track(t, config, allow_artist_split=False)
    assert r.proposed_title == "Talkin Revolution"
