"""Shared fixtures."""

from __future__ import annotations

import pytest

from music_normalizer.config import Config


@pytest.fixture()
def config(tmp_path) -> Config:
    return Config.model_validate({
        "staging_root": str(tmp_path / "staging"),
        "jobs_dir": str(tmp_path / "jobs"),
        "junk_prefixes": ["va -", "[free]"],
        "junk_suffixes": ["(original mix)", "- original mix", "(free download)"],
        "scene_markers": ["webrip", "promo"],
        "separator_rules": {
            "collapse": ["_", "  "],
            "replace_with_dash": [" -- ", " – ", " — "],
        },
        "remove_original_mix": True,
        "remove_bpm_suffix": True,
    })
