"""Config loader.

YAML lives on disk; this module produces a typed :class:`Config` object so the rest
of the pipeline never has to touch raw dicts.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class SeparatorRules(BaseModel):
    collapse: list[str] = Field(default_factory=lambda: ["_", "   ", "  "])
    replace_with_dash: list[str] = Field(default_factory=lambda: [" -- ", " – ", " — "])


class ConfidenceThresholds(BaseModel):
    min_auto_approve: float = 0.9
    low_confidence_flag: float = 0.6


class CacheConfig(BaseModel):
    skip_unchanged_albums: bool = True


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b-instruct"
    request_timeout_s: int = 120
    temperature: float = 0.1
    max_tracks_per_call: int = 40


class ProbeConfig(BaseModel):
    sample_positions: list[str] = Field(
        default_factory=lambda: ["first", "second", "middle", "last"]
    )
    clean_threshold: float = 0.9
    deterministic_threshold: float = 0.6


class Config(BaseModel):
    staging_root: Path
    jobs_dir: Path = Path("./jobs")
    supported_extensions: list[str] = Field(
        default_factory=lambda: [".flac", ".mp3", ".m4a", ".ogg", ".opus", ".wav"]
    )
    dry_run_default: bool = True
    remove_original_mix: bool = True
    remove_bpm_suffix: bool = True
    junk_prefixes: list[str] = Field(default_factory=list)
    junk_suffixes: list[str] = Field(default_factory=list)
    scene_markers: list[str] = Field(default_factory=list)
    separator_rules: SeparatorRules = Field(default_factory=SeparatorRules)
    confidence: ConfidenceThresholds = Field(default_factory=ConfidenceThresholds)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)


def load_config(path: Path) -> Config:
    """Read a YAML file and return a validated Config."""
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)


def default_config_path() -> Path:
    """Conventional location: ./config.yaml in the current working directory."""
    return Path.cwd() / "config.yaml"
