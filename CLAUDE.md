# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo layout

The entire project lives under `music_normalizer/`. All commands below are run from that directory (it holds its own `pyproject.toml`, `config.yaml`, `src/`, `tests/`, and `jobs/`). The repo root is otherwise empty.

## Commands

```bash
cd music_normalizer

# Install (requires Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Test / lint / typecheck
pytest
pytest tests/test_rules.py::test_name    # single test
ruff check src tests
mypy src                                  # strict mode, per pyproject

# The CLI entrypoint (installed via project.scripts)
music-normalizer scan
music-normalizer suggest [--album <id>] [--force-llm | --no-llm]
music-normalizer review <album_id>
music-normalizer apply <album_id> [--no-dry-run]
music-normalizer report

# Bulk variants (across every album under jobs_dir)
music-normalizer export-pending
music-normalizer auto-approve --min-confidence 0.95 [--allow-llm]
music-normalizer review-all
music-normalizer apply-all [--no-dry-run]
```

The `suggest` command requires Ollama running locally for the LLM path (`ollama pull qwen3:4b-instruct-2507-q4_K_M && ollama serve`). If Ollama is unreachable the pipeline logs a warning and degrades to deterministic-only — tests do not require it.

`config.yaml` is loaded from the current working directory by default. `config.example.yaml` is the template.

## Architecture

### Safety-first invariants (do not break)

- Only `TITLE` and `ARTIST` tags are ever written. Album, album-artist, year, genre, label, filename, etc. are read-only. `tags.WRITABLE_EXTENSIONS` excludes `.wav` deliberately — mutagen's WAV surface is inconsistent, so `apply` raises an explicit error rather than risk a silent no-op.
- Nothing is written to disk until `review` produces `review_decisions.json` *and* `apply` is run. `apply` defaults to `--dry-run` unless `config.dry_run_default` is overridden.
- Rules engine only proposes a fix when it can do so with high confidence. When in doubt it returns input unchanged and lets the classifier/LLM decide. No rule invents text — every transform is a strip or a case change.
- Capitalization is a *soft* concern: `rules._needs_case_fix` only touches all-lowercase or all-uppercase strings, never rewrites already-mixed casing.
- Artist extraction from `Artist - Title` patterns requires album-level corroboration (`rules.album_supports_artist_split`): at least 2 tracks in the album must show the pattern before any track gets split. A single ambiguous dash is left for manual review.

### Pipeline

```
scanner → grouper → rules ─┐
                           ├── probe → classifier ──► deterministic
                           │                     └──► LLM (ollama)
                           └── validator ──► suggestions.json
                                              │
                                              ▼
                           review (TUI) ──► review_decisions.json
                                              │
                                              ▼
                           apply ──► audio files + apply.log.json
```

`suggestions.suggest_for_album` is the top-level orchestrator. It runs the probe, the deterministic rules, asks the classifier, then dispatches to `_keep_all` / `_from_candidates` / `_ask_llm` based on `AlbumClassification`. Every path ends in `validator.validate_suggestions`.

### Classification gates (`classifier.py`)

| Class         | Requires                                                                  |
| ------------- | ------------------------------------------------------------------------- |
| CLEAN         | `probe ≥ clean_threshold` AND rules proposed no changes                   |
| DETERMINISTIC | `probe ≥ deterministic_threshold` AND every candidate confident (≥0.75) AND no remaining blocking flags |
| BROKEN        | >50% of tracks have unreadable/generic titles after cleanup               |
| MIXED         | everything else → LLM is consulted                                        |

Blocking flags: `UNREADABLE_TITLE`, `GENERIC_TRACK_NAME`, `EMBEDDED_ALBUM_NAME`, `SUSPICIOUS_SWAP_ARTIST_TITLE`.

### Issue flags: `original` vs `remaining`

Every `Suggestion` carries two sets (`models.py`):

- `original_issue_flags` — what was wrong on the raw tags (pre-cleanup). Kept for audit and the review UI.
- `remaining_issue_flags` — what still looks wrong after the deterministic pass. **Classifier, confidence scoring, and downstream logic must read this set**, not `issue_flags`. `issue_flags` is a back-compat alias that points at `remaining_issue_flags`.

### LLM path (`ollama_client.py` + `prompts.py`)

The album is chunked into batches of at most `ollama.max_tracks_per_call`. Each response is validated against the expected schema via `prompts.validate_response` **before** any `Suggestion` is constructed. A chunk that fails validation falls back to deterministic output for that chunk only — the rest of the album still benefits from LLM cleanup. Tracks the LLM omits entirely also fall back to deterministic.

### Data model contract (`models.py`)

The Pydantic models are the contract between modules. Every artifact written under `jobs/<album_id>/` is a serialized one of these:

| File                     | Model              | Written by |
| ------------------------ | ------------------ | ---------- |
| `manifest.json`          | `AlbumManifest`    | scan       |
| `suggestions.json`       | `AlbumSuggestions` | suggest    |
| `review.csv`             | (flat CSV)         | suggest    |
| `review_decisions.json`  | `list[ReviewDecision]` | review |
| `apply.log.json`         | `list[ApplyResult]` | apply     |

`AlbumManifest.fingerprint` is computed from sorted `(path, mtime, size)` tuples. `scan` skips albums whose fingerprint matches the prior manifest when `cache.skip_unchanged_albums` is true — any file change invalidates the cache, which is the safety-first behavior we want.

`PENDING_TRACK_ACTIONS` (= `{PROPOSE_FIX, UNREADABLE_NEEDS_MANUAL_REVIEW}`) and `AUTO_APPROVE_NOTE` are shared constants used across `bulk`, `report`, and `review` so everyone agrees on which rows still need a human decision and how to distinguish auto-approved from hand-approved rows.

### Review is resumable

`review.review_album` persists `review_decisions.json` **after every track**, and skips tracks that already have a decision on restart. Do not batch decisions at the end — that would regress resume-on-crash behavior.

## Testing conventions

Tests cover the safety-critical layers: deterministic rules, probe, classifier, validator, tags, prompts, filename fallback, album corroboration, bulk. The LLM client and Typer CLI are intentionally kept thin and are not covered. The shared `config` fixture (`tests/conftest.py`) builds a `Config` with realistic junk-prefix/suffix/scene-marker lists — reuse it rather than re-building configs inline.
