# music-normalizer

Review-first CLI for cleaning up messy **TITLE** and **ARTIST** tags on niche
psytrance / darkpsy / psycore libraries. Designed to be run locally, with a local
Ollama instance, on a staging folder. Never renames files. Never touches album,
album artist, year, genre, or label.

## Philosophy

- **Rules first, LLM second.** A deterministic rules engine handles the easy cases;
  Ollama is only called when the album looks genuinely ambiguous.
- **Album-first, track-second.** A cheap 4-track probe classifies each album as
  *clean / deterministic / mixed / broken* before any expensive work.
- **Review before apply.** Everything the tool proposes lands in a reviewable
  JSON + CSV pair. Nothing is written to disk until you approve it.
- **Minimal edits.** When in doubt, we keep the original. Capitalization is a
  soft concern — we only fix all-lowercase or all-uppercase, never rewrite
  already-mixed casing.
- **Artist extraction needs corroboration.** We only split `Artist - Title`
  patterns when at least two tracks in the album show the same pattern. A single
  ambiguous dash is left for manual review.
- **Audit everything.** Every album gets its own `jobs/<id>/` directory with a
  manifest, suggestions, decisions, and an apply log.

## Classification gates

| Class         | Requires                                                                 |
| ------------- | ------------------------------------------------------------------------ |
| CLEAN         | probe ≥ `clean_threshold` AND no deterministic changes proposed          |
| DETERMINISTIC | probe ≥ `deterministic_threshold` AND all candidates confident AND no remaining blocking flags |
| BROKEN        | majority of tracks have unreadable/generic titles after cleanup          |
| MIXED         | everything else → LLM is consulted                                       |

Issue flags are tracked in two sets: `original_issue_flags` (what was wrong on
the raw tags) and `remaining_issue_flags` (what still looks wrong after the
deterministic pass). Confidence and classification use the remaining set; the
review UI shows both.

## Install

Requires Python 3.11+.

```bash
cd music_normalizer
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional: install and start [Ollama](https://ollama.com/) for the LLM path.

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama serve   # usually auto-starts
```

## Configure

```bash
cp config.example.yaml config.yaml
$EDITOR config.yaml  # point staging_root at your folder
```

Key options:

| Option                            | Meaning                                            |
| --------------------------------- | -------------------------------------------------- |
| `staging_root`                    | Folder to scan (recursive)                         |
| `jobs_dir`                        | Where per-album artifacts are written              |
| `dry_run_default`                 | If `true`, `apply` requires `--no-dry-run` to write |
| `remove_original_mix`             | Strip trailing "Original Mix"                      |
| `remove_bpm_suffix`               | Strip "(160 bpm)" style suffixes                   |
| `junk_prefixes` / `junk_suffixes` | Case-insensitive string lists                      |
| `confidence.min_auto_approve`     | Threshold for future auto-approve mode             |
| `cache.skip_unchanged_albums`     | Skip albums whose files are unchanged since last scan |
| `ollama.model`                    | Ollama model name                                  |
| `probe.clean_threshold`           | Probe score ≥ this → skip LLM                      |

## Usage

The pipeline has five stages. Run them in order per album (or in bulk).

```bash
# 1. Scan the staging folder and build per-album manifests.
music-normalizer scan

# 2. Produce suggestions. Uses deterministic rules first; escalates to Ollama
#    only for albums classified as MIXED or BROKEN.
music-normalizer suggest

#    Optional flags:
#      --album <id>     only this album
#      --force-llm      skip shortcuts, always call Ollama
#      --no-llm         deterministic-only (no network)

# 3. Review an album interactively. You get approve / skip / edit per track.
music-normalizer review <album_id>

# 4. Apply the approved decisions to the audio files. Dry-run by default.
music-normalizer apply <album_id>                    # dry-run
music-normalizer apply <album_id> --no-dry-run       # actually write tags

# 5. Summarize the whole jobs directory.
music-normalizer report
```

### Album id

Each album folder gets a short hash id (12 hex chars) derived from its path
relative to `staging_root`. You'll see it in the output of `scan` and `suggest`:

```
7c3f4d2e9a1b classification=deterministic probe=0.85 llm=false tracks=12
```

### Artifacts per album

Under `jobs/<album_id>/`:

| File                   | Written by | Contents                                     |
| ---------------------- | ---------- | -------------------------------------------- |
| `manifest.json`        | scan       | Tracks + fingerprint                         |
| `suggestions.json`     | suggest    | Proposed changes + classification + probe    |
| `review.csv`           | suggest    | Human-readable spreadsheet of the above      |
| `review_decisions.json`| review     | Your approve / skip / edit verdicts          |
| `apply.log.json`       | apply      | Before/after audit log (dry-run or real)     |

## Supported formats

Read:  FLAC, MP3, M4A/MP4, OGG (Vorbis), Opus, WAV
Write: FLAC, MP3, M4A/MP4, OGG (Vorbis), Opus

WAV is read but **not written** in v1 — mutagen's WAV surface is inconsistent
across players, so `apply` will raise an explicit error rather than risk a
silent no-op. Configure your staging folder accordingly or convert WAVs first.

## LLM behavior

When an album is escalated to Ollama, the album is chunked into batches of at
most `ollama.max_tracks_per_call` tracks. Each response is validated strictly
against the expected schema before any `Suggestion` is constructed; if a chunk
fails validation, that chunk falls back to deterministic output and the rest of
the album still benefits from LLM cleanup.

## Review UX

Decisions are persisted to `review_decisions.json` **after every track**, so
quitting with `q` or an unexpected crash never loses completed verdicts. Re-running
`music-normalizer review <album>` resumes — already-decided tracks are skipped.

## Development

```bash
pytest                 # run tests
ruff check src tests   # lint
mypy src               # type check
```

Tests cover the deterministic rules, probe, classifier, and validator — the
safety-critical layers. The LLM client and Typer CLI are intentionally kept thin.

## Architecture

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

## TODOs (post-v1)

- [ ] Web-backed metadata lookup (MusicBrainz) behind an explicit flag.
- [ ] Rename files (disabled in v1 by design).
- [ ] Album / album-artist / year normalization.
- [ ] Proper rollback: restore original tags from the apply log.
- [ ] Parallelize per-album `suggest` (currently serial).
- [ ] Richer TUI (full-screen list with per-field cursor).
- [ ] Auto-approve mode gated by `confidence.min_auto_approve`.
- [ ] JSON-schema validation of Ollama output before we even hand it to the validator.

## License

MIT.
