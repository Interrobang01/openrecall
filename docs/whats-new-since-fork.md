# What's new since this fork

This document summarizes major functionality added in this fork, with special attention to the two overhaul commits:

- `6cfd0b2` — **ui overhaul**
- `52b7ec8` — **more overhaul and features, yay**

It focuses on user-visible and operator-visible changes (UI, capture/storage, search, configuration, reliability, and performance controls).

## High-level additions

- AV1 segment-based storage backend with compressed thumbnail workflow.
- Multi-monitor-aware timeline and search UX improvements.
- Stronger capture controls (pause/resume, pause forever, hotkeys, hard stop).
- Runtime configuration page backed by `openrecall_config.json`.
- Search upgrades (metric selector, expression queries, phrase handling).
- OCR architecture controls for runtime tuning.
- Startup resilience: ffmpeg capability checks, segment decode fallbacks, quarantine/cleanup for unreadable recent segments.
- Metrics and status APIs/UI to observe performance and storage.

## Overhaul commit 1: `6cfd0b2` (ui overhaul)

Major additions inferred from this overhaul:

- Expanded web UI with richer controls and utility actions:
  - timeline date range controls,
  - monitor ordering options,
  - modal navigation UX,
  - direct links to config/metrics and storage folder actions.
- Search UX and scoring upgrades:
  - metric selector in `/search`,
  - improved score formatting,
  - expression-style embedding query support,
  - exact phrase extraction support.
- API and diagnostics surface expansion:
  - `/api/stats`, `/api/status`, `/api/recovery-status`, `/metrics`.
- Database and timeline plumbing for AV1/thumb-first data flow:
  - segment + thumbnail metadata retrieval optimized for timeline/search rendering.

## Overhaul commit 2: `52b7ec8` (more overhaul and features, yay)

Major additions inferred from this overhaul:

- Global hotkeys integration (`openrecall/hotkeys.py`) for fast capture control:
  - pause 5m,
  - pause 30m,
  - pause forever,
  - resume.
- Runtime configuration growth in `openrecall/config.py`:
  - expanded `OPENRECALL_*` settings set,
  - persistent config writing/reading,
  - clearer runtime key management.
- Capture/runtime observability improvements:
  - richer capture-state telemetry,
  - stronger per-stage timing and status reporting.
- Additional resilience and startup/runtime safeguards in app/screenshot/utils paths.

## Adjacent major commits (since fork)

### Storage migration and media pipeline

- `7328a5b` — switched storage from per-frame WebP to AV1 segment pipeline + thumbnails.
- Segment/frame indexing support matured around AV1 usage:
  - segment lookup by frame index (`get_segment_frame_index`) to avoid timestamp drift issues.
- Added startup validation for ffmpeg AV1 capabilities:
  - requires `libsvtav1` encoder and `libdav1d` or `libaom-av1` decoder.

### OCR and model/runtime tuning

- `4a8caeb` — faster OCR path and package updates.
- Added OCR architecture/provider controls:
  - detector/recognizer architecture overrides,
  - provider and thread settings,
- NLP fallback hardening (`5b700ce`):
  - improved embedding model load handling,
  - explicit CPU fallback behavior.

### UI personalization and transparency

- `59e45d6` and earlier transparency-related commits:
  - personalization and visibility/transparency-oriented UI improvements,
  - additional UI controls around capture state and readability.

### Capture and reliability improvements

- `94b6122` — dual monitor fix.
- `d5df5fd` + `cd9826f` — expanded verbosity/logging controls.
- Additional fixes across app/search/capture code paths to improve runtime stability.

## Current routes and controls added in this fork

### Core pages

- `/` — timeline page
- `/search` — search page
- `/config` — runtime configuration page
- `/metrics` — performance dashboard

### Media and operational endpoints

- `/frame` — frame extraction endpoint
- `/api/stats` — storage/database stats
- `/api/status` — capture runtime status
- `/api/recovery-status` — startup recovery summary
- `/open-folder` — open storage folder from UI

### Capture control endpoints

- `/api/capture/pause` (POST)
- `/api/capture/pause-forever` (POST)
- `/api/capture/resume` (POST)
- `/api/hard-stop` (POST)

## Runtime options added/expanded

### CLI

- `--storage-path`
- `--primary-monitor-only`

### Environment/config keys (high impact)

- Storage/AV1:
  - `OPENRECALL_STORAGE_BACKEND`
  - `OPENRECALL_FFMPEG_BIN`
  - `OPENRECALL_AV1_CRF`
  - `OPENRECALL_AV1_PRESET`
  - `OPENRECALL_AV1_PLAYBACK_FPS`
  - `OPENRECALL_AV1_SEGMENT_SECONDS`
  - `OPENRECALL_AV1_SEGMENT_FRAMES`
  - `OPENRECALL_THUMB_QUALITY`
  - `OPENRECALL_THUMB_MAX_DIMENSION`
- Capture/perf/logging:
  - `OPENRECALL_CAPTURE_INTERVAL_SECONDS`
  - `OPENRECALL_SIMILARITY_FRAME_WIDTH`
  - `OPENRECALL_VERBOSE_CAPTURE_LOGS`
- OCR/embeddings:
  - `OPENRECALL_EMBEDDING_DEVICE`
  - `OPENRECALL_OCR_DEVICE`
  - `OPENRECALL_OCR_CPU_THREADS`
  - `OPENRECALL_OCR_DET_ARCH`
  - `OPENRECALL_OCR_RECO_ARCH`
- Privacy/hotkeys:
  - `OPENRECALL_BLACKLIST_WINDOWS`
  - `OPENRECALL_BLACKLIST_WORDS`
  - `OPENRECALL_HOTKEY_PAUSE_5M`
  - `OPENRECALL_HOTKEY_PAUSE_30M`
  - `OPENRECALL_HOTKEY_PAUSE_FOREVER`
  - `OPENRECALL_HOTKEY_RESUME`

## Notes

- This summary is intentionally feature-focused rather than a full line-by-line changelog.
- For exact implementation details, inspect `openrecall/app.py`, `openrecall/config.py`, `openrecall/screenshot.py`, `openrecall/ocr.py`, `openrecall/nlp.py`, and related tests.
