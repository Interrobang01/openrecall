# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenRecall is a privacy-first, open-source screen recording and semantic search tool (alternative to Microsoft Recall / Rewind.ai). It continuously captures screenshots, extracts text via OCR, generates embeddings, and stores everything locally in SQLite. Users browse a timeline or search semantically through a Flask web UI.

## Commands

```bash
# Install
pip install -e .

# Run the app (serves on http://localhost:8082)
python3 -m openrecall.app

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_database.py

# CLI options
python3 -m openrecall.app --storage-path /path/to/data --primary-monitor-only
```

## Architecture

**Background capture loop** (`screenshot.py:record_screenshots_thread`): Every 3 seconds, checks user activity, captures screenshots via `mss`, compares with previous using MSSIM (threshold 0.9). On change: saves WebP to disk → OCR via `ocr.py` → embedding via `nlp.py` → stores in SQLite via `database.py`.

**Web UI** (`app.py`): Flask app on port 8082. `/` serves a timeline slider over stored screenshots. `/search` performs semantic search by computing query embedding and ranking entries by cosine similarity.

**Key modules:**
- `config.py` — CLI args and platform-aware storage paths (appdata)
- `database.py` — SQLite wrapper using `Entry` namedtuple; embeddings stored as BLOBs
- `nlp.py` — `sentence-transformers` with `all-MiniLM-L6-v2` model (384-dim); `get_embedding()` and `cosine_similarity()`
- `ocr.py` — `python-doctr` with MobileNet for text extraction from screenshots
- `utils.py` — Platform-specific helpers (active app/window title, user activity detection) for Windows/macOS/Linux

## Code Style

Follows `.tinycoder/rules/python_style_guide.md`:
- PEP 8 with 4-space indentation
- Type hints on all functions
- Google-style docstrings
- `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
- f-strings for formatting, context managers for resources, specific exception types

## Platform Notes

- Python 3.11 required (see `.python-version`)
- Platform-specific dependencies: Windows needs `pywin32`/`psutil`, macOS needs `pyobjc`, Linux needs `xprop`/`xprintidle` system packages
- OCR uses the official `python-doctr==1.0.1` PyPI package (PyTorch backend only, no TensorFlow)

## Dependency Constraints

- **`numpy` must stay `<2.0.0`** — `python-doctr` declares this upper bound; attempting `numpy>=2` causes a hard install conflict
- **`transformers` must be `>=5.0.0`** — versions 4.57.x had a bug that unconditionally imported TensorFlow at startup (via the loss module chain), which crashed against the system's `tensorflow==2.9.3` + `protobuf==6.33.4`. Transformers 5.x fixed this.
- **Do not revert `python-doctr` to the old `koenvaneijk` git fork** — that fork was frozen at June 2024, used the removed `huggingface_hub.Repository` API, and always imported TensorFlow on startup
