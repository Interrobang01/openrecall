```
   ____                   ____                  ____
  / __ \____  ___  ____  / __ \___  _________ _/ / /
 / / / / __ \/ _ \/ __ \/ /_/ / _ \/ ___/ __ `/ / /
/ /_/ / /_/ /  __/ / / / _, _/  __/ /__/ /_/ / / /
\____/ .___/\___/_/ /_/_/ |_|\___/\___/\__,_/_/_/
    /_/
```
Welcome to my vibecoded fork of OpenRecall. Use at your own risk. Our motto: Move fast and break things.

# Take Control of Your Digital Memory

OpenRecall is a fully open-source, privacy-first alternative to proprietary solutions like Microsoft's Windows Recall or Limitless' Rewind.ai. With OpenRecall, you can easily access your digital history, enhancing your memory and productivity without compromising your privacy.

## What does it do?

OpenRecall captures your digital history through regularly taken snapshots. Captures are stored as AV1 video segments with aggressively compressed WebP thumbnails, then analyzed with OCR and made searchable so you can quickly find specific information by typing relevant keywords into OpenRecall. You can also manually scroll back through your history to revisit past activities.

https://github.com/openrecall/openrecall/assets/16676419/cfc579cb-165b-43e4-9325-9160da6487d2

## Why Choose OpenRecall?

OpenRecall offers several key advantages over closed-source alternatives:

- **Transparency**: OpenRecall is 100% open-source, allowing you to audit the source code for potential backdoors or privacy-invading features.
- **Cross-platform Support**: OpenRecall works on Windows, macOS, and Linux, giving you the freedom to use it on your preferred operating system.
- **Privacy-focused**: Your data is stored locally on your device, no internet connection or cloud is required. In addition, you have the option to encrypt the data on a removable disk for added security, read how in our [guide](docs/encryption.md) here.
- **Hardware Compatibility**: OpenRecall is designed to work with a [wide range of hardware](docs/hardware.md), unlike proprietary solutions that may require specific certified devices.

<p align="center">
  <a href="https://twitter.com/elonmusk/status/1792690964672450971" target="_blank">
    <img src="images/black_mirror.png" alt="Elon Musk Tweet" width="400">
  </a>
</p>

## Features

- **AV1-first storage pipeline**
  - Stores captures as rolling AV1 segments (`.mkv`) plus aggressively compressed WebP thumbnails.
  - Tunable quality/performance knobs for AV1 (`CRF`, `preset`, playback FPS, segment rotation).
  - Startup ffmpeg capability checks (`libsvtav1` + AV1 decoder) with clear failure messages.

- **Timeline and frame recovery UX**
  - Date-range filtered timeline with monitor-aware grouping and modal expansion.
  - Thumbnail-first loading with lazy upgrade to full frame extraction.
  - Robust frame extraction fallback paths and client retries for in-progress segment writes.

- **Semantic and expression search**
  - Local OCR + local embeddings for private semantic search.
  - Advanced search dropdown in the top bar keeps the search field compact while exposing all filters.
  - Multiple search metrics: `cosine`, `dot`, `euclidean`, `manhattan`.
  - Date-range filter defaults to first/last recorded screenshot timestamps.
  - Focused-window substring filter (`app`/window title), plus monitor filter.
  - Proximity dedup filter compresses near-duplicate runs by keeping top-ranked representatives.
  - Proximity slider uses a logarithmic scale from `1s` up to `min(1 year, half of dataset span)`.
  - Supports quoted exact phrases and embedding expressions like `(queen) - (king) + (woman)`.

- **Capture controls and hotkeys**
  - Pause for 5m/30m, pause forever, and resume from the web UI.
  - Optional global hotkeys for pause/resume actions.
  - Live capture status indicator and recent capture telemetry.

- **Multi-monitor support**
  - Per-monitor capture entries with timeline rendering across monitors.
  - Optional single-display mode via `--primary-monitor-only`.
  - Reverse monitor ordering toggle in timeline views.

- **Runtime configuration and observability**
  - In-app config editor (`/config`) backed by `openrecall_config.json`.
  - Metrics dashboard (`/metrics`) and JSON status/storage APIs.
  - Storage badge, startup recovery badge, and open-storage-folder action from the UI.

- **OCR tuning**
  - Configurable OCR detector/recognizer architectures and execution providers.
  - Device and thread controls for OCR and embeddings.

- **Privacy controls and safety defaults**
  - Local-only storage with no required cloud component.
  - Blacklists for sensitive windows/terms with legacy-default migration.
  - Easy encrypted-volume usage through `--storage-path`.

- **Reliability and recovery**
  - Startup quarantine for unreadable tail segments with cleanup of related metadata.
  - CPU-safe fallbacks for unsupported acceleration paths.
  - Extra diagnostic logging controls for capture-stage timing.

## What's new in this fork

For a commit-by-commit summary of everything added since this fork (including both overhaul commits), see [docs/whats-new-since-fork.md](docs/whats-new-since-fork.md).

<p align="center">
  <img src="images/lisa_rewind.webp" alt="Lisa Rewind" width="400">
</p>


## Comparison



| Feature          | OpenRecall                    | Windows Recall                                  | Rewind.ai                              |
|------------------|-------------------------------|--------------------------------------------------|----------------------------------------|
| Transparency     | Open-source                   | Closed-source                                    | Closed-source                          |
| Supported Hardware | All                         | Copilot+ certified Windows hardware              | M1/M2 Apple Silicon                    |
| OS Support       | Windows, macOS, Linux         | Windows                                          | macOS                                  |
| Privacy          | On-device, self-hosted        | Microsoft's privacy policy applies               | Connected to ChatGPT                   |
| Cost             | Free                          | Part of Windows 11 (requires specialized hardware) | Monthly subscription                   |

## Quick links
- [Roadmap](https://github.com/orgs/openrecall/projects/2) and you can [vote for your favorite features](https://github.com/openrecall/openrecall/discussions/9#discussion-6775473)
- [FAQ](https://github.com/openrecall/openrecall/wiki/FAQ)
- [What changed in this fork](docs/whats-new-since-fork.md)

## Get Started

### Prerequisites
- Python 3.11
- MacOSX/Windows/Linux
- Git

Linux tray icon support also requires GTK/AppIndicator bindings at the system level:

```bash
sudo apt install python3-gi gir1.2-gtk-3.0 gir1.2-ayatanaappindicator3-0.1
```

If OpenRecall runs from a venv, ensure that environment can access those bindings
(`--system-site-packages`) or install `PyGObject` in the venv.

To install:
```
python3 -m pip install --upgrade --no-cache-dir git+https://github.com/Interrobang01/openrecall.git
```

This will almost certainly not work, so afterwards, run:
```
claude --model opus "Please help me fix my install of openrecall https://github.com/Interrobang01/openrecall"
```

To run:
```
python3 -m openrecall.app
```
Open your browser to:
[http://localhost:8082](http://localhost:8082) to access OpenRecall.

## Arguments
`--storage-path` (default: user data path for your OS): allows you to specify the path where the media (`media/segments`, `media/thumbnails`) and database should be stored. We recommend [creating an encrypted volume](docs/encryption.md) to store your data.

`--primary-monitor-only` (default: False): only record the primary monitor (rather than individual screenshots for other monitors)

## Performance & GPU notes

OpenRecall runs OCR and embeddings through ONNX Runtime.

Embedding acceleration depends on your ONNX Runtime execution providers.

- NVIDIA: `CUDAExecutionProvider`
- Apple Silicon: `CoreMLExecutionProvider`
- CPU fallback: `CPUExecutionProvider`

If a requested provider is unavailable, OpenRecall falls back to CPU.

You can tune performance with environment variables:

- `OPENRECALL_CAPTURE_INTERVAL_SECONDS` (default `60.0`, min `1.0`)
- `OPENRECALL_CAPTURE_STALL_SECONDS` (default `300`; set `0` to disable watchdog auto-restart on stalled `capturing`/`encoding_pending` states)
- `OPENRECALL_SIMILARITY_FRAME_WIDTH` (default `0` = disabled/original full-size behavior; min `0`)
- `OPENRECALL_VERBOSE_CAPTURE_LOGS` (default `false`; set to `1`/`true` to enable stage/timing CLI prints)
- `MALLOC_ARENA_MAX` (default startup value `2` when unset; lower can reduce retained RSS on glibc systems)
- `OPENRECALL_EMBEDDING_DEVICE` (embedding provider preference: `auto`/`cpu`/`cuda`/`coreml`; default `auto`)
- `OPENRECALL_EMBEDDING_MODEL` (embedding model name; default `sentence-transformers/all-MiniLM-L6-v2`)
- `OPENRECALL_OCR_DEVICE` (OCR provider preference: `auto`/`cpu`/`cuda`/`coreml`; default `auto`)
- `OPENRECALL_OCR_CPU_THREADS` (override ONNX Runtime CPU threads; default `2`)
- `OPENRECALL_OCR_DET_ARCH` (default `db_mobilenet_v3_large`)
- `OPENRECALL_OCR_RECO_ARCH` (default `crnn_mobilenet_v3_small` for speed/quality balance)
- `OPENRECALL_STORAGE_BACKEND` (must be `av1_hybrid`)
- `OPENRECALL_FFMPEG_BIN` (default `ffmpeg`)
- `OPENRECALL_AV1_CRF` (default `38`, lower = larger files / higher quality)
- `OPENRECALL_AV1_PRESET` (default `9`, lower = slower encode / better compression)
- `OPENRECALL_AV1_THREADS` (default `0` = ffmpeg default; set `1..N` to cap encoder threads)
- `OPENRECALL_AV1_SVTAV1_PARAMS` (default empty; raw `-svtav1-params` string, e.g. `lp=2:scd=0`)
- `OPENRECALL_AV1_PLAYBACK_FPS` (default `2.0`, min `0.1`; encoded segment framerate used by video players)
- `OPENRECALL_AV1_SEGMENT_FRAMES` (default `30`; min `1`; number of lossless full-res WebPs buffered before each AV1 segment flush)
- `OPENRECALL_THUMB_QUALITY` (default `8`, range `1..100`)
- `OPENRECALL_THUMB_MAX_DIMENSION` (default `320`, range `64..4096`)

### AV1 backend requirements

OpenRecall now requires an ffmpeg build with:

- `libsvtav1` encoder
- and at least one AV1 decoder: `libdav1d` or `libaom-av1`

On startup, OpenRecall checks ffmpeg capabilities and exits with a clear error if requirements are missing.

Examples:

```bash
# Lower CPU usage (less frequent capture)
OPENRECALL_CAPTURE_INTERVAL_SECONDS=5 python3 -m openrecall.app

# Force CPU explicitly
OPENRECALL_EMBEDDING_DEVICE=cpu OPENRECALL_OCR_DEVICE=cpu python3 -m openrecall.app
```

## Uninstall instructions

To uninstall OpenRecall and remove all stored data:

1. Uninstall the package:
   ```
   python3 -m pip uninstall openrecall
   ```

2. Remove stored data:
   - On Windows:
     ```
     rmdir /s %APPDATA%\openrecall
     ```
   - On macOS:
     ```
     rm -rf ~/Library/Application\ Support/openrecall
     ```
   - On Linux:
     ```
     rm -rf ~/.local/share/openrecall
     ```

Note: If you specified a custom storage path at any time using the `--storage-path` argument, make sure to remove that directory too.

## License

OpenRecall is released under the [AGPLv3](https://opensource.org/licenses/AGPL-3.0), ensuring that it remains open and accessible to everyone.
