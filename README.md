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

- **Time Travel**: Revisit and explore your past digital activities seamlessly across Windows, macOS, or Linux.
- **Local-First AI**: OpenRecall harnesses the power of local AI processing to keep your data private and secure.
- **Semantic Search**: Advanced local OCR interprets your history, providing robust semantic search capabilities.
- **Full Control Over Storage**: Your data is stored locally, giving you complete control over its management and security.

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

## Get Started

### Prerequisites
- Python 3.11
- MacOSX/Windows/Linux
- Git

To install:
```
python3 -m pip install --upgrade --no-cache-dir git+https://github.com/openrecall/openrecall.git
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

OpenRecall runs OCR through ONNX Runtime (OnnxTR) and embeddings through PyTorch.

Embedding acceleration is available only when your PyTorch build supports your GPU backend.

- NVIDIA: CUDA build of PyTorch
- Apple Silicon: MPS backend
- AMD on Linux: ROCm-compatible GPU **and** a ROCm-enabled PyTorch build (where `torch.version.hip` is set)

If `torch.cuda.is_available()` is false and `torch.version.hip` is `None`, OpenRecall runs on CPU.

You can tune performance with environment variables:

- `OPENRECALL_CAPTURE_INTERVAL_SECONDS` (default `3.0`, min `1.0`)
- `OPENRECALL_SIMILARITY_FRAME_WIDTH` (default `0` = disabled/original full-size behavior; min `0`)
- `OPENRECALL_OCR_MAX_DIMENSION` (default `0` = disabled/original full-size behavior; min `0`)
- `OPENRECALL_VERBOSE_CAPTURE_LOGS` (default `false`; set to `1`/`true` to enable stage/timing CLI prints)
- `OPENRECALL_EMBEDDING_DEVICE` (override embedding device, e.g. `cpu`)
- `OPENRECALL_OCR_DEVICE` (OCR provider preference: `auto`/`cpu`/`cuda`/`coreml`; default `auto`)
- `OPENRECALL_OCR_CPU_THREADS` (override ONNX Runtime CPU threads; default auto)
- `OPENRECALL_OCR_DET_ARCH` (default `db_mobilenet_v3_large`)
- `OPENRECALL_OCR_RECO_ARCH` (default `crnn_mobilenet_v3_small` for speed/quality balance)
- `OPENRECALL_OCR_AB_TEST` (set `1`/`true` to run secondary OCR model for live A/B metrics)
- `OPENRECALL_OCR_AB_DET_ARCH` (A/B detector override; default follows primary detector)
- `OPENRECALL_OCR_AB_RECO_ARCH` (A/B recognizer override; default `crnn_mobilenet_v3_large`)
- `OPENRECALL_STORAGE_BACKEND` (must be `av1_hybrid`)
- `OPENRECALL_FFMPEG_BIN` (default `ffmpeg`)
- `OPENRECALL_AV1_CRF` (default `38`, lower = larger files / higher quality)
- `OPENRECALL_AV1_PRESET` (default `8`, lower = slower encode / better compression)
- `OPENRECALL_AV1_SEGMENT_SECONDS` (default `120`, min `1.0`)
- `OPENRECALL_THUMB_QUALITY` (default `8`, range `1..100`)
- `OPENRECALL_THUMB_MAX_DIMENSION` (default `320`, range `64..4096`)

### AV1 backend requirements

OpenRecall now requires an ffmpeg build with:

- `libsvtav1` encoder
- and at least one AV1 decoder: `libdav1d` or `libaom-av1`

On startup, OpenRecall checks ffmpeg capabilities and exits with a clear error if requirements are missing.

Examples:

```bash
# Lower CPU usage (less frequent capture + smaller OCR input)
OPENRECALL_CAPTURE_INTERVAL_SECONDS=5 OPENRECALL_OCR_MAX_DIMENSION=960 python3 -m openrecall.app

# Force CPU explicitly
OPENRECALL_EMBEDDING_DEVICE=cpu OPENRECALL_OCR_DEVICE=cpu python3 -m openrecall.app
```

## Directly compare OCR text (small vs large)

If A/B testing is enabled, OpenRecall can expose the latest OCR text from both models.

1. Start OpenRecall with A/B enabled (example below compares small recognizer vs large recognizer):

```bash
OPENRECALL_OCR_AB_TEST=1 \
OPENRECALL_OCR_RECO_ARCH=crnn_mobilenet_v3_small \
OPENRECALL_OCR_AB_RECO_ARCH=crnn_mobilenet_v3_large \
python3 -m openrecall.app
```

2. Generate a capture (change screen content so a new frame is saved).

3. Fetch the latest side-by-side OCR payload:

```bash
curl -sS http://127.0.0.1:8082/api/ocr-ab-compare
```

Response fields:
- `primary_text`: text from the primary OCR model (usually the faster/smaller config)
- `ab_text`: text from the A/B model (usually the larger reference)
- `token_recall` and `char_similarity`: quality overlap metrics for that capture

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
