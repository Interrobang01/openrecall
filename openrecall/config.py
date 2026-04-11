import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

parser = argparse.ArgumentParser(description="OpenRecall")

parser.add_argument(
    "--storage-path",
    default=None,
    help="Path to store the screenshots and database",
)

parser.add_argument(
    "--primary-monitor-only",
    action="store_true",
    help="Only record the primary monitor",
    default=False,
)

args, _unknown_args = parser.parse_known_args()


def _get_env_int(name: str, default: int, minimum: int, maximum: Optional[int] = None) -> int:
    """Reads an integer env var with bounds and fallback."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError:
        return default

    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _get_env_float(name: str, default: float, minimum: float) -> float:
    """Reads a float env var with bounds and fallback."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return max(minimum, float(raw_value))
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    """Reads a boolean env var with fallback."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def get_appdata_folder(app_name="openrecall"):
    if sys.platform == "win32":
        appdata = os.getenv("APPDATA")
        if not appdata:
            raise EnvironmentError("APPDATA environment variable is not set.")
        path = os.path.join(appdata, app_name)
    elif sys.platform == "darwin":
        home = os.path.expanduser("~")
        path = os.path.join(home, "Library", "Application Support", app_name)
    else:
        home = os.path.expanduser("~")
        path = os.path.join(home, ".local", "share", app_name)
    os.makedirs(path, exist_ok=True)
    return path


if args.storage_path:
    appdata_folder = args.storage_path
else:
    appdata_folder = get_appdata_folder()
db_path = os.path.join(appdata_folder, "recall.db")

media_path = os.path.join(appdata_folder, "media")
segments_path = os.path.join(media_path, "segments")
thumbnails_path = os.path.join(media_path, "thumbnails")
pending_frames_path = os.path.join(media_path, "pending_frames")

# Backward-friendly alias used by older code paths, now points to thumbnails.
screenshots_path = thumbnails_path

for storage_dir in (appdata_folder, media_path, segments_path, thumbnails_path, pending_frames_path):
    os.makedirs(storage_dir, exist_ok=True)


config_file_path = os.path.join(appdata_folder, "openrecall_config.json")


def _load_config_file(path: str) -> Dict[str, object]:
    """Loads JSON config map from disk, returning empty map on invalid/missing file."""
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as config_file:
            loaded = json.load(config_file)
    except (OSError, json.JSONDecodeError):
        return {}

    return loaded if isinstance(loaded, dict) else {}


_file_config = _load_config_file(config_file_path)


def _get_setting_raw(name: str):
    env_value = os.getenv(name)
    if env_value is not None:
        return env_value
    return _file_config.get(name)


def _get_config_int(
    name: str,
    default: int,
    minimum: int,
    maximum: Optional[int] = None,
) -> int:
    raw_value = _get_setting_raw(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except (ValueError, TypeError):
        return default

    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _get_config_float(name: str, default: float, minimum: float) -> float:
    raw_value = _get_setting_raw(name)
    if raw_value is None:
        return default

    try:
        return max(minimum, float(raw_value))
    except (ValueError, TypeError):
        return default


def _get_config_str(name: str, default: str) -> str:
    raw_value = _get_setting_raw(name)
    if raw_value is None:
        return default
    value = str(raw_value).strip()
    return value if value else default


def _get_config_bool(name: str, default: bool) -> bool:
    raw_value = _get_setting_raw(name)
    if raw_value is None:
        return default

    if isinstance(raw_value, bool):
        return raw_value

    normalized = str(raw_value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


OPENRECALL_STORAGE_BACKEND = _get_config_str("OPENRECALL_STORAGE_BACKEND", "av1_hybrid")
OPENRECALL_FFMPEG_BIN = _get_config_str("OPENRECALL_FFMPEG_BIN", "ffmpeg")
OPENRECALL_AV1_CRF = _get_config_int("OPENRECALL_AV1_CRF", default=38, minimum=0, maximum=63)
OPENRECALL_AV1_PRESET = _get_config_str("OPENRECALL_AV1_PRESET", "9")
OPENRECALL_AV1_THREADS = _get_config_int(
    "OPENRECALL_AV1_THREADS",
    default=0,
    minimum=0,
)
OPENRECALL_AV1_SVTAV1_PARAMS = _get_config_str(
    "OPENRECALL_AV1_SVTAV1_PARAMS",
    "",
)
OPENRECALL_AV1_PLAYBACK_FPS = _get_config_float(
    "OPENRECALL_AV1_PLAYBACK_FPS",
    default=2.0,
    minimum=0.1,
)
OPENRECALL_THUMB_QUALITY = _get_config_int(
    "OPENRECALL_THUMB_QUALITY",
    default=8,
    minimum=1,
    maximum=100,
)
OPENRECALL_THUMB_MAX_DIMENSION = _get_config_int(
    "OPENRECALL_THUMB_MAX_DIMENSION",
    default=320,
    minimum=64,
    maximum=4096,
)

OPENRECALL_CAPTURE_INTERVAL_SECONDS = _get_config_float(
    "OPENRECALL_CAPTURE_INTERVAL_SECONDS",
    default=60.0,
    minimum=1.0,
)
OPENRECALL_AV1_SEGMENT_FRAMES = _get_config_int(
    "OPENRECALL_AV1_SEGMENT_FRAMES",
    default=30,
    minimum=1,
)
OPENRECALL_SIMILARITY_FRAME_WIDTH = _get_config_int(
    "OPENRECALL_SIMILARITY_FRAME_WIDTH",
    default=0,
    minimum=0,
)
OPENRECALL_VERBOSE_CAPTURE_LOGS = _get_config_bool(
    "OPENRECALL_VERBOSE_CAPTURE_LOGS",
    default=False,
)
OPENRECALL_CAPTURE_STALL_SECONDS = _get_config_int(
    "OPENRECALL_CAPTURE_STALL_SECONDS",
    default=300,
    minimum=0,
)
OPENRECALL_BLACKLIST_LEGACY_DEFAULT_TERMS = (
    "bitwarden,Password Manager,tor,incognito,приватный просмотр"
)
OPENRECALL_BLACKLIST_DEFAULT_TERMS = (
    "bitwarden,Password Manager,Tor Browser,incognito,приватный просмотр"
)


def _normalize_blacklist_terms(raw_terms: str) -> str:
    """Normalizes comma-separated blacklist terms for comparison."""
    return ",".join(
        term.strip().lower()
        for term in (raw_terms or "").split(",")
        if term and term.strip()
    )


def _migrate_legacy_blacklist_default(raw_terms: str) -> str:
    """Rewrites legacy default blacklist values to the safer current defaults."""
    if (
        _normalize_blacklist_terms(raw_terms)
        == _normalize_blacklist_terms(OPENRECALL_BLACKLIST_LEGACY_DEFAULT_TERMS)
    ):
        return OPENRECALL_BLACKLIST_DEFAULT_TERMS
    return raw_terms


OPENRECALL_BLACKLIST_WINDOWS = _get_config_str(
    "OPENRECALL_BLACKLIST_WINDOWS",
    OPENRECALL_BLACKLIST_DEFAULT_TERMS,
)
OPENRECALL_BLACKLIST_WINDOWS = _migrate_legacy_blacklist_default(
    OPENRECALL_BLACKLIST_WINDOWS
)
OPENRECALL_BLACKLIST_WORDS = _get_config_str(
    "OPENRECALL_BLACKLIST_WORDS",
    OPENRECALL_BLACKLIST_DEFAULT_TERMS,
)
OPENRECALL_BLACKLIST_WORDS = _migrate_legacy_blacklist_default(
    OPENRECALL_BLACKLIST_WORDS
)
OPENRECALL_HOTKEY_PAUSE_5M = _get_config_str(
    "OPENRECALL_HOTKEY_PAUSE_5M",
    "<ctrl>+<shift>+<alt>+5",
)
OPENRECALL_HOTKEY_PAUSE_30M = _get_config_str(
    "OPENRECALL_HOTKEY_PAUSE_30M",
    "<ctrl>+<shift>+<alt>+0",
)
OPENRECALL_HOTKEY_PAUSE_FOREVER = _get_config_str(
    "OPENRECALL_HOTKEY_PAUSE_FOREVER",
    "<ctrl>+<shift>+<alt>+l",
)
OPENRECALL_HOTKEY_RESUME = _get_config_str(
    "OPENRECALL_HOTKEY_RESUME",
    "<ctrl>+<shift>+<alt>+p",
)

RUNTIME_CONFIG_KEYS = [
    "OPENRECALL_STORAGE_BACKEND",
    "OPENRECALL_FFMPEG_BIN",
    "OPENRECALL_AV1_CRF",
    "OPENRECALL_AV1_PRESET",
    "OPENRECALL_AV1_THREADS",
    "OPENRECALL_AV1_SVTAV1_PARAMS",
    "OPENRECALL_AV1_PLAYBACK_FPS",
    "OPENRECALL_THUMB_QUALITY",
    "OPENRECALL_THUMB_MAX_DIMENSION",
    "OPENRECALL_CAPTURE_INTERVAL_SECONDS",
    "OPENRECALL_AV1_SEGMENT_FRAMES",
    "OPENRECALL_SIMILARITY_FRAME_WIDTH",
    "OPENRECALL_VERBOSE_CAPTURE_LOGS",
    "OPENRECALL_CAPTURE_STALL_SECONDS",
    "OPENRECALL_BLACKLIST_WINDOWS",
    "OPENRECALL_BLACKLIST_WORDS",
    "OPENRECALL_HOTKEY_PAUSE_5M",
    "OPENRECALL_HOTKEY_PAUSE_30M",
    "OPENRECALL_HOTKEY_PAUSE_FOREVER",
    "OPENRECALL_HOTKEY_RESUME",
]


def get_runtime_config_values() -> Dict[str, object]:
    """Returns effective runtime config values currently loaded by this process."""
    return {
        "OPENRECALL_STORAGE_BACKEND": OPENRECALL_STORAGE_BACKEND,
        "OPENRECALL_FFMPEG_BIN": OPENRECALL_FFMPEG_BIN,
        "OPENRECALL_AV1_CRF": OPENRECALL_AV1_CRF,
        "OPENRECALL_AV1_PRESET": OPENRECALL_AV1_PRESET,
        "OPENRECALL_AV1_THREADS": OPENRECALL_AV1_THREADS,
        "OPENRECALL_AV1_SVTAV1_PARAMS": OPENRECALL_AV1_SVTAV1_PARAMS,
        "OPENRECALL_AV1_PLAYBACK_FPS": OPENRECALL_AV1_PLAYBACK_FPS,
        "OPENRECALL_THUMB_QUALITY": OPENRECALL_THUMB_QUALITY,
        "OPENRECALL_THUMB_MAX_DIMENSION": OPENRECALL_THUMB_MAX_DIMENSION,
        "OPENRECALL_CAPTURE_INTERVAL_SECONDS": OPENRECALL_CAPTURE_INTERVAL_SECONDS,
        "OPENRECALL_AV1_SEGMENT_FRAMES": OPENRECALL_AV1_SEGMENT_FRAMES,
        "OPENRECALL_SIMILARITY_FRAME_WIDTH": OPENRECALL_SIMILARITY_FRAME_WIDTH,
        "OPENRECALL_VERBOSE_CAPTURE_LOGS": OPENRECALL_VERBOSE_CAPTURE_LOGS,
        "OPENRECALL_CAPTURE_STALL_SECONDS": OPENRECALL_CAPTURE_STALL_SECONDS,
        "OPENRECALL_BLACKLIST_WINDOWS": OPENRECALL_BLACKLIST_WINDOWS,
        "OPENRECALL_BLACKLIST_WORDS": OPENRECALL_BLACKLIST_WORDS,
        "OPENRECALL_HOTKEY_PAUSE_5M": OPENRECALL_HOTKEY_PAUSE_5M,
        "OPENRECALL_HOTKEY_PAUSE_30M": OPENRECALL_HOTKEY_PAUSE_30M,
        "OPENRECALL_HOTKEY_PAUSE_FOREVER": OPENRECALL_HOTKEY_PAUSE_FOREVER,
        "OPENRECALL_HOTKEY_RESUME": OPENRECALL_HOTKEY_RESUME,
    }


def write_runtime_config_file(new_values: Dict[str, object]) -> None:
    """Writes persistent JSON config file used on next process startup."""
    payload = {
        key: new_values[key]
        for key in RUNTIME_CONFIG_KEYS
        if key in new_values
    }
    with open(config_file_path, "w", encoding="utf-8") as config_file:
        json.dump(payload, config_file, indent=2, sort_keys=True)


def _run_ffmpeg_command(ffmpeg_bin: str, ffmpeg_args: List[str]) -> str:
    """Runs ffmpeg command and returns stdout/stderr output as text."""
    try:
        result = subprocess.run(
            [ffmpeg_bin, *ffmpeg_args],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ffmpeg binary '{ffmpeg_bin}' was not found. "
            "Set OPENRECALL_FFMPEG_BIN to a valid ffmpeg executable path."
        ) from exc

    output = f"{result.stdout}\n{result.stderr}".strip()
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg command failed with exit code {result.returncode}: "
            f"{ffmpeg_bin} {' '.join(ffmpeg_args)}\n{output}"
        )
    return output


def ffmpeg_smoke_check(ffmpeg_bin: Optional[str] = None) -> None:
    """Verifies that ffmpeg is callable."""
    selected_bin = ffmpeg_bin or OPENRECALL_FFMPEG_BIN
    _run_ffmpeg_command(selected_bin, ["-hide_banner", "-version"])


def check_ffmpeg_av1_capabilities(ffmpeg_bin: Optional[str] = None) -> None:
    """Validates required AV1 encoder/decoder capabilities for OpenRecall."""
    selected_bin = ffmpeg_bin or OPENRECALL_FFMPEG_BIN
    ffmpeg_smoke_check(selected_bin)

    encoder_output = _run_ffmpeg_command(selected_bin, ["-hide_banner", "-encoders"])
    decoder_output = _run_ffmpeg_command(selected_bin, ["-hide_banner", "-decoders"])

    missing_requirements = []
    if "libsvtav1" not in encoder_output:
        missing_requirements.append("encoder libsvtav1")
    if "libdav1d" not in decoder_output and "libaom-av1" not in decoder_output:
        missing_requirements.append("decoder libdav1d or libaom-av1")

    if missing_requirements:
        missing = ", ".join(missing_requirements)
        raise RuntimeError(
            "AV1 backend is required but ffmpeg is missing required support: "
            f"{missing}. "
            f"Checked binary: {selected_bin}."
        )
