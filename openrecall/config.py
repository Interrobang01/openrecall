import argparse
import os
import subprocess
import sys
from typing import List, Optional

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


OPENRECALL_STORAGE_BACKEND = (os.getenv("OPENRECALL_STORAGE_BACKEND") or "av1_hybrid").strip()
OPENRECALL_FFMPEG_BIN = (os.getenv("OPENRECALL_FFMPEG_BIN") or "ffmpeg").strip()
OPENRECALL_AV1_CRF = _get_env_int("OPENRECALL_AV1_CRF", default=38, minimum=0, maximum=63)
OPENRECALL_AV1_PRESET = (os.getenv("OPENRECALL_AV1_PRESET") or "8").strip()
OPENRECALL_AV1_SEGMENT_SECONDS = _get_env_float(
    "OPENRECALL_AV1_SEGMENT_SECONDS",
    default=120.0,
    minimum=1.0,
)
OPENRECALL_THUMB_QUALITY = _get_env_int(
    "OPENRECALL_THUMB_QUALITY",
    default=8,
    minimum=1,
    maximum=100,
)
OPENRECALL_THUMB_MAX_DIMENSION = _get_env_int(
    "OPENRECALL_THUMB_MAX_DIMENSION",
    default=320,
    minimum=64,
    maximum=4096,
)


if args.storage_path:
    appdata_folder = args.storage_path
else:
    appdata_folder = get_appdata_folder()
db_path = os.path.join(appdata_folder, "recall.db")

media_path = os.path.join(appdata_folder, "media")
segments_path = os.path.join(media_path, "segments")
thumbnails_path = os.path.join(media_path, "thumbnails")

# Backward-friendly alias used by older code paths, now points to thumbnails.
screenshots_path = thumbnails_path

for storage_dir in (appdata_folder, media_path, segments_path, thumbnails_path):
    os.makedirs(storage_dir, exist_ok=True)


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
