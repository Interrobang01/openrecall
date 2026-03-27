import collections
import ctypes
import gc
import logging
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mss
import numpy as np
from PIL import Image

from openrecall.config import (
    OPENRECALL_CAPTURE_INTERVAL_SECONDS,
    OPENRECALL_AV1_CRF,
    OPENRECALL_AV1_PRESET,
    OPENRECALL_AV1_SVTAV1_PARAMS,
    OPENRECALL_AV1_THREADS,
    OPENRECALL_AV1_PLAYBACK_FPS,
    OPENRECALL_AV1_SEGMENT_FRAMES,
    OPENRECALL_BLACKLIST_WINDOWS,
    OPENRECALL_BLACKLIST_WORDS,
    OPENRECALL_FFMPEG_BIN,
    OPENRECALL_SIMILARITY_FRAME_WIDTH,
    OPENRECALL_THUMB_MAX_DIMENSION,
    OPENRECALL_THUMB_QUALITY,
    OPENRECALL_VERBOSE_CAPTURE_LOGS,
    args,
    pending_frames_path,
    segments_path,
    thumbnails_path,
)
from openrecall.database import insert_entry
from openrecall.nlp import get_embedding
from openrecall.ocr import extract_text_and_diagnostics_from_image
from openrecall.utils import (
    get_open_window_descriptors,
    get_active_app_name,
    get_active_window_title,
    is_user_active,
    send_system_notification,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Memory reclamation — return freed native pages to the OS after each cycle.
# ---------------------------------------------------------------------------
_libc: Optional[ctypes.CDLL] = None

def _reclaim_native_memory() -> None:
    """Run gc.collect and malloc_trim to return freed pages to the OS."""
    global _libc
    gc.collect()
    if platform.system() != "Linux":
        return
    try:
        if _libc is None:
            _libc = ctypes.CDLL("libc.so.6")
        _libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass  # Non-glibc system or unavailable — silently skip


# Shared state updated by the capture loop; read by Flask routes.
capture_state: Dict[str, Any] = {
    "last_capture_ts": 0,
    "last_segment_ts": 0,
    "captures_this_session": 0,
    "last_mssim": None,
    "recent_timings": collections.deque(maxlen=10),  # list of timing dicts
    "paused_until_ts": 0,
    "paused_indefinitely": False,
    "stop_requested": False,
    "status": "starting",
    "status_updated_ts": int(time.time()),
    "last_blocked_reason": "",
    "last_blocked_terms": [],
    "last_blocked_ts": 0,
    "last_error": "",
    "last_error_ts": 0,
    "error_count": 0,
}


def _set_capture_status(status: str) -> None:
    """Updates shared capture state status marker with timestamp."""
    capture_state["status"] = status
    capture_state["status_updated_ts"] = int(time.time())


def _set_capture_blocked_status(reason: str, terms: List[str]) -> None:
    """Stores blacklist-blocked details and marks capture state as blocked."""
    now_ts = int(time.time())
    capture_state["status"] = reason
    capture_state["status_updated_ts"] = now_ts
    capture_state["last_blocked_reason"] = reason
    capture_state["last_blocked_terms"] = sorted(set(terms))
    capture_state["last_blocked_ts"] = now_ts


def _record_capture_error(exc: Exception) -> None:
    """Stores and logs capture-loop errors without terminating the thread."""
    now_ts = int(time.time())
    message = f"{exc.__class__.__name__}: {exc}"
    capture_state["last_error"] = message
    capture_state["last_error_ts"] = now_ts
    capture_state["error_count"] = int(capture_state.get("error_count") or 0) + 1
    _set_capture_status("error")
    logger.exception("Capture loop error: %s", message)


def _parse_blacklist_terms(raw_terms: str) -> List[str]:
    """Parses comma-separated blacklist terms into lowercase tokens."""
    if not raw_terms:
        return []
    return [
        term.strip().lower()
        for term in raw_terms.split(",")
        if term and term.strip()
    ]


WINDOW_BLACKLIST_TERMS = _parse_blacklist_terms(OPENRECALL_BLACKLIST_WINDOWS)
OCR_BLACKLIST_TERMS = _parse_blacklist_terms(OPENRECALL_BLACKLIST_WORDS)


def _find_blacklist_matches(haystack: str, blacklist_terms: List[str]) -> List[str]:
    """Returns matched blacklist terms with word/phrase-aware case-insensitive matching."""
    normalized_haystack = re.sub(r"\s+", " ", (haystack or "").lower()).strip()
    if not normalized_haystack:
        return []

    matches: List[str] = []
    for term in blacklist_terms:
        normalized_term = re.sub(r"\s+", " ", (term or "").lower()).strip()
        if not normalized_term:
            continue

        if " " in normalized_term:
            if normalized_term in normalized_haystack:
                matches.append(term)
            continue

        word_pattern = rf"(?<!\w){re.escape(normalized_term)}(?!\w)"
        if re.search(word_pattern, normalized_haystack):
            matches.append(term)

    return matches


def _notify_capture_pause(message: str) -> None:
    """Sends best-effort system notification for capture pause/resume events."""
    send_system_notification("OpenRecall", message)


def set_capture_pause_for_seconds(pause_seconds: int) -> int:
    """Pauses capture loop for requested seconds and returns pause-until timestamp."""
    pause_until_ts = int(time.time()) + max(0, pause_seconds)
    capture_state["paused_until_ts"] = pause_until_ts
    capture_state["paused_indefinitely"] = False
    _set_capture_status("paused")
    pause_minutes = max(1, int(round(max(0, pause_seconds) / 60.0)))
    _notify_capture_pause(f"Capture paused for {pause_minutes} minute(s).")
    return pause_until_ts


def set_capture_pause_forever() -> None:
    """Pauses capture loop until explicitly resumed."""
    capture_state["paused_until_ts"] = 0
    capture_state["paused_indefinitely"] = True
    _set_capture_status("paused")
    _notify_capture_pause("Capture paused indefinitely.")


def clear_capture_pause() -> None:
    """Resumes capture loop by clearing active pause."""
    was_paused = is_capture_paused()
    capture_state["paused_until_ts"] = 0
    capture_state["paused_indefinitely"] = False
    _set_capture_status("running")
    if was_paused:
        _notify_capture_pause("Capture resumed.")


def request_capture_stop() -> None:
    """Requests the capture loop to terminate gracefully."""
    capture_state["stop_requested"] = True
    _set_capture_status("stopping")


def is_capture_paused(now_ts: Optional[int] = None) -> bool:
    """Returns whether capture is currently paused."""
    if bool(capture_state.get("paused_indefinitely")):
        return True
    now = int(time.time()) if now_ts is None else now_ts
    return now < int(capture_state.get("paused_until_ts") or 0)


def _capture_print(message: str) -> None:
    """Prints a flushed capture event message for CLI visibility."""
    if OPENRECALL_VERBOSE_CAPTURE_LOGS:
        print(f"[openrecall.capture] {message}", flush=True)

THUMB_MAX_DIMENSION = OPENRECALL_THUMB_MAX_DIMENSION
AV1_SEGMENT_FRAMES = OPENRECALL_AV1_SEGMENT_FRAMES


class MonitorAv1SegmentWriter:
    """Writes monitor frames into rolling AV1 segments via ffmpeg."""

    def __init__(self, monitor_id: int) -> None:
        self.monitor_id = monitor_id
        self.segment_filename: Optional[str] = None
        self.segment_started_at_ms: Optional[int] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._process: Optional[subprocess.Popen] = None
        self._segment_frame_index: int = 0

    def _is_expired(self) -> bool:
        if self.segment_started_at_ms is None:
            return True
        return self._segment_frame_index >= AV1_SEGMENT_FRAMES

    def close(self) -> None:
        """Closes the active ffmpeg segment process if present."""
        if self._process is None:
            return

        process = self._process
        self._process = None

        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
            finally:
                process.stdin = None

        stderr_bytes = b""
        try:
            _, stderr_bytes = process.communicate(timeout=30)
        except ValueError:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)

            if process.stderr is not None:
                stderr_bytes = process.stderr.read() or b""
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr_bytes = process.communicate()

        if process.returncode not in (0, None):
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                "ffmpeg AV1 writer exited with "
                f"code {process.returncode} for monitor {self.monitor_id}: {stderr_text}"
            )

    def _start(self, width: int, height: int, timestamp_ms: int) -> None:
        self.segment_started_at_ms = timestamp_ms
        self.segment_filename = f"{timestamp_ms}_m{self.monitor_id}.mkv"
        self._width = width
        self._height = height
        self._segment_frame_index = 0

        segment_filepath = os.path.join(segments_path, self.segment_filename)
        fps = OPENRECALL_AV1_PLAYBACK_FPS

        ffmpeg_command = [
            OPENRECALL_FFMPEG_BIN,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libsvtav1",
            "-crf",
            str(OPENRECALL_AV1_CRF),
            "-preset",
            OPENRECALL_AV1_PRESET,
        ]

        if OPENRECALL_AV1_THREADS > 0:
            ffmpeg_command.extend(["-threads", str(OPENRECALL_AV1_THREADS)])

        if OPENRECALL_AV1_SVTAV1_PARAMS:
            ffmpeg_command.extend(["-svtav1-params", OPENRECALL_AV1_SVTAV1_PARAMS])

        ffmpeg_command.append(segment_filepath)

        self._process = subprocess.Popen(
            ffmpeg_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self._process.stdin is None:
            raise RuntimeError(
                f"Failed to start ffmpeg AV1 writer for monitor {self.monitor_id}."
            )

    def write_frame(self, frame: np.ndarray, timestamp_ms: int) -> Tuple[str, int]:
        """Writes a frame and returns (segment_filename, segment_pts_ms)."""
        height, width = frame.shape[:2]
        should_rotate = (
            self._process is None
            or self._width != width
            or self._height != height
            or self._is_expired()
        )

        if should_rotate:
            self.close()
            self._start(width, height, timestamp_ms)

        if (
            self._process is None
            or self._process.stdin is None
            or self.segment_filename is None
            or self.segment_started_at_ms is None
        ):
            raise RuntimeError(
                f"AV1 writer process is not available for monitor {self.monitor_id}."
            )

        frame_bytes = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
        try:
            self._process.stdin.write(frame_bytes)
        except (BrokenPipeError, OSError) as exc:
            stderr_text = ""
            if self._process.stderr is not None:
                stderr_text = self._process.stderr.read().decode(
                    "utf-8",
                    errors="replace",
                )
            raise RuntimeError(
                f"ffmpeg AV1 writer failed for monitor {self.monitor_id}: {stderr_text}"
            ) from exc

        segment_pts_ms = int(round(self._segment_frame_index * (1000.0 / OPENRECALL_AV1_PLAYBACK_FPS)))
        self._segment_frame_index += 1
        return self.segment_filename, segment_pts_ms


@dataclass
class PendingSegmentFrame:
    """Stores metadata for a lossless pending frame awaiting segment encode."""

    thumb_filename: str
    pending_filepath: str


class MonitorPendingSegmentBuffer:
    """Batches lossless monitor frames and encodes to AV1 at threshold."""

    def __init__(self, monitor_id: int) -> None:
        self.monitor_id = monitor_id
        self.segment_filename: Optional[str] = None
        self.segment_started_at_ms: Optional[int] = None
        self._frames: List[PendingSegmentFrame] = []

    def _reset(self) -> None:
        self.segment_filename = None
        self.segment_started_at_ms = None
        self._frames = []

    def _ensure_active_segment(self, capture_id_ms: int) -> None:
        if self.segment_filename is not None and self.segment_started_at_ms is not None:
            return

        self.segment_started_at_ms = capture_id_ms
        self.segment_filename = f"{capture_id_ms}_m{self.monitor_id}.mkv"

    def add_frame(self, frame: np.ndarray, capture_id_ms: int, thumb_filename: str) -> Tuple[str, int, bool]:
        """Adds one lossless frame and returns segment metadata + flush hint."""
        self._ensure_active_segment(capture_id_ms)
        if self.segment_filename is None:
            raise RuntimeError("Pending segment filename is unavailable")

        pending_filepath = os.path.join(pending_frames_path, thumb_filename)
        _save_pending_fullres(frame, pending_filepath)

        frame_index = len(self._frames)
        segment_pts_ms = int(round(frame_index * (1000.0 / OPENRECALL_AV1_PLAYBACK_FPS)))
        self._frames.append(
            PendingSegmentFrame(
                thumb_filename=thumb_filename,
                pending_filepath=pending_filepath,
            )
        )

        return self.segment_filename, segment_pts_ms, len(self._frames) >= AV1_SEGMENT_FRAMES

    def flush_to_segment(self) -> Optional[str]:
        """Encodes pending lossless frames into a single AV1 segment."""
        if not self._frames:
            self._reset()
            return None

        if self.segment_filename is None or self.segment_started_at_ms is None:
            raise RuntimeError("Pending segment metadata is missing")

        writer = MonitorAv1SegmentWriter(self.monitor_id)
        segment_name = self.segment_filename

        try:
            for frame_data in self._frames:
                with Image.open(frame_data.pending_filepath) as pending_image:
                    frame_rgb = np.asarray(pending_image.convert("RGB"), dtype=np.uint8)
                writer.write_frame(frame_rgb, self.segment_started_at_ms)
        finally:
            writer.close()

        for frame_data in self._frames:
            try:
                os.remove(frame_data.pending_filepath)
            except OSError:
                pass

        self._reset()
        return segment_name


def _save_thumbnail(image: np.ndarray, capture_id_ms: int, monitor_id: int) -> str:
    """Creates and stores a compressed lossy WebP thumbnail for a capture."""
    thumb_image = Image.fromarray(image)
    width, height = thumb_image.size
    longest_side = max(width, height)

    if longest_side > THUMB_MAX_DIMENSION:
        scale = THUMB_MAX_DIMENSION / float(longest_side)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        thumb_image = thumb_image.resize(
            (resized_width, resized_height),
            Image.Resampling.LANCZOS,
        )

    thumb_filename = f"{capture_id_ms}_m{monitor_id}.webp"
    thumb_filepath = os.path.join(thumbnails_path, thumb_filename)
    thumb_image.save(
        thumb_filepath,
        format="WEBP",
        quality=OPENRECALL_THUMB_QUALITY,
        method=6,
        lossless=False,
    )
    return thumb_filename


def _save_pending_fullres(image: np.ndarray, pending_filepath: str) -> None:
    """Stores lossless full-resolution frame for pending batch segment encoding."""
    Image.fromarray(image).save(
        pending_filepath,
        format="WEBP",
        method=6,
        lossless=True,
    )


def _resize_for_ocr(image: np.ndarray) -> np.ndarray:
    """Returns original image for OCR (no downscaling)."""
    return image


def _prepare_similarity_frame(
    image: np.ndarray,
    target_width: int = OPENRECALL_SIMILARITY_FRAME_WIDTH,
) -> np.ndarray:
    """Builds a compact grayscale frame used for cheap similarity checks."""
    height, width = image.shape[:2]
    if target_width > 0 and width > target_width:
        scale = target_width / float(width)
        target_height = max(1, int(height * scale))
        image = np.asarray(
            Image.fromarray(image).resize(
                (target_width, target_height),
                Image.Resampling.BILINEAR,
            )
        )

    return (
        0.2989 * image[..., 0].astype(np.float32)
        + 0.5870 * image[..., 1].astype(np.float32)
        + 0.1140 * image[..., 2].astype(np.float32)
    )


def mean_structured_similarity_index(
    img1: np.ndarray, img2: np.ndarray, L: int = 255
) -> float:
    """Calculates the Mean Structural Similarity Index (MSSIM) between two frames.

    Args:
        img1: The first frame as a NumPy array.
        img2: The second frame as a NumPy array.
        L: The dynamic range of the pixel values (default is 255).

    Returns:
        The MSSIM value between the two frames (float between -1 and 1).
    """
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2

    mu1: float = np.mean(img1)
    mu2: float = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    ssim_index = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_index


def is_similar(
    img1: np.ndarray, img2: np.ndarray, similarity_threshold: float = 0.9
) -> bool:
    """Checks if two preprocessed frames are similar based on MSSIM.

    Args:
        img1: The first frame as a NumPy array.
        img2: The second frame as a NumPy array.
        similarity_threshold: The threshold above which images are considered similar.

    Returns:
        True if the images are similar, False otherwise.
    """
    similarity: float = mean_structured_similarity_index(img1, img2)
    return similarity >= similarity_threshold


def take_screenshots() -> List[Tuple[int, np.ndarray]]:
    """Takes screenshots of all connected monitors or just the primary one.

    Depending on the `args.primary_monitor_only` flag, captures either
    all monitors or only the primary monitor (index 1 in mss.monitors).

    Returns:
        A list of (monitor_id, screenshot) tuples, where each screenshot
        is a NumPy array (RGB).
    """
    screenshots: List[Tuple[int, np.ndarray]] = []
    with mss.mss() as sct:
        # sct.monitors[0] is the combined view of all monitors
        # sct.monitors[1] is the primary monitor
        # sct.monitors[2:] are other monitors
        monitor_indices = range(1, len(sct.monitors))  # Skip the 'all monitors' entry

        if args.primary_monitor_only:
            monitor_indices = [1]  # Only index 1 corresponds to the primary monitor

        for i in monitor_indices:
            # Ensure the index is valid before attempting to grab
            if i < len(sct.monitors):
                monitor_info = sct.monitors[i]
                # Grab the screen
                sct_img = sct.grab(monitor_info)
                # Convert to numpy array and change BGRA to RGB
                screenshot = np.array(sct_img)[:, :, [2, 1, 0]]
                screenshots.append((i, screenshot))
            else:
                # Handle case where primary_monitor_only is True but only one monitor exists (all monitors view)
                # This case might need specific handling depending on desired behavior.
                # For now, we just skip if the index is out of bounds.
                print(f"Warning: Monitor index {i} out of bounds. Skipping.")

    return screenshots


def record_screenshots_thread() -> None:
    """
    Continuously records screenshots, processes them, and stores relevant data.

    Checks for user activity and image similarity before processing and saving
    screenshots, associated OCR text, embeddings, and active application info.
    Runs in an infinite loop, intended to be executed in a separate thread.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    initial_screenshots = take_screenshots()
    last_similarity_frames: Dict[int, np.ndarray] = {
        monitor_id: _prepare_similarity_frame(screenshot)
        for monitor_id, screenshot in initial_screenshots
    }
    pending_segment_buffers: Dict[int, MonitorPendingSegmentBuffer] = {}
    _set_capture_status("running")
    _capture_print(f"screenshots_taken monitors={len(last_similarity_frames)}")
    was_paused_last_iteration = False

    def _flush_buffer_for_monitor(monitor_id: int) -> None:
        buffer = pending_segment_buffers.get(monitor_id)
        if buffer is None:
            return

        segment_name = buffer.flush_to_segment()
        if segment_name:
            capture_state["last_segment_ts"] = int(time.time())

    try:
        while True:
            try:
                if capture_state.get("stop_requested"):
                    _set_capture_status("stopped")
                    break

                now_ts = int(time.time())
                currently_paused = is_capture_paused(now_ts)
                if was_paused_last_iteration and not currently_paused:
                    _notify_capture_pause("Capture resumed (pause timer ended).")

                if currently_paused:
                    _set_capture_status("paused")
                    for monitor_id in list(pending_segment_buffers.keys()):
                        try:
                            _set_capture_status("encoding_pending")
                            _flush_buffer_for_monitor(monitor_id)
                        except RuntimeError as exc:
                            logger.warning("Failed flushing pending AV1 segment for monitor %s: %s", monitor_id, exc)
                        del pending_segment_buffers[monitor_id]
                    was_paused_last_iteration = True
                    time.sleep(1.0)
                    continue

                was_paused_last_iteration = False

                if WINDOW_BLACKLIST_TERMS:
                    open_window_descriptors = get_open_window_descriptors()
                    open_windows_haystack = "\n".join(open_window_descriptors)
                    matched_window_terms = _find_blacklist_matches(
                        open_windows_haystack,
                        WINDOW_BLACKLIST_TERMS,
                    )
                    if matched_window_terms:
                        _set_capture_blocked_status(
                            "blocked_window_blacklist",
                            matched_window_terms,
                        )
                        _capture_print(
                            "capture_blocked_by_window_blacklist "
                            f"terms={','.join(sorted(set(matched_window_terms)))}"
                        )
                        time.sleep(OPENRECALL_CAPTURE_INTERVAL_SECONDS)
                        continue

                if not is_user_active():
                    _set_capture_status("user_inactive")
                    time.sleep(OPENRECALL_CAPTURE_INTERVAL_SECONDS)
                    continue

                _set_capture_status("capturing")

                current_screenshots: List[Tuple[int, np.ndarray]] = take_screenshots()
                _capture_print(f"screenshots_taken monitors={len(current_screenshots)}")
                current_monitor_ids = {monitor_id for monitor_id, _ in current_screenshots}

                removed_monitors = set(pending_segment_buffers.keys()) - current_monitor_ids
                for removed_monitor_id in removed_monitors:
                    try:
                        _set_capture_status("encoding_pending")
                        _flush_buffer_for_monitor(removed_monitor_id)
                    except RuntimeError as exc:
                        logger.warning("Failed flushing pending AV1 segment for monitor %s: %s", removed_monitor_id, exc)
                    del pending_segment_buffers[removed_monitor_id]

                if set(last_similarity_frames.keys()) != current_monitor_ids:
                    _capture_print(
                        "monitor_layout_changed "
                        f"previous={len(last_similarity_frames)} current={len(current_screenshots)}"
                    )
                    last_similarity_frames = {
                        monitor_id: _prepare_similarity_frame(screenshot)
                        for monitor_id, screenshot in current_screenshots
                    }
                    _set_capture_status("monitor_layout_changed")
                    time.sleep(OPENRECALL_CAPTURE_INTERVAL_SECONDS)
                    continue

                for monitor_id, current_screenshot in current_screenshots:
                    t_frame_start = time.perf_counter()
                    current_similarity_frame = _prepare_similarity_frame(current_screenshot)
                    last_similarity_frame = last_similarity_frames[monitor_id]

                    mssim_val = mean_structured_similarity_index(
                        current_similarity_frame,
                        last_similarity_frame,
                    )
                    t_mssim_ms = (time.perf_counter() - t_frame_start) * 1000
                    capture_state["last_mssim"] = round(mssim_val, 4)
                    _capture_print(
                        f"similarity_checked monitor={monitor_id} mssim={mssim_val:.4f} "
                        f"mssim_ms={t_mssim_ms:.1f}"
                    )

                    if mssim_val < 0.9:
                        last_similarity_frames[monitor_id] = current_similarity_frame
                        timestamp = int(time.time())
                        capture_id_ms = int(time.time() * 1000)

                        _capture_print(f"ocr_start monitor={monitor_id} timestamp={timestamp}")
                        t_ocr_start = time.perf_counter()
                        ocr_input = _resize_for_ocr(current_screenshot)
                        text, ocr_diagnostics = extract_text_and_diagnostics_from_image(ocr_input)
                        t_ocr_ms = (time.perf_counter() - t_ocr_start) * 1000

                        active_app_name: str = get_active_app_name() or "Unknown App"
                        active_window_title: str = get_active_window_title() or "Unknown Title"

                        _capture_print(
                            f"ocr_stop monitor={monitor_id} timestamp={timestamp} "
                            f"ocr_ms={t_ocr_ms:.1f} text_len={len(text.strip())}"
                        )

                        app_title_matches = _find_blacklist_matches(
                            f"{active_app_name}\n{active_window_title}",
                            WINDOW_BLACKLIST_TERMS,
                        )
                        if app_title_matches:
                            _set_capture_blocked_status(
                                "blocked_active_window_blacklist",
                                app_title_matches,
                            )
                            _capture_print(
                                "capture_blocked_by_active_window_blacklist "
                                f"monitor={monitor_id} terms={','.join(sorted(set(app_title_matches)))}"
                            )
                            continue

                        ocr_matches = _find_blacklist_matches(text, OCR_BLACKLIST_TERMS)
                        if ocr_matches:
                            _set_capture_blocked_status(
                                "blocked_ocr_blacklist",
                                ocr_matches,
                            )
                            _capture_print(
                                "capture_blocked_by_ocr_blacklist "
                                f"monitor={monitor_id} terms={','.join(sorted(set(ocr_matches)))}"
                            )
                            continue

                        t_embed_ms = 0.0
                        t_encode_ms = 0.0
                        t_db_ms = 0.0
                        if text.strip():
                            _capture_print(f"embedding_start monitor={monitor_id} timestamp={timestamp}")
                            t_embed_start = time.perf_counter()
                            embedding: np.ndarray = get_embedding(text)
                            t_embed_ms = (time.perf_counter() - t_embed_start) * 1000
                            _capture_print(
                                f"embedding_stop monitor={monitor_id} timestamp={timestamp} "
                                f"embedding_ms={t_embed_ms:.1f}"
                            )

                            _capture_print(f"db_write_start monitor={monitor_id} timestamp={timestamp}")
                            t_encode_start = time.perf_counter()

                            thumb_filename = _save_thumbnail(
                                current_screenshot,
                                capture_id_ms,
                                monitor_id,
                            )

                            pending_buffer = pending_segment_buffers.setdefault(
                                monitor_id,
                                MonitorPendingSegmentBuffer(monitor_id),
                            )
                            segment_filename, segment_pts_ms, should_flush_segment = pending_buffer.add_frame(
                                current_screenshot,
                                capture_id_ms,
                                thumb_filename,
                            )

                            if should_flush_segment:
                                _set_capture_status("encoding_pending")
                                flushed_segment = pending_buffer.flush_to_segment()
                                if flushed_segment:
                                    capture_state["last_segment_ts"] = int(time.time())

                            t_encode_ms = (time.perf_counter() - t_encode_start) * 1000

                            t_db_start = time.perf_counter()
                            insert_entry(
                                text,
                                timestamp,
                                embedding,
                                active_app_name,
                                active_window_title,
                                monitor_id=monitor_id,
                                segment_filename=segment_filename,
                                segment_pts_ms=segment_pts_ms,
                                thumb_filename=thumb_filename,
                            )
                            t_db_ms = (time.perf_counter() - t_db_start) * 1000
                            _capture_print(
                                "db_write_stop "
                                f"monitor={monitor_id} timestamp={timestamp} "
                                f"encode_ms={t_encode_ms:.1f} db_ms={t_db_ms:.1f} "
                                f"segment={segment_filename} pts_ms={segment_pts_ms} "
                                f"thumb={thumb_filename}"
                            )
                        else:
                            _capture_print(
                                f"embedding_skipped monitor={monitor_id} timestamp={timestamp} reason=no_text"
                            )

                        total_ms = (time.perf_counter() - t_frame_start) * 1000
                        timing = {
                            "timestamp": timestamp,
                            "mssim_ms": round(t_mssim_ms, 1),
                            "ocr_ms": round(t_ocr_ms, 1),
                            "ocr_primary_ms": ocr_diagnostics.get("primary_ms"),
                            "embedding_ms": round(t_embed_ms, 1),
                            "encode_ms": round(t_encode_ms, 1),
                            "db_ms": round(t_db_ms, 1),
                            "total_ms": round(total_ms, 1),
                            "had_text": bool(text.strip()),
                        }
                        capture_state["recent_timings"].append(timing)
                        capture_state["last_capture_ts"] = timestamp
                        capture_state["captures_this_session"] += 1
                        _set_capture_status("captured")

                _reclaim_native_memory()
                _set_capture_status("running")
                time.sleep(OPENRECALL_CAPTURE_INTERVAL_SECONDS)
            except Exception as exc:
                _record_capture_error(exc)
                time.sleep(1.0)
    finally:
        for monitor_id, pending_buffer in pending_segment_buffers.items():
            try:
                _set_capture_status("encoding_pending")
                flushed_segment = pending_buffer.flush_to_segment()
                if flushed_segment:
                    capture_state["last_segment_ts"] = int(time.time())
            except RuntimeError as exc:
                logger.warning("Failed flushing pending AV1 segment for monitor %s: %s", monitor_id, exc)
        _set_capture_status("stopped")
