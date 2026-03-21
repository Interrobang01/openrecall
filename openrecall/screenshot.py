import collections
import logging
import os
import time
from typing import Any, Dict, List

import mss
import numpy as np
from PIL import Image

from openrecall.config import screenshots_path, args
from openrecall.database import insert_entry
from openrecall.nlp import get_embedding
from openrecall.ocr import extract_text_from_image
from openrecall.utils import (
    get_active_app_name,
    get_active_window_title,
    is_user_active,
)

logger = logging.getLogger(__name__)

# Shared state updated by the capture loop; read by Flask routes.
capture_state: Dict[str, Any] = {
    "last_capture_ts": 0,
    "captures_this_session": 0,
    "last_mssim": None,
    "recent_timings": collections.deque(maxlen=10),  # list of timing dicts
}


def _capture_print(message: str) -> None:
    """Prints a flushed capture event message for CLI visibility."""
    print(f"[openrecall.capture] {message}", flush=True)


def _get_env_int(name: str, default: int, minimum: int) -> int:
    """Reads an integer env var with bounds and fallback."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return max(minimum, int(raw_value))
    except ValueError:
        logger.warning("Invalid %s value '%s'. Falling back to %s.", name, raw_value, default)
        return default


def _get_env_float(name: str, default: float, minimum: float) -> float:
    """Reads a float env var with bounds and fallback."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return max(minimum, float(raw_value))
    except ValueError:
        logger.warning("Invalid %s value '%s'. Falling back to %.2f.", name, raw_value, default)
        return default


CAPTURE_INTERVAL_SECONDS: float = _get_env_float(
    "OPENRECALL_CAPTURE_INTERVAL_SECONDS", default=3.0, minimum=1.0
)
SIMILARITY_FRAME_WIDTH: int = _get_env_int(
    "OPENRECALL_SIMILARITY_FRAME_WIDTH", default=0, minimum=0
)
OCR_MAX_DIMENSION: int = _get_env_int(
    "OPENRECALL_OCR_MAX_DIMENSION", default=0, minimum=0
)


def _resize_for_ocr(image: np.ndarray, max_dimension: int = OCR_MAX_DIMENSION) -> np.ndarray:
    """Resizes the image for OCR if its longest side is larger than max_dimension."""
    height, width = image.shape[:2]
    longest_side = max(height, width)

    if max_dimension <= 0:
        return image

    if longest_side <= max_dimension:
        return image

    scale = max_dimension / float(longest_side)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    resized_image = Image.fromarray(image).resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(resized_image)


def _prepare_similarity_frame(
    image: np.ndarray, target_width: int = SIMILARITY_FRAME_WIDTH
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


def take_screenshots() -> List[np.ndarray]:
    """Takes screenshots of all connected monitors or just the primary one.

    Depending on the `args.primary_monitor_only` flag, captures either
    all monitors or only the primary monitor (index 1 in mss.monitors).

    Returns:
        A list of screenshots, where each screenshot is a NumPy array (RGB).
    """
    screenshots: List[np.ndarray] = []
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
                screenshots.append(screenshot)
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
    # TODO: Move this environment variable setting to the application's entry point.
    # HACK: Prevents a warning/error from the huggingface/tokenizers library
    # when used in environments where multiprocessing fork safety is a concern.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    last_similarity_frames: List[np.ndarray] = [
        _prepare_similarity_frame(screenshot) for screenshot in take_screenshots()
    ]
    _capture_print(f"screenshots_taken monitors={len(last_similarity_frames)}")

    while True:
        if not is_user_active():
            time.sleep(CAPTURE_INTERVAL_SECONDS)
            continue

        current_screenshots: List[np.ndarray] = take_screenshots()
        _capture_print(f"screenshots_taken monitors={len(current_screenshots)}")

        # Ensure we have a last frame for each current screenshot.
        # This handles cases where monitor setup might change (though unlikely mid-run)
        if len(last_similarity_frames) != len(current_screenshots):
            # If monitor count changes, reset and continue.
            _capture_print(
                "monitor_layout_changed "
                f"previous={len(last_similarity_frames)} current={len(current_screenshots)}"
            )
            last_similarity_frames = [
                _prepare_similarity_frame(screenshot)
                for screenshot in current_screenshots
            ]
            time.sleep(CAPTURE_INTERVAL_SECONDS)
            continue


        for i, current_screenshot in enumerate(current_screenshots):
            t_frame_start = time.perf_counter()
            current_similarity_frame = _prepare_similarity_frame(current_screenshot)
            last_similarity_frame = last_similarity_frames[i]

            mssim_val = mean_structured_similarity_index(current_similarity_frame, last_similarity_frame)
            t_mssim_ms = (time.perf_counter() - t_frame_start) * 1000
            capture_state["last_mssim"] = round(mssim_val, 4)
            _capture_print(
                f"similarity_checked monitor={i + 1} mssim={mssim_val:.4f} "
                f"mssim_ms={t_mssim_ms:.1f}"
            )

            if mssim_val < 0.9:
                last_similarity_frames[i] = current_similarity_frame
                image = Image.fromarray(current_screenshot)
                timestamp = int(time.time())
                filename = f"{timestamp}.webp"
                filepath = os.path.join(screenshots_path, filename)
                _capture_print(
                    f"capture_changed monitor={i + 1} timestamp={timestamp} file={filepath}"
                )
                image.save(
                    filepath,
                    format="webp",
                    lossless=True,
                )

                _capture_print(f"ocr_start monitor={i + 1} timestamp={timestamp}")
                t_ocr_start = time.perf_counter()
                ocr_input = _resize_for_ocr(current_screenshot)
                text: str = extract_text_from_image(ocr_input)
                t_ocr_ms = (time.perf_counter() - t_ocr_start) * 1000
                _capture_print(
                    f"ocr_stop monitor={i + 1} timestamp={timestamp} "
                    f"ocr_ms={t_ocr_ms:.1f} text_len={len(text.strip())}"
                )

                t_embed_ms = 0.0
                t_db_ms = 0.0
                if text.strip():
                    _capture_print(f"embedding_start monitor={i + 1} timestamp={timestamp}")
                    t_embed_start = time.perf_counter()
                    embedding: np.ndarray = get_embedding(text)
                    t_embed_ms = (time.perf_counter() - t_embed_start) * 1000
                    _capture_print(
                        f"embedding_stop monitor={i + 1} timestamp={timestamp} "
                        f"embedding_ms={t_embed_ms:.1f}"
                    )

                    active_app_name: str = get_active_app_name() or "Unknown App"
                    active_window_title: str = get_active_window_title() or "Unknown Title"

                    _capture_print(f"db_write_start monitor={i + 1} timestamp={timestamp}")
                    t_db_start = time.perf_counter()
                    insert_entry(text, timestamp, embedding, active_app_name, active_window_title)
                    t_db_ms = (time.perf_counter() - t_db_start) * 1000
                    _capture_print(
                        f"db_write_stop monitor={i + 1} timestamp={timestamp} db_ms={t_db_ms:.1f}"
                    )
                else:
                    _capture_print(
                        f"embedding_skipped monitor={i + 1} timestamp={timestamp} reason=no_text"
                    )

                total_ms = (time.perf_counter() - t_frame_start) * 1000
                timing = {
                    "timestamp": timestamp,
                    "mssim_ms": round(t_mssim_ms, 1),
                    "ocr_ms": round(t_ocr_ms, 1),
                    "embedding_ms": round(t_embed_ms, 1),
                    "db_ms": round(t_db_ms, 1),
                    "total_ms": round(total_ms, 1),
                    "had_text": bool(text.strip()),
                }
                capture_state["recent_timings"].append(timing)
                capture_state["last_capture_ts"] = timestamp
                capture_state["captures_this_session"] += 1
                _capture_print(
                    "metrics "
                    f"timestamp={timestamp} monitor={i + 1} mssim={mssim_val:.4f} "
                    f"mssim_ms={timing['mssim_ms']:.1f} ocr_ms={timing['ocr_ms']:.1f} "
                    f"embedding_ms={timing['embedding_ms']:.1f} db_ms={timing['db_ms']:.1f} "
                    f"total_ms={timing['total_ms']:.1f} had_text={timing['had_text']} "
                    f"captures_this_session={capture_state['captures_this_session']}"
                )

        time.sleep(CAPTURE_INTERVAL_SECONDS)
