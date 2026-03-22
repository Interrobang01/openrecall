import logging
from typing import Optional

from openrecall.config import (
    OPENRECALL_HOTKEY_PAUSE_5M,
    OPENRECALL_HOTKEY_PAUSE_30M,
    OPENRECALL_HOTKEY_PAUSE_FOREVER,
    OPENRECALL_HOTKEY_RESUME,
)
from openrecall.screenshot import (
    clear_capture_pause,
    set_capture_pause_for_seconds,
    set_capture_pause_forever,
)

logger = logging.getLogger(__name__)


def start_hotkey_listener() -> Optional[object]:
    """Starts global pause/resume hotkeys when pynput is available."""
    try:
        from pynput import keyboard  # type: ignore
    except Exception as exc:
        logger.warning("Global hotkeys unavailable (pynput import failed): %s", exc)
        return None

    hotkey_map = {
        OPENRECALL_HOTKEY_PAUSE_5M: lambda: set_capture_pause_for_seconds(5 * 60),
        OPENRECALL_HOTKEY_PAUSE_30M: lambda: set_capture_pause_for_seconds(30 * 60),
        OPENRECALL_HOTKEY_PAUSE_FOREVER: set_capture_pause_forever,
        OPENRECALL_HOTKEY_RESUME: clear_capture_pause,
    }

    try:
        listener = keyboard.GlobalHotKeys(hotkey_map)
        listener.start()
        logger.info(
            "Global hotkeys enabled: 5m=%s 30m=%s forever=%s resume=%s",
            OPENRECALL_HOTKEY_PAUSE_5M,
            OPENRECALL_HOTKEY_PAUSE_30M,
            OPENRECALL_HOTKEY_PAUSE_FOREVER,
            OPENRECALL_HOTKEY_RESUME,
        )
        return listener
    except Exception as exc:
        logger.warning("Global hotkey listener failed to start: %s", exc)
        return None
