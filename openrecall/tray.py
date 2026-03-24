import os
import sys
import time
import webbrowser
from threading import Thread
from typing import Callable, Optional, Tuple

from openrecall.config import pending_frames_path
from openrecall.screenshot import (
    capture_state,
    clear_capture_pause,
    is_capture_paused,
    request_capture_stop,
    set_capture_pause_for_seconds,
)

_PENDING_EXTENSIONS = {".webp", ".png"}
_CAPTURE_HIGHLIGHT_SECONDS = 6


def _count_pending_frames() -> int:
    """Returns count of pending full-resolution frames awaiting segment encode."""
    try:
        with os.scandir(pending_frames_path) as entries:
            return sum(
                1
                for entry in entries
                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in _PENDING_EXTENSIONS
            )
    except OSError:
        return 0


def _format_age(seconds: int) -> str:
    """Formats age in seconds as compact human-readable text."""
    safe_seconds = max(0, int(seconds))
    if safe_seconds < 60:
        return f"{safe_seconds}s"
    if safe_seconds < 3600:
        return f"{safe_seconds // 60}m"
    return f"{safe_seconds // 3600}h"


def _derive_tray_state(now_ts: int) -> Tuple[str, str, str]:
    """Derives (title, subtitle, icon_name) for tray display."""
    status = str(capture_state.get("status") or "unknown")
    status_updated_ts = int(capture_state.get("status_updated_ts") or 0)
    paused = is_capture_paused(now_ts)
    pending_count = _count_pending_frames()
    last_capture_ts = int(capture_state.get("last_capture_ts") or 0)
    blocked_terms = list(capture_state.get("last_blocked_terms") or [])

    if bool(capture_state.get("stop_requested")):
        return "Stopping", "Capture stop requested", "process-stop"

    if paused:
        if bool(capture_state.get("paused_indefinitely")):
            return "Paused", "Indefinitely", "media-playback-pause"
        paused_until = int(capture_state.get("paused_until_ts") or 0)
        remaining = max(0, paused_until - now_ts)
        return "Paused", f"{_format_age(remaining)} left", "media-playback-pause"

    if status.startswith("blocked"):
        terms_hint = ", ".join(blocked_terms[:2]) if blocked_terms else "blacklist"
        return "Blocked", terms_hint, "system-lock-screen"

    is_actively_encoding = (
        status == "encoding_pending"
        and status_updated_ts > 0
        and (now_ts - status_updated_ts) <= 8
    )
    if is_actively_encoding:
        return "Encoding", f"{pending_count} pending", "emblem-synchronizing"

    if last_capture_ts > 0:
        capture_age = max(0, now_ts - last_capture_ts)
        if capture_age <= _CAPTURE_HIGHLIGHT_SECONDS:
            return "Captured", "just now", "camera-photo"
        if status == "user_inactive":
            return "Idle", "waiting for activity", "changes-prevent"
        return "Running", f"last cap {_format_age(capture_age)} ago", "media-record"

    if status == "user_inactive":
        return "Idle", "waiting for activity", "changes-prevent"

    if status == "capturing":
        return "Capturing", "processing frame", "camera-photo"

    return "Running", "watching for changes", "media-record"


def _create_menu_item(Gtk, label: str, callback: Callable[[], None], enabled: bool = True):
    """Creates a Gtk menu item and wires activate callback."""
    item = Gtk.MenuItem(label=label)
    item.set_sensitive(enabled)
    if enabled:
        item.connect("activate", lambda _widget: callback())
    return item


def _create_indicator_menu(Gtk):
    """Builds tray menu with core actions and status row."""
    menu = Gtk.Menu()
    status_item = Gtk.MenuItem(label="OpenRecall: starting")
    status_item.set_sensitive(False)
    menu.append(status_item)

    menu.append(Gtk.SeparatorMenuItem())
    menu.append(_create_menu_item(Gtk, "Open Web UI", lambda: webbrowser.open("http://127.0.0.1:8082")))
    menu.append(_create_menu_item(Gtk, "Pause 5m", lambda: set_capture_pause_for_seconds(5 * 60)))
    menu.append(_create_menu_item(Gtk, "Pause 30m", lambda: set_capture_pause_for_seconds(30 * 60)))
    menu.append(_create_menu_item(Gtk, "Resume", clear_capture_pause))

    menu.append(Gtk.SeparatorMenuItem())

    def _quit_app() -> None:
        request_capture_stop()
        Gtk.main_quit()

    menu.append(_create_menu_item(Gtk, "Quit OpenRecall", _quit_app))
    menu.show_all()
    return menu, status_item


def start_linux_tray() -> Optional[Thread]:
    """Starts a minimal Linux tray icon in a daemon thread.

    Returns:
        Optional[Thread]: Background tray thread when started, otherwise None.
    """
    if not sys.platform.startswith("linux"):
        return None

    if not os.getenv("DISPLAY") and not os.getenv("WAYLAND_DISPLAY"):
        print("Tray skipped: no active graphical display detected.")
        return None

    try:
        import gi
        gi.require_version("Gtk", "3.0")
        from gi.repository import Gtk
    except Exception as exc:
        print(f"Tray unavailable: GTK bindings not available ({exc}).")
        return None

    def _run_tray_loop() -> None:
        try:
            indicator_created = False
            indicator = None
            menu = None
            status_item = None
            status_icon = None

            from gi.repository import GLib

            for module_name in ("AyatanaAppIndicator3", "AppIndicator3"):
                try:
                    import gi
                    gi.require_version(module_name, "0.1")
                    module = __import__(f"gi.repository.{module_name}", fromlist=[module_name])
                    IndicatorCategory = module.IndicatorCategory
                    IndicatorStatus = module.IndicatorStatus

                    indicator = module.Indicator.new(
                        "openrecall",
                        "media-record",
                        IndicatorCategory.APPLICATION_STATUS,
                    )
                    indicator.set_status(IndicatorStatus.ACTIVE)
                    indicator.set_title("OpenRecall")
                    menu, status_item = _create_indicator_menu(Gtk)
                    indicator.set_menu(menu)
                    indicator_created = True
                    print(f"Tray started via {module_name}.")
                    break
                except Exception:
                    continue

            if not indicator_created:
                status_icon = Gtk.StatusIcon.new_from_icon_name("media-record")
                status_icon.set_title("OpenRecall")
                status_icon.set_tooltip_text("OpenRecall is running")
                status_icon.set_visible(True)
                menu, status_item = _create_indicator_menu(Gtk)

                def _on_popup(_icon, button, activate_time):
                    menu.popup(None, None, Gtk.StatusIcon.position_menu, _icon, button, activate_time)

                status_icon.connect("popup-menu", _on_popup)
                indicator = status_icon
                print("Tray started via Gtk.StatusIcon fallback.")

            def _refresh_indicator() -> bool:
                now_ts = int(time.time())
                title, subtitle, icon_name = _derive_tray_state(now_ts)
                tooltip = f"OpenRecall: {title} — {subtitle}"
                status_prefix = "● " if title == "Captured" else ""

                if status_item is not None:
                    status_item.set_label(f"Status: {status_prefix}{title} ({subtitle})")

                if status_icon is not None:
                    status_icon.set_from_icon_name(icon_name)
                    status_icon.set_tooltip_text(tooltip)

                if indicator is not None and hasattr(indicator, "set_icon_full"):
                    indicator.set_icon_full(icon_name, tooltip)
                if indicator is not None and hasattr(indicator, "set_label"):
                    indicator.set_label(f"{status_prefix}{title} · {subtitle}", "OpenRecall")

                return True

            _refresh_indicator()
            GLib.timeout_add_seconds(1, _refresh_indicator)

            Gtk.main()

            # Keep references alive for the loop lifetime.
            _ = (indicator, menu, status_item, status_icon)
        except Exception as exc:
            print(f"Tray failed to start: {exc}")

    tray_thread = Thread(target=_run_tray_loop, daemon=True, name="openrecall-tray")
    tray_thread.start()
    return tray_thread
