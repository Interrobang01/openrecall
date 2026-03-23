import io
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from openrecall.screenshot import MonitorAv1SegmentWriter, MonitorPendingSegmentBuffer
import openrecall.screenshot as screenshot


class DummyStdin:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class DummyProcessRaisesValueError:
    def __init__(self, returncode_after_wait: int, stderr_bytes: bytes) -> None:
        self.stdin = DummyStdin()
        self.stderr = io.BytesIO(stderr_bytes)
        self.returncode = None
        self._returncode_after_wait = returncode_after_wait
        self.killed = False

    def communicate(self, timeout: int = 0):
        raise ValueError("flush of closed file")

    def wait(self, timeout: int = 0) -> int:
        self.returncode = self._returncode_after_wait
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class TestMonitorAv1SegmentWriterClose(unittest.TestCase):
    def test_close_handles_value_error_after_stdin_close(self):
        writer = MonitorAv1SegmentWriter(monitor_id=1)
        process = DummyProcessRaisesValueError(returncode_after_wait=0, stderr_bytes=b"")
        stdin_ref = process.stdin
        writer._process = process

        writer.close()

        self.assertTrue(stdin_ref.closed)
        self.assertIsNone(writer._process)

    def test_close_raises_runtime_error_for_nonzero_exit_after_fallback(self):
        writer = MonitorAv1SegmentWriter(monitor_id=2)
        process = DummyProcessRaisesValueError(
            returncode_after_wait=1,
            stderr_bytes=b"ffmpeg failed",
        )
        writer._process = process

        with self.assertRaisesRegex(RuntimeError, "ffmpeg AV1 writer exited"):
            writer.close()


class DummyWriterStdin:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.closed = False

    def write(self, data: bytes) -> int:
        self.buffer.extend(data)
        return len(data)

    def close(self) -> None:
        self.closed = True


class DummyWriterProcess:
    def __init__(self) -> None:
        self.stdin = DummyWriterStdin()
        self.stderr = io.BytesIO(b"")
        self.returncode = 0

    def communicate(self, timeout: int = 0):
        return b"", b""


class TestMonitorAv1SegmentWriterRotation(unittest.TestCase):
    def test_rotates_by_frame_count(self):
        writer = MonitorAv1SegmentWriter(monitor_id=1)
        frame = np.zeros((2, 2, 3), dtype=np.uint8)

        with mock.patch.object(screenshot, "AV1_SEGMENT_FRAMES", 2), mock.patch.object(
            screenshot, "OPENRECALL_AV1_PLAYBACK_FPS", 2.0
        ), mock.patch.object(
            screenshot.subprocess,
            "Popen",
            side_effect=[DummyWriterProcess(), DummyWriterProcess()],
        ):
            segment_1, _ = writer.write_frame(frame, timestamp_ms=1000)
            segment_2, _ = writer.write_frame(frame, timestamp_ms=2000)
            segment_3, _ = writer.write_frame(frame, timestamp_ms=3000)

        self.assertEqual(segment_1, segment_2)
        self.assertNotEqual(segment_2, segment_3)

    def test_segment_pts_ms_uses_playback_fps(self):
        writer = MonitorAv1SegmentWriter(monitor_id=2)
        frame = np.zeros((2, 2, 3), dtype=np.uint8)

        with mock.patch.object(screenshot, "AV1_SEGMENT_FRAMES", 10), mock.patch.object(
            screenshot, "OPENRECALL_AV1_PLAYBACK_FPS", 2.0
        ), mock.patch.object(
            screenshot.subprocess,
            "Popen",
            return_value=DummyWriterProcess(),
        ):
            _, pts_0 = writer.write_frame(frame, timestamp_ms=1000)
            _, pts_1 = writer.write_frame(frame, timestamp_ms=2000)
            _, pts_2 = writer.write_frame(frame, timestamp_ms=3000)

        self.assertEqual(pts_0, 0)
        self.assertEqual(pts_1, 500)
        self.assertEqual(pts_2, 1000)


class TestMonitorAv1SegmentWriterCommand(unittest.TestCase):
    def test_includes_threads_and_svtav1_params_when_configured(self):
        writer = MonitorAv1SegmentWriter(monitor_id=7)
        frame = np.zeros((2, 2, 3), dtype=np.uint8)

        with mock.patch.object(screenshot, "OPENRECALL_AV1_THREADS", 2), mock.patch.object(
            screenshot,
            "OPENRECALL_AV1_SVTAV1_PARAMS",
            "lp=2:scd=0",
        ), mock.patch.object(
            screenshot.subprocess,
            "Popen",
            return_value=DummyWriterProcess(),
        ) as popen_mock:
            writer.write_frame(frame, timestamp_ms=1000)

        ffmpeg_cmd = popen_mock.call_args[0][0]
        self.assertIn("-threads", ffmpeg_cmd)
        self.assertIn("2", ffmpeg_cmd)
        self.assertIn("-svtav1-params", ffmpeg_cmd)
        self.assertIn("lp=2:scd=0", ffmpeg_cmd)

    def test_omits_threads_and_svtav1_params_when_defaults_used(self):
        writer = MonitorAv1SegmentWriter(monitor_id=8)
        frame = np.zeros((2, 2, 3), dtype=np.uint8)

        with mock.patch.object(screenshot, "OPENRECALL_AV1_THREADS", 0), mock.patch.object(
            screenshot,
            "OPENRECALL_AV1_SVTAV1_PARAMS",
            "",
        ), mock.patch.object(
            screenshot.subprocess,
            "Popen",
            return_value=DummyWriterProcess(),
        ) as popen_mock:
            writer.write_frame(frame, timestamp_ms=1000)

        ffmpeg_cmd = popen_mock.call_args[0][0]
        self.assertNotIn("-threads", ffmpeg_cmd)
        self.assertNotIn("-svtav1-params", ffmpeg_cmd)


class TestMonitorPendingSegmentBuffer(unittest.TestCase):
    def test_add_frame_tracks_segment_and_flush_threshold(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            screenshot,
            "pending_frames_path",
            tmpdir,
        ), mock.patch.object(
            screenshot,
            "AV1_SEGMENT_FRAMES",
            2,
        ), mock.patch.object(
            screenshot,
            "OPENRECALL_AV1_PLAYBACK_FPS",
            2.0,
        ):
            buffer = MonitorPendingSegmentBuffer(monitor_id=3)

            segment_1, pts_1, should_flush_1 = buffer.add_frame(frame, 1000, "1000_m3.webp")
            segment_2, pts_2, should_flush_2 = buffer.add_frame(frame, 2000, "2000_m3.webp")

            self.assertEqual(segment_1, "1000_m3.mkv")
            self.assertEqual(segment_2, "1000_m3.mkv")
            self.assertEqual(pts_1, 0)
            self.assertEqual(pts_2, 500)
            self.assertFalse(should_flush_1)
            self.assertTrue(should_flush_2)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "1000_m3.webp")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "2000_m3.webp")))

    def test_flush_to_segment_encodes_and_cleans_pending_frames(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        created_writers = []

        class FakeBatchWriter:
            def __init__(self, monitor_id: int) -> None:
                self.monitor_id = monitor_id
                self.write_count = 0
                created_writers.append(self)

            def write_frame(self, frame_to_write: np.ndarray, timestamp_ms: int):
                self.write_count += 1
                return f"{timestamp_ms}_m{self.monitor_id}.mkv", 0

            def close(self) -> None:
                return

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            screenshot,
            "pending_frames_path",
            tmpdir,
        ), mock.patch.object(
            screenshot,
            "MonitorAv1SegmentWriter",
            FakeBatchWriter,
        ):
            buffer = MonitorPendingSegmentBuffer(monitor_id=4)
            buffer.add_frame(frame, 1111, "1111_m4.webp")
            buffer.add_frame(frame, 2222, "2222_m4.webp")

            segment_name = buffer.flush_to_segment()

            self.assertEqual(segment_name, "1111_m4.mkv")
            self.assertEqual(len(created_writers), 1)
            self.assertEqual(created_writers[0].write_count, 2)
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "1111_m4.webp")))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "2222_m4.webp")))


class TestCapturePauseAndBlacklist(unittest.TestCase):
    def setUp(self):
        screenshot.capture_state["paused_until_ts"] = 0
        screenshot.capture_state["paused_indefinitely"] = False

    def test_pause_forever_sets_state_and_is_paused(self):
        with mock.patch.object(screenshot, "send_system_notification", return_value=True):
            screenshot.set_capture_pause_forever()

        self.assertTrue(screenshot.capture_state["paused_indefinitely"])
        self.assertTrue(screenshot.is_capture_paused())

    def test_clear_capture_pause_resets_forever_pause(self):
        with mock.patch.object(screenshot, "send_system_notification", return_value=True):
            screenshot.set_capture_pause_forever()
            screenshot.clear_capture_pause()

        self.assertFalse(screenshot.capture_state["paused_indefinitely"])
        self.assertEqual(screenshot.capture_state["paused_until_ts"], 0)
        self.assertFalse(screenshot.is_capture_paused())

    def test_blacklist_match_is_case_insensitive(self):
        terms = ["bitwarden", "incognito"]
        matches = screenshot._find_blacklist_matches("BitWarden vault and Incognito tab", terms)
        self.assertEqual(matches, ["bitwarden", "incognito"])

    def test_blacklist_short_term_does_not_match_inside_word(self):
        terms = ["tor"]
        matches = screenshot._find_blacklist_matches("Mozilla Navigator window", terms)
        self.assertEqual(matches, [])

    def test_blacklist_phrase_matches_window_name(self):
        terms = ["tor browser"]
        matches = screenshot._find_blacklist_matches("Tor Browser - Private browsing", terms)
        self.assertEqual(matches, ["tor browser"])


if __name__ == "__main__":
    unittest.main()
