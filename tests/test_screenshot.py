import io
import unittest

from openrecall.screenshot import MonitorAv1SegmentWriter


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


if __name__ == "__main__":
    unittest.main()
