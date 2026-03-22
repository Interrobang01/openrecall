import os
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

# Temporarily adjust path to import from openrecall
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import database with a temporary db path patched in config first.
temp_db_file = tempfile.NamedTemporaryFile(delete=False)
mock_db_path = temp_db_file.name
temp_db_file.close()

with patch("openrecall.config.db_path", mock_db_path):
    from openrecall.database import (
        create_db,
        delete_entries_by_segment_filenames,
        get_all_entries,
        get_media_entries_for_segments,
        get_segment_frame_index,
        get_timeline_entries,
        get_timestamps,
        insert_entry,
    )
    import openrecall.database

    openrecall.database.db_path = mock_db_path

from openrecall.config import check_ffmpeg_av1_capabilities, ffmpeg_smoke_check


class TestDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = mock_db_path
        create_db()

    @classmethod
    def tearDownClass(cls):
        try:
            if hasattr(cls, "conn") and cls.conn:
                cls.conn.close()
        except Exception:
            pass
        os.remove(cls.db_path)
        sys.path.pop(0)

    def setUp(self):
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM entries")
        self.conn.commit()

    def tearDown(self):
        if self.conn:
            self.conn.close()

    def test_create_db_schema_and_indexes(self):
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(entries)")
        columns = [row[1] for row in cursor.fetchall()]
        self.assertEqual(
            columns,
            [
                "id",
                "app",
                "title",
                "text",
                "timestamp",
                "monitor_id",
                "segment_filename",
                "segment_pts_ms",
                "thumb_filename",
                "embedding",
            ],
        )

        cursor.execute("PRAGMA index_list(entries)")
        indexes = {row[1] for row in cursor.fetchall()}
        self.assertIn("idx_entries_thumb_filename", indexes)
        self.assertIn("idx_timestamp", indexes)
        self.assertIn("idx_timestamp_monitor", indexes)
        self.assertIn("idx_segment_lookup", indexes)

    def test_insert_and_get_all_entries(self):
        timestamp = int(time.time())
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        inserted_id = insert_entry(
            "Test text",
            timestamp,
            embedding,
            "TestApp",
            "TestTitle",
            monitor_id=1,
            segment_filename="1700000000000_m1.mkv",
            segment_pts_ms=1200,
            thumb_filename="1700000000000_m1.webp",
        )
        self.assertIsNotNone(inserted_id)

        entries = get_all_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].id, inserted_id)
        self.assertEqual(entries[0].app, "TestApp")
        self.assertEqual(entries[0].title, "TestTitle")
        self.assertEqual(entries[0].text, "Test text")
        self.assertEqual(entries[0].timestamp, timestamp)
        self.assertEqual(entries[0].monitor_id, 1)
        self.assertEqual(entries[0].segment_filename, "1700000000000_m1.mkv")
        self.assertEqual(entries[0].segment_pts_ms, 1200)
        self.assertEqual(entries[0].thumb_filename, "1700000000000_m1.webp")
        np.testing.assert_array_almost_equal(entries[0].embedding, embedding)

    def test_insert_duplicate_thumb_filename(self):
        timestamp = int(time.time())
        embedding = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        first_id = insert_entry(
            "First",
            timestamp,
            embedding,
            "App1",
            "Title1",
            monitor_id=1,
            segment_filename="segment_a.mkv",
            segment_pts_ms=0,
            thumb_filename="dup.webp",
        )
        self.assertIsNotNone(first_id)

        second_id = insert_entry(
            "Second",
            timestamp,
            embedding,
            "App2",
            "Title2",
            monitor_id=2,
            segment_filename="segment_b.mkv",
            segment_pts_ms=333,
            thumb_filename="dup.webp",
        )
        self.assertIsNone(second_id)

        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM entries WHERE thumb_filename = 'dup.webp'")
        self.assertEqual(cursor.fetchone()[0], 1)

    def test_get_timeline_entries(self):
        timestamp = int(time.time())
        embedding = np.array([0.5] * 5, dtype=np.float32)

        insert_entry(
            "M1",
            timestamp,
            embedding,
            "App",
            "Title",
            monitor_id=1,
            segment_filename="shared_segment.mkv",
            segment_pts_ms=111,
            thumb_filename="frame_1.webp",
        )
        insert_entry(
            "M2",
            timestamp,
            embedding,
            "App",
            "Title",
            monitor_id=2,
            segment_filename="shared_segment.mkv",
            segment_pts_ms=222,
            thumb_filename="frame_2.webp",
        )

        timeline_entries = get_timeline_entries()
        self.assertEqual(len(timeline_entries), 2)
        self.assertEqual(timeline_entries[0].monitor_id, 1)
        self.assertEqual(timeline_entries[0].segment_filename, "shared_segment.mkv")
        self.assertEqual(timeline_entries[0].segment_pts_ms, 111)
        self.assertEqual(timeline_entries[0].thumb_filename, "frame_1.webp")
        self.assertEqual(timeline_entries[1].monitor_id, 2)
        self.assertEqual(timeline_entries[1].segment_pts_ms, 222)

    def test_get_timestamps(self):
        embedding = np.array([0.2] * 5, dtype=np.float32)
        ts1 = int(time.time())
        ts2 = ts1 + 10
        ts3 = ts1 - 10

        insert_entry(
            "T1",
            ts1,
            embedding,
            "App1",
            "Title1",
            segment_filename="s1.mkv",
            segment_pts_ms=1,
            thumb_filename="t1.webp",
        )
        insert_entry(
            "T2",
            ts2,
            embedding,
            "App2",
            "Title2",
            segment_filename="s2.mkv",
            segment_pts_ms=2,
            thumb_filename="t2.webp",
        )
        insert_entry(
            "T3",
            ts3,
            embedding,
            "App3",
            "Title3",
            segment_filename="s3.mkv",
            segment_pts_ms=3,
            thumb_filename="t3.webp",
        )

        self.assertEqual(get_timestamps(), [ts2, ts1, ts3])

    def test_get_segment_frame_index(self):
        embedding = np.array([0.3, 0.1], dtype=np.float32)
        ts = int(time.time())

        insert_entry(
            "A",
            ts,
            embedding,
            "App",
            "Title",
            segment_filename="seg_lookup.mkv",
            segment_pts_ms=700,
            thumb_filename="a.webp",
        )
        insert_entry(
            "B",
            ts + 1,
            embedding,
            "App",
            "Title",
            segment_filename="seg_lookup.mkv",
            segment_pts_ms=1700,
            thumb_filename="b.webp",
        )
        insert_entry(
            "C",
            ts + 2,
            embedding,
            "App",
            "Title",
            segment_filename="seg_lookup.mkv",
            segment_pts_ms=2700,
            thumb_filename="c.webp",
        )

        self.assertEqual(get_segment_frame_index("seg_lookup.mkv", "a.webp"), 0)
        self.assertEqual(get_segment_frame_index("seg_lookup.mkv", "b.webp"), 1)
        self.assertEqual(get_segment_frame_index("seg_lookup.mkv", "c.webp"), 2)
        self.assertIsNone(get_segment_frame_index("seg_lookup.mkv", "missing.webp"))

    def test_insert_entry_requires_segment_and_thumbnail(self):
        embedding = np.array([0.1, 0.2], dtype=np.float32)
        timestamp = int(time.time())

        with self.assertRaisesRegex(ValueError, "segment_filename"):
            insert_entry("x", timestamp, embedding, "a", "b", thumb_filename="x.webp")

        with self.assertRaisesRegex(ValueError, "thumb_filename"):
            insert_entry(
                "x",
                timestamp,
                embedding,
                "a",
                "b",
                segment_filename="x.mkv",
            )

    def test_get_media_entries_for_segments(self):
        embedding = np.array([0.1, 0.2], dtype=np.float32)
        ts = int(time.time())

        insert_entry(
            "A",
            ts,
            embedding,
            "App",
            "Title",
            segment_filename="seg_a.mkv",
            segment_pts_ms=0,
            thumb_filename="a.webp",
        )
        insert_entry(
            "B",
            ts + 1,
            embedding,
            "App",
            "Title",
            segment_filename="seg_b.mkv",
            segment_pts_ms=500,
            thumb_filename="b.webp",
        )

        media_entries = get_media_entries_for_segments(["seg_b.mkv"])
        self.assertEqual(media_entries, [("seg_b.mkv", "b.webp")])

    def test_delete_entries_by_segment_filenames(self):
        embedding = np.array([0.1, 0.2], dtype=np.float32)
        ts = int(time.time())

        insert_entry(
            "A",
            ts,
            embedding,
            "App",
            "Title",
            segment_filename="seg_delete.mkv",
            segment_pts_ms=0,
            thumb_filename="delete_a.webp",
        )
        insert_entry(
            "B",
            ts + 1,
            embedding,
            "App",
            "Title",
            segment_filename="seg_keep.mkv",
            segment_pts_ms=500,
            thumb_filename="keep_b.webp",
        )

        deleted = delete_entries_by_segment_filenames(["seg_delete.mkv"])
        self.assertEqual(deleted, 1)

        entries = get_all_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].segment_filename, "seg_keep.mkv")


class TestFfmpegHooks(unittest.TestCase):
    def test_ffmpeg_smoke_check_success(self):
        with patch("openrecall.config.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0
            run_mock.return_value.stdout = "ffmpeg version test"
            run_mock.return_value.stderr = ""

            ffmpeg_smoke_check("ffmpeg-custom")

            run_mock.assert_called_once()
            called_command = run_mock.call_args[0][0]
            self.assertEqual(called_command[0], "ffmpeg-custom")
            self.assertIn("-version", called_command)

    def test_ffmpeg_smoke_check_missing_binary(self):
        with patch("openrecall.config.subprocess.run", side_effect=FileNotFoundError()):
            with self.assertRaisesRegex(RuntimeError, "OPENRECALL_FFMPEG_BIN"):
                ffmpeg_smoke_check("missing-ffmpeg")

    def test_check_ffmpeg_av1_capabilities_missing_encoder(self):
        with patch("openrecall.config.ffmpeg_smoke_check"):
            with patch(
                "openrecall.config._run_ffmpeg_command",
                side_effect=[
                    "encoders without required one",
                    " V..... libdav1d AV1 decoder",
                ],
            ):
                with self.assertRaisesRegex(RuntimeError, "libsvtav1"):
                    check_ffmpeg_av1_capabilities("ffmpeg")


if __name__ == "__main__":
    unittest.main()
