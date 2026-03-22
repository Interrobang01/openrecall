import unittest
import sqlite3
import os
import tempfile
import time
import numpy as np
from unittest.mock import patch

# Temporarily adjust path to import from openrecall
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from openrecall.database, mocking db_path *before* the import
# Create a temporary file path that will be used by the mock
temp_db_file = tempfile.NamedTemporaryFile(delete=False)
mock_db_path = temp_db_file.name
temp_db_file.close() # Close the file handle, but the file persists because delete=False

with patch('openrecall.config.db_path', mock_db_path):
    from openrecall.database import (
        create_db,
        insert_entry,
        get_all_entries,
        get_timeline_entries,
        get_timestamps,
        Entry,
    )
    # Also patch db_path within the database module itself if it was imported directly there
    import openrecall.database
    openrecall.database.db_path = mock_db_path


class TestDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a temporary database file for all tests in this class."""
        # The database path is already patched by the module-level patch
        cls.db_path = mock_db_path
        # Ensure the database and table are created once
        create_db()

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary database file after all tests."""
        # Try closing connection if any test left it open (though setUp/tearDown should handle this)
        try:
            if hasattr(cls, 'conn') and cls.conn:
                cls.conn.close()
        except Exception:
            pass # Ignore errors during cleanup
        os.remove(cls.db_path)
        # Clean up sys.path modification
        sys.path.pop(0)


    def setUp(self):
        """Connect to the database and clear entries before each test."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM entries")
        self.conn.commit()
        # No need to close here, will be handled by tearDown or next setUp potentially

    def tearDown(self):
        """Close the database connection after each test."""
        if self.conn:
            self.conn.close()

    def test_create_db(self):
        """Test if create_db creates the table and expected indexes."""
        # Check if table exists
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='entries'")
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 'entries')

        # Check timestamp index exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_timestamp'")
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 'idx_timestamp')

        # Check image filename unique index exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_entries_image_filename'"
        )
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 'idx_entries_image_filename')

    def test_02_insert_entry(self):
        """Test inserting a single entry."""
        ts = int(time.time())
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        inserted_id = insert_entry("Test text", ts, embedding, "TestApp", "TestTitle")

        self.assertIsNotNone(inserted_id)
        self.assertIsInstance(inserted_id, int)

        # Verify the entry exists in the DB
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM entries WHERE id = ?", (inserted_id,))
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        # (id, app, title, text, timestamp, monitor_id, image_filename, embedding_blob)
        self.assertEqual(result[1], "TestApp")
        self.assertEqual(result[2], "TestTitle")
        self.assertEqual(result[3], "Test text")
        self.assertEqual(result[4], ts)
        self.assertEqual(result[5], 1)
        self.assertEqual(result[6], f"{ts}_m1.webp")
        retrieved_embedding = np.frombuffer(result[7], dtype=np.float32)
        np.testing.assert_array_almost_equal(retrieved_embedding, embedding)

    def test_insert_same_timestamp_multiple_monitors(self):
        """Test inserting same timestamp for different monitors."""
        ts = int(time.time())
        embedding1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        embedding2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)

        id1 = insert_entry(
            "First text",
            ts,
            embedding1,
            "App1",
            "Title1",
            monitor_id=1,
            image_filename=f"{ts}_m1.webp",
        )
        self.assertIsNotNone(id1)

        id2 = insert_entry(
            "Second text",
            ts,
            embedding2,
            "App2",
            "Title2",
            monitor_id=2,
            image_filename=f"{ts}_m2.webp",
        )
        self.assertIsNotNone(id2)

        # Verify both entries exist for same timestamp
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM entries WHERE timestamp = ?", (ts,))
        count = cursor.fetchone()[0]
        self.assertEqual(count, 2)

    def test_insert_duplicate_image_filename(self):
        """Test inserting an entry with duplicate image filename (should be ignored)."""
        ts = int(time.time())
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        image_filename = f"{ts}_m1.webp"

        id1 = insert_entry("First text", ts, embedding, "App1", "Title1", 1, image_filename)
        self.assertIsNotNone(id1)

        id2 = insert_entry("Second text", ts, embedding, "App2", "Title2", 1, image_filename)
        self.assertIsNone(id2, "Inserting duplicate image filename should return None")

        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM entries WHERE image_filename = ?", (image_filename,))
        count = cursor.fetchone()[0]
        self.assertEqual(count, 1)

    def test_get_all_entries_empty(self):
        """Test getting entries from an empty database."""
        entries = get_all_entries()
        self.assertEqual(entries, [])

    def test_get_all_entries_multiple(self):
        """Test retrieving multiple entries."""
        ts1 = int(time.time())
        ts2 = ts1 + 10
        ts3 = ts1 - 10 # Ensure ordering works
        emb1 = np.array([0.1] * 5, dtype=np.float32)
        emb2 = np.array([0.2] * 5, dtype=np.float32)
        emb3 = np.array([0.3] * 5, dtype=np.float32)

        insert_entry("Text 1", ts1, emb1, "App1", "Title1")
        insert_entry("Text 2", ts2, emb2, "App2", "Title2")
        insert_entry("Text 3", ts3, emb3, "App3", "Title3")

        entries = get_all_entries()
        self.assertEqual(len(entries), 3)

        # Entries should be ordered by timestamp DESC
        self.assertEqual(entries[0].timestamp, ts2)
        self.assertEqual(entries[0].text, "Text 2")
        self.assertEqual(entries[0].app, "App2")
        self.assertEqual(entries[0].title, "Title2")
        self.assertEqual(entries[0].monitor_id, 1)
        self.assertEqual(entries[0].image_filename, f"{ts2}_m1.webp")
        np.testing.assert_array_almost_equal(entries[0].embedding, emb2)
        self.assertIsInstance(entries[0].id, int)

        self.assertEqual(entries[1].timestamp, ts1)
        self.assertEqual(entries[1].text, "Text 1")
        np.testing.assert_array_almost_equal(entries[1].embedding, emb1)

        self.assertEqual(entries[2].timestamp, ts3)
        self.assertEqual(entries[2].text, "Text 3")
        np.testing.assert_array_almost_equal(entries[2].embedding, emb3)

    def test_get_timestamps_empty(self):
        """Test getting timestamps from an empty database."""
        timestamps = get_timestamps()
        self.assertEqual(timestamps, [])

    def test_get_timestamps_multiple(self):
        """Test retrieving multiple timestamps."""
        ts1 = int(time.time())
        ts2 = ts1 + 10
        ts3 = ts1 - 10
        emb = np.array([0.1] * 5, dtype=np.float32) # Embedding content doesn't matter here

        insert_entry("T1", ts1, emb, "A1", "T1")
        insert_entry("T2", ts2, emb, "A2", "T2")
        insert_entry("T3", ts3, emb, "A3", "T3")

        timestamps = get_timestamps()
        self.assertEqual(len(timestamps), 3)
        # Timestamps should be ordered DESC
        self.assertEqual(timestamps, [ts2, ts1, ts3])

    def test_get_timeline_entries(self):
        """Test timeline entries include monitor id and image filename."""
        ts = int(time.time())
        emb = np.array([0.5] * 5, dtype=np.float32)

        insert_entry("M1", ts, emb, "App", "Title", monitor_id=1, image_filename=f"{ts}_m1.webp")
        insert_entry("M2", ts, emb, "App", "Title", monitor_id=2, image_filename=f"{ts}_m2.webp")

        timeline_entries = get_timeline_entries()
        self.assertEqual(len(timeline_entries), 2)
        self.assertEqual(timeline_entries[0].timestamp, ts)
        self.assertEqual(timeline_entries[0].monitor_id, 1)
        self.assertEqual(timeline_entries[0].image_filename, f"{ts}_m1.webp")
        self.assertEqual(timeline_entries[1].monitor_id, 2)

    def test_migrate_legacy_timestamp_unique_schema(self):
        """Test migration from legacy timestamp-unique schema preserves data."""
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS entries")
        cursor.execute(
            """
            CREATE TABLE entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                app TEXT,
                title TEXT,
                text TEXT,
                timestamp INTEGER UNIQUE,
                embedding BLOB
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON entries (timestamp)")

        ts = int(time.time())
        emb = np.array([0.7, 0.8, 0.9], dtype=np.float32)
        cursor.execute(
            "INSERT INTO entries (app, title, text, timestamp, embedding) VALUES (?, ?, ?, ?, ?)",
            ("LegacyApp", "LegacyTitle", "Legacy text", ts, emb.tobytes()),
        )
        self.conn.commit()

        # Trigger schema migration.
        create_db()

        cursor.execute("PRAGMA table_info(entries)")
        columns = [row[1] for row in cursor.fetchall()]
        self.assertIn("monitor_id", columns)
        self.assertIn("image_filename", columns)

        # Legacy row should be preserved with default monitor/file metadata.
        cursor.execute(
            "SELECT monitor_id, image_filename, text FROM entries WHERE timestamp = ?",
            (ts,),
        )
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 1)
        self.assertEqual(row[1], f"{ts}.webp")
        self.assertEqual(row[2], "Legacy text")

        # After migration, same timestamp for a second monitor should be allowed.
        inserted_id = insert_entry(
            "Second monitor",
            ts,
            emb,
            "App2",
            "Title2",
            monitor_id=2,
            image_filename=f"{ts}_m2.webp",
        )
        self.assertIsNotNone(inserted_id)


if __name__ == '__main__':
    unittest.main()
