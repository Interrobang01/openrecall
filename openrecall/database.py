import sqlite3
from collections import namedtuple
import numpy as np
from typing import List, Optional, Tuple

from openrecall.config import db_path

# Define the structure of a database entry using namedtuple
Entry = namedtuple(
    "Entry",
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
TimelineEntry = namedtuple(
    "TimelineEntry",
    [
        "timestamp",
        "monitor_id",
        "segment_filename",
        "segment_pts_ms",
        "thumb_filename",
        "app",
        "title",
        "text",
        "embedding_magnitude",
        "embedding_is_zero",
    ],
)

EXPECTED_SCHEMA_COLUMNS = [
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
]


def _entries_table_exists(cursor: sqlite3.Cursor) -> bool:
    """Checks whether the entries table exists."""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='entries'"
    )
    return cursor.fetchone() is not None


def _get_entries_columns(cursor: sqlite3.Cursor) -> List[str]:
    """Returns the list of columns currently available in entries."""
    cursor.execute("PRAGMA table_info(entries)")
    return [row[1] for row in cursor.fetchall()]


def _create_entries_table(cursor: sqlite3.Cursor, table_name: str = "entries") -> None:
    """Creates entries table using the AV1-first schema."""
    cursor.execute(
        f"""CREATE TABLE IF NOT EXISTS {table_name} (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               app TEXT,
               title TEXT,
               text TEXT,
               timestamp INTEGER NOT NULL,
               monitor_id INTEGER NOT NULL,
               segment_filename TEXT NOT NULL,
               segment_pts_ms INTEGER NOT NULL,
               thumb_filename TEXT NOT NULL,
               embedding BLOB
           )"""
    )


def _ensure_entries_schema(cursor: sqlite3.Cursor) -> None:
    """Ensures a fresh AV1-first schema; drops incompatible legacy tables."""
    if _entries_table_exists(cursor):
        existing_columns = _get_entries_columns(cursor)
        if existing_columns != EXPECTED_SCHEMA_COLUMNS:
            cursor.execute("DROP TABLE entries")

    _create_entries_table(cursor)
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_entries_thumb_filename ON entries (thumb_filename)"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON entries (timestamp)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_timestamp_monitor ON entries (timestamp, monitor_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_segment_lookup ON entries (segment_filename, segment_pts_ms)"
    )


def _should_vacuum(cursor: sqlite3.Cursor, threshold_ratio: float = 0.2) -> bool:
    """Returns True when SQLite free pages are high enough to justify VACUUM."""
    cursor.execute("PRAGMA page_count")
    page_count = int(cursor.fetchone()[0])
    if page_count <= 0:
        return False

    cursor.execute("PRAGMA freelist_count")
    freelist_count = int(cursor.fetchone()[0])
    if freelist_count <= 0:
        return False

    return (freelist_count / float(page_count)) >= threshold_ratio


def create_db() -> None:
    """
    Creates the SQLite database and the 'entries' table if they don't exist.

    The table schema includes columns for an auto-incrementing ID, application name,
    window title, extracted text, timestamp, and text embedding.
    """
    should_vacuum = False
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            _ensure_entries_schema(cursor)
            conn.commit()
            should_vacuum = _should_vacuum(cursor)

        if should_vacuum:
            with sqlite3.connect(db_path) as vacuum_conn:
                vacuum_conn.execute("VACUUM")
    except sqlite3.Error as e:
        print(f"Database error during table creation: {e}")


def get_all_entries() -> List[Entry]:
    """
    Retrieves all entries from the database.

    Returns:
        List[Entry]: A list of all entries as Entry namedtuples.
                     Returns an empty list if the table is empty or an error occurs.
    """
    entries: List[Entry] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row  # Return rows as dictionary-like objects
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    id,
                    app,
                    title,
                    text,
                    timestamp,
                    monitor_id,
                    segment_filename,
                    segment_pts_ms,
                    thumb_filename,
                    embedding
                FROM entries
                ORDER BY timestamp DESC, monitor_id ASC
                """
            )
            results = cursor.fetchall()
            for row in results:
                # Deserialize the embedding blob back into a NumPy array
                embedding_blob = row["embedding"]
                embedding = (
                    np.frombuffer(embedding_blob, dtype=np.float32)
                    if embedding_blob
                    else np.array([], dtype=np.float32)
                )
                entries.append(
                    Entry(
                        id=row["id"],
                        app=row["app"],
                        title=row["title"],
                        text=row["text"],
                        timestamp=row["timestamp"],
                        monitor_id=row["monitor_id"],
                        segment_filename=row["segment_filename"],
                        segment_pts_ms=row["segment_pts_ms"],
                        thumb_filename=row["thumb_filename"],
                        embedding=embedding,
                    )
                )
    except sqlite3.Error as e:
        print(f"Database error while fetching all entries: {e}")
    return entries


def get_timestamps() -> List[int]:
    """
    Retrieves all timestamps from the database, ordered descending.

    Returns:
        List[int]: A list of all timestamps.
                   Returns an empty list if the table is empty or an error occurs.
    """
    timestamps: List[int] = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Use the index for potentially faster retrieval
            cursor.execute("SELECT timestamp FROM entries ORDER BY timestamp DESC")
            results = cursor.fetchall()
            timestamps = [result[0] for result in results]
    except sqlite3.Error as e:
        print(f"Database error while fetching timestamps: {e}")
    return timestamps


def get_timeline_entries() -> List[TimelineEntry]:
    """Retrieves timeline entries with monitor identity and thumbnail metadata."""
    timeline_entries: List[TimelineEntry] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT timestamp, monitor_id, segment_filename, segment_pts_ms, thumb_filename, app, title, text, embedding
                FROM entries
                ORDER BY timestamp DESC, monitor_id ASC
                """
            )
            results = cursor.fetchall()
            for row in results:
                embedding_blob = row["embedding"]
                embedding = (
                    np.frombuffer(embedding_blob, dtype=np.float32)
                    if embedding_blob
                    else np.array([], dtype=np.float32)
                )
                embedding_magnitude = float(np.linalg.norm(embedding)) if embedding.size else 0.0
                timeline_entries.append(
                    TimelineEntry(
                        timestamp=row["timestamp"],
                        monitor_id=row["monitor_id"],
                        segment_filename=row["segment_filename"],
                        segment_pts_ms=row["segment_pts_ms"],
                        thumb_filename=row["thumb_filename"],
                        app=row["app"],
                        title=row["title"],
                        text=row["text"],
                        embedding_magnitude=embedding_magnitude,
                        embedding_is_zero=embedding_magnitude <= 1e-8,
                    )
                )
    except sqlite3.Error as e:
        print(f"Database error while fetching timeline entries: {e}")
    return timeline_entries


def get_segment_frame_index(segment_filename: str, thumb_filename: str) -> Optional[int]:
    """Returns zero-based frame index within a segment for a thumbnail entry."""
    frame_index: Optional[int] = None
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id
                FROM entries
                WHERE segment_filename = ? AND thumb_filename = ?
                LIMIT 1
                """,
                (segment_filename, thumb_filename),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            entry_id = int(row[0])
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM entries
                WHERE segment_filename = ? AND id < ?
                """,
                (segment_filename, entry_id),
            )
            count_row = cursor.fetchone()
            frame_index = int(count_row[0]) if count_row is not None else None
    except sqlite3.Error as e:
        print(f"Database error while resolving segment frame index: {e}")

    return frame_index


def get_media_entries_for_segments(segment_filenames: List[str]) -> List[Tuple[str, str]]:
    """Returns (segment_filename, thumb_filename) pairs for selected segments."""
    if not segment_filenames:
        return []

    media_entries: List[Tuple[str, str]] = []
    unique_segments = sorted(set(segment_filenames))
    placeholders = ",".join("?" for _ in unique_segments)

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT segment_filename, thumb_filename
                FROM entries
                WHERE segment_filename IN ({placeholders})
                """,
                unique_segments,
            )
            media_entries = [(row[0], row[1]) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"Database error while fetching media entries for segments: {e}")

    return media_entries


def delete_entries_by_segment_filenames(segment_filenames: List[str]) -> int:
    """Deletes entries whose segment_filename is in segment_filenames."""
    if not segment_filenames:
        return 0

    unique_segments = sorted(set(segment_filenames))
    placeholders = ",".join("?" for _ in unique_segments)
    deleted_count = 0

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                DELETE FROM entries
                WHERE segment_filename IN ({placeholders})
                """,
                unique_segments,
            )
            conn.commit()
            deleted_count = cursor.rowcount
    except sqlite3.Error as e:
        print(f"Database error while deleting entries by segment filename: {e}")

    return deleted_count


def insert_entry(
    text: str,
    timestamp: int,
    embedding: np.ndarray,
    app: str,
    title: str,
    monitor_id: int = 1,
    segment_filename: Optional[str] = None,
    segment_pts_ms: int = 0,
    thumb_filename: Optional[str] = None,
) -> Optional[int]:
    """
    Inserts a new entry into the database.

    Args:
        text (str): The extracted text content.
        timestamp (int): The Unix timestamp of the screenshot.
        embedding (np.ndarray): The embedding vector for the text.
        app (str): The name of the active application.
        title (str): The title of the active window.
        monitor_id (int): The monitor id the screenshot was captured from.
        segment_filename (Optional[str]): The AV1 segment filename on disk.
        segment_pts_ms (int): Capture offset in milliseconds within the segment.
        thumb_filename (Optional[str]): The JPEG thumbnail filename on disk.

    Returns:
        Optional[int]: The ID of the newly inserted row, or None if insertion fails.
                       Prints an error message to stderr on failure.
    """
    if not segment_filename:
        raise ValueError("segment_filename is required for AV1 storage backend")
    if not thumb_filename:
        raise ValueError("thumb_filename is required for AV1 storage backend")

    embedding_bytes: bytes = embedding.astype(np.float32).tobytes()

    last_row_id: Optional[int] = None
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO entries (
                    text,
                    timestamp,
                    monitor_id,
                    segment_filename,
                    segment_pts_ms,
                    thumb_filename,
                    embedding,
                    app,
                    title
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thumb_filename) DO NOTHING
                """,
                (
                    text,
                    timestamp,
                    monitor_id,
                    segment_filename,
                    segment_pts_ms,
                    thumb_filename,
                    embedding_bytes,
                    app,
                    title,
                ),
            )
            conn.commit()
            if cursor.rowcount > 0:
                last_row_id = cursor.lastrowid

    except sqlite3.Error as e:
        print(f"Database error during insertion: {e}")
    return last_row_id
