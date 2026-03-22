import sqlite3
from collections import namedtuple
import numpy as np
from typing import Any, List, Optional, Tuple

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
        "image_filename",
        "embedding",
    ],
)
TimelineEntry = namedtuple("TimelineEntry", ["timestamp", "monitor_id", "image_filename"])


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


def _has_unique_timestamp_constraint(cursor: sqlite3.Cursor) -> bool:
    """Checks if entries has a unique index/constraint on timestamp alone."""
    cursor.execute("PRAGMA index_list(entries)")
    index_rows = cursor.fetchall()
    for index_row in index_rows:
        index_name = index_row[1]
        is_unique = bool(index_row[2])
        if not is_unique:
            continue

        safe_index_name = index_name.replace('"', '""')
        cursor.execute(f'PRAGMA index_info("{safe_index_name}")')
        indexed_columns = [idx_row[2] for idx_row in cursor.fetchall()]
        if indexed_columns == ["timestamp"]:
            return True
    return False


def _create_entries_table(cursor: sqlite3.Cursor, table_name: str = "entries") -> None:
    """Creates entries table using the latest schema."""
    cursor.execute(
        f"""CREATE TABLE IF NOT EXISTS {table_name} (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               app TEXT,
               title TEXT,
               text TEXT,
               timestamp INTEGER,
               monitor_id INTEGER NOT NULL DEFAULT 1,
               image_filename TEXT NOT NULL UNIQUE,
               embedding BLOB
           )"""
    )


def _rebuild_entries_table(
    cursor: sqlite3.Cursor,
    has_monitor_id: bool,
    has_image_filename: bool,
) -> None:
    """Rebuilds entries table to drop legacy constraints and apply latest schema."""
    _create_entries_table(cursor, table_name="entries_new")

    monitor_expr = "monitor_id" if has_monitor_id else "1"
    image_expr = "image_filename" if has_image_filename else "CAST(timestamp AS TEXT) || '.webp'"
    cursor.execute(
        f"""
        INSERT OR IGNORE INTO entries_new
            (id, app, title, text, timestamp, monitor_id, image_filename, embedding)
        SELECT
            id,
            app,
            title,
            text,
            timestamp,
            COALESCE({monitor_expr}, 1),
            COALESCE(NULLIF({image_expr}, ''), CAST(timestamp AS TEXT) || '.webp'),
            embedding
        FROM entries
        ORDER BY id
        """
    )

    cursor.execute("DROP TABLE entries")
    cursor.execute("ALTER TABLE entries_new RENAME TO entries")


def _ensure_entries_schema(cursor: sqlite3.Cursor) -> None:
    """Ensures entries table exists with monitor-aware schema and indexes."""
    if not _entries_table_exists(cursor):
        _create_entries_table(cursor)
    else:
        columns = _get_entries_columns(cursor)
        has_monitor_id = "monitor_id" in columns
        has_image_filename = "image_filename" in columns

        if _has_unique_timestamp_constraint(cursor):
            _rebuild_entries_table(
                cursor,
                has_monitor_id=has_monitor_id,
                has_image_filename=has_image_filename,
            )
        else:
            if not has_monitor_id:
                cursor.execute(
                    "ALTER TABLE entries ADD COLUMN monitor_id INTEGER NOT NULL DEFAULT 1"
                )
            if not has_image_filename:
                cursor.execute("ALTER TABLE entries ADD COLUMN image_filename TEXT")

            cursor.execute("UPDATE entries SET monitor_id = COALESCE(monitor_id, 1)")
            cursor.execute(
                """
                UPDATE entries
                SET image_filename = CAST(timestamp AS TEXT) || '.webp'
                WHERE image_filename IS NULL OR image_filename = ''
                """
            )

    cursor.execute(
        """
        DELETE FROM entries
        WHERE id NOT IN (
            SELECT MAX(id)
            FROM entries
            GROUP BY image_filename
        )
        """
    )
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_entries_image_filename ON entries (image_filename)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_timestamp ON entries (timestamp)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_timestamp_monitor ON entries (timestamp, monitor_id)"
    )


def create_db() -> None:
    """
    Creates the SQLite database and the 'entries' table if they don't exist.

    The table schema includes columns for an auto-incrementing ID, application name,
    window title, extracted text, timestamp, and text embedding.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            _ensure_entries_schema(cursor)
            conn.commit()
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
                SELECT id, app, title, text, timestamp, monitor_id, image_filename, embedding
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
                        image_filename=row["image_filename"],
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
    """Retrieves timeline entries with monitor identity and image filename."""
    timeline_entries: List[TimelineEntry] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT timestamp, monitor_id, image_filename
                FROM entries
                ORDER BY timestamp DESC, monitor_id ASC
                """
            )
            results = cursor.fetchall()
            for row in results:
                timeline_entries.append(
                    TimelineEntry(
                        timestamp=row["timestamp"],
                        monitor_id=row["monitor_id"],
                        image_filename=row["image_filename"],
                    )
                )
    except sqlite3.Error as e:
        print(f"Database error while fetching timeline entries: {e}")
    return timeline_entries


def insert_entry(
    text: str,
    timestamp: int,
    embedding: np.ndarray,
    app: str,
    title: str,
    monitor_id: int = 1,
    image_filename: Optional[str] = None,
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
        image_filename (Optional[str]): The screenshot filename on disk.

    Returns:
        Optional[int]: The ID of the newly inserted row, or None if insertion fails.
                       Prints an error message to stderr on failure.
    """
    embedding_bytes: bytes = embedding.astype(np.float32).tobytes() # Ensure consistent dtype
    if not image_filename:
        image_filename = f"{timestamp}_m{monitor_id}.webp"

    last_row_id: Optional[int] = None
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO entries (text, timestamp, monitor_id, image_filename, embedding, app, title)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_filename) DO NOTHING
                """,
                (
                    text,
                    timestamp,
                    monitor_id,
                    image_filename,
                    embedding_bytes,
                    app,
                    title,
                ),
            )
            conn.commit()
            if cursor.rowcount > 0: # Check if insert actually happened
                last_row_id = cursor.lastrowid
            # else:
                # Optionally log that a duplicate timestamp was encountered
                # print(f"Skipped inserting entry with duplicate timestamp: {timestamp}")

    except sqlite3.Error as e:
        # More specific error handling can be added (e.g., IntegrityError for UNIQUE constraint)
        print(f"Database error during insertion: {e}")
    return last_row_id
