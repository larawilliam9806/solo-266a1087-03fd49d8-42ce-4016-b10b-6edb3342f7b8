"""SQLite metadata layer.

Two tables:

* ``files``       -- registry of every parquet file that belongs to a daily
                     partition.  ``state`` is ``active`` (visible to readers)
                     or ``retired`` (replaced by compaction, waiting for
                     in-flight readers to drain before the bytes are
                     unlinked).  ``materialized`` records whether the file's
                     hourly buckets have been persisted.
* ``compactions`` -- one row per compaction job.  A partial unique index
                     guarantees at most one ``running`` job per day, which is
                     what lets us reject concurrent compactions for the same
                     day atomically (even across a process restart).
* ``file_hours``  -- per-file UTC hourly buckets (``obs_count``, salinity
                     ``sal_sum``), written in the *same transaction* that
                     registers or retires the file, so no file is ever
                     counted twice and no observation is ever dropped by a
                     compaction switch.
* ``hourly_summaries``
                     -- the last completed aggregate per day plus its
                     row/bucket counts, used to skip recomputation and only
                     trusted while it matches the current per-file buckets.

The compaction *switch* (retire inputs + register output + mark job done)
happens in a single SQLite transaction, so a crash can never leave the
registry half-switched.
"""

from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    day        TEXT NOT NULL,
    path       TEXT NOT NULL UNIQUE,
    kind       TEXT NOT NULL CHECK (kind IN ('upload', 'compacted')),
    state      TEXT NOT NULL CHECK (state IN ('active', 'retired')),
    rows       INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    -- 1 once this file's rows have been reduced to ``file_hours`` buckets;
    -- 0 only for files that predate hourly summaries / crashed backfills.
    materialized INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_files_day_state ON files(day, state);

CREATE TABLE IF NOT EXISTS compactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    day         TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('running', 'done', 'failed')),
    input_paths TEXT NOT NULL,          -- JSON array, snapshot taken at job start
    input_rows  INTEGER NOT NULL,
    output_path TEXT,
    output_rows INTEGER,
    error       TEXT,
    created_at  TEXT NOT NULL,
    finished_at TEXT
);
-- At most one running compaction per day (enforced by the DB itself).
CREATE UNIQUE INDEX IF NOT EXISTS ux_compactions_one_running_per_day
    ON compactions(day) WHERE status = 'running';

-- Hourly buckets for one parquet file, keyed by the file registry path.
CREATE TABLE IF NOT EXISTS file_hours (
    path       TEXT NOT NULL,
    buoy_id    TEXT NOT NULL,
    hour_utc   TEXT NOT NULL,           -- YYYY-MM-DDTHH:00:00Z
    obs_count  INTEGER NOT NULL,
    sal_sum    REAL NOT NULL,
    PRIMARY KEY (path, buoy_id, hour_utc)
);

-- Completed day summaries.  Inserted only once every active file of the day
-- has its buckets, in the same transaction, so readers never see partials.
CREATE TABLE IF NOT EXISTS hourly_summaries (
    day          TEXT PRIMARY KEY,
    row_count    INTEGER NOT NULL,     -- observations counted (duplicates included)
    bucket_count INTEGER NOT NULL,     -- distinct (buoy, hour) buckets
    computed_at  TEXT NOT NULL
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    """Open a short-lived connection with sane durability/concurrency pragmas."""
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # FULL so a committed upload / compaction switch survives a power cut.
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_schema(db_path: str) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring a database created by an older version up to the current schema.

    Adding ``files.materialized`` is the only migration so far.  Existing
    files default to 0, i.e. their hourly buckets are backfilled lazily on
    the first hourly query for their day.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(files)").fetchall()}
    if "materialized" not in cols:
        conn.execute(
            "ALTER TABLE files ADD COLUMN materialized INTEGER NOT NULL DEFAULT 0"
        )
