"""
DAAP Feedback Store — SQLite persistence for run history and user ratings.

Stores enough data for Phase 2 RL optimization:
topology used, execution metrics, user rating, comment.
"""

import json
import sqlite3
import time
from pathlib import Path

from daap.retention import get_data_dir, get_retention_days


class FeedbackStore:
    """
    SQLite store for run history, ratings, and metrics.

    Phase 1: local SQLite — zero setup.
    Phase 3: migrate to Postgres if multi-node deployment is needed.
    """

    def __init__(self, db_path: str | None = None, retention_days: int | None = None):
        resolved = Path(db_path) if db_path else get_data_dir() / "daap_feedback.db"
        self.db_path = str(resolved)
        self.retention_days = retention_days or get_retention_days()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id            TEXT    NOT NULL,
                    timestamp             REAL    NOT NULL,
                    topology_json         TEXT,
                    execution_result      TEXT,
                    total_cost_usd        REAL,
                    total_latency_seconds REAL,
                    success               INTEGER,
                    error                 TEXT,
                    rating                INTEGER,
                    comment               TEXT
                )
            """)
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runs_timestamp
                ON runs(timestamp)
                """
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def store_run(
        self,
        session_id: str,
        topology_json: dict | None,
        execution_result: dict | None,
    ) -> int:
        """Insert a completed run record. Returns the new row id."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO runs
                   (session_id, timestamp, topology_json, execution_result,
                    total_cost_usd, total_latency_seconds, success, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    time.time(),
                    json.dumps(topology_json) if topology_json else None,
                    json.dumps(execution_result) if execution_result else None,
                    execution_result.get("cost_usd", 0) if execution_result else 0,
                    execution_result.get("latency_seconds", 0) if execution_result else 0,
                    1 if execution_result and execution_result.get("success") else 0,
                    execution_result.get("error") if execution_result else None,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def store_rating(
        self,
        session_id: str,
        rating: int,
        comment: str = "",
        topology_json: dict | None = None,
        execution_result: dict | None = None,
    ) -> None:
        """
        Attach a rating to the most recent run for this session.
        If no run exists yet, inserts a new row with the rating.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id FROM runs WHERE session_id = ? ORDER BY timestamp DESC LIMIT 1",
                (session_id,),
            )
            row = cursor.fetchone()
            if row:
                conn.execute(
                    "UPDATE runs SET rating = ?, comment = ? WHERE id = ?",
                    (rating, comment, row[0]),
                )
            else:
                conn.execute(
                    """INSERT INTO runs
                       (session_id, timestamp, topology_json, execution_result,
                        rating, comment, success)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        time.time(),
                        json.dumps(topology_json) if topology_json else None,
                        json.dumps(execution_result) if execution_result else None,
                        rating,
                        comment,
                        1,
                    ),
                )
            conn.commit()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_runs_for_session(self, session_id: str) -> list[dict]:
        """Return all runs for a session, newest first."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM runs WHERE session_id = ? ORDER BY timestamp DESC",
                (session_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_rated_runs(self) -> list[dict]:
        """Return all runs that have a user rating. Used for Phase 2 RL."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM runs WHERE rating IS NOT NULL ORDER BY timestamp DESC"
            )
            return [dict(row) for row in cursor.fetchall()]

    def purge_expired(self, retention_days: int | None = None) -> int:
        """Delete run rows older than the configured retention window."""
        days = retention_days or self.retention_days
        cutoff = time.time() - days * 86400
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM runs WHERE timestamp < ?", (cutoff,))
            conn.commit()
            return cur.rowcount
