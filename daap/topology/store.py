"""
DAAP Topology Store — SQLite persistence for saved topologies and run history.

Each topology is identified by (topology_id, version).
list_topologies returns the latest version per topology_id.
Run history is capped at max_runs per topology (oldest purged when over limit).
Soft-delete sets deleted_at; purge_expired hard-deletes past TTL.
"""

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path

from daap.retention import get_data_dir, get_retention_days
from daap.topology.models import StoredTopology, TopologyRun
from daap.topology.naming import auto_name_from_prompt

logger = logging.getLogger(__name__)


class TopologyStore:
    """SQLite store for saved topologies and their run history."""

    def __init__(self, db_path: str | None = None, retention_days: int | None = None):
        resolved = Path(db_path) if db_path else get_data_dir() / "daap_topology.db"
        self.db_path = str(resolved)
        self.retention_days = retention_days or get_retention_days()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS topologies (
                    topology_id     TEXT    NOT NULL,
                    version         INTEGER NOT NULL DEFAULT 1,
                    user_id         TEXT    NOT NULL DEFAULT 'default',
                    name            TEXT    NOT NULL,
                    spec_json       TEXT    NOT NULL,
                    created_at      REAL    NOT NULL,
                    updated_at      REAL    NOT NULL,
                    deleted_at      REAL,
                    delete_ttl_days INTEGER NOT NULL DEFAULT 30,
                    max_runs        INTEGER NOT NULL DEFAULT 10,
                    PRIMARY KEY (topology_id, version)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS topology_runs (
                    run_id           TEXT    PRIMARY KEY,
                    topology_id      TEXT    NOT NULL,
                    topology_version INTEGER NOT NULL,
                    user_id          TEXT    NOT NULL,
                    ran_at           REAL    NOT NULL,
                    user_prompt      TEXT,
                    result_json      TEXT,
                    success          INTEGER NOT NULL DEFAULT 0,
                    latency_seconds  REAL    NOT NULL DEFAULT 0.0,
                    input_tokens     INTEGER NOT NULL DEFAULT 0,
                    output_tokens    INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_topologies_user
                ON topologies(user_id, deleted_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runs_topology
                ON topology_runs(topology_id, ran_at DESC)
                """
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_topology(row: sqlite3.Row) -> StoredTopology:
        return StoredTopology(
            topology_id=row["topology_id"],
            version=row["version"],
            user_id=row["user_id"],
            name=row["name"],
            spec=json.loads(row["spec_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            deleted_at=row["deleted_at"],
            max_runs=row["max_runs"],
        )

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> TopologyRun:
        result_raw = row["result_json"]
        return TopologyRun(
            run_id=row["run_id"],
            topology_id=row["topology_id"],
            topology_version=row["topology_version"],
            user_id=row["user_id"],
            ran_at=row["ran_at"],
            user_prompt=row["user_prompt"],
            result=json.loads(result_raw) if result_raw else None,
            success=bool(row["success"]),
            latency_seconds=row["latency_seconds"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
        )

    @staticmethod
    def _coerce_int(value, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _get_latest_version(self, conn: sqlite3.Connection, topology_id: str) -> int | None:
        row = conn.execute(
            "SELECT MAX(version) AS max_ver FROM topologies WHERE topology_id = ?",
            (topology_id,),
        ).fetchone()
        return row["max_ver"] if row and row["max_ver"] is not None else None

    # ------------------------------------------------------------------
    # Write — topology
    # ------------------------------------------------------------------

    def save_topology(
        self,
        spec: dict,
        user_id: str,
        name: str | None = None,
        overwrite: bool = True,
    ) -> StoredTopology:
        """
        Save a topology spec.

        overwrite=True: replace latest version in place.
        overwrite=False: insert new version with incremented version number.
        """
        topology_id = spec.get("topology_id") or f"topo-{uuid.uuid4().hex[:8]}"
        now = time.time()
        resolved_name = name or auto_name_from_prompt(spec.get("user_prompt", ""))

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            latest_version = self._get_latest_version(conn, topology_id)

            created_at = now
            max_runs = 10
            delete_ttl_days = self.retention_days

            if latest_version is not None:
                latest_row = conn.execute(
                    "SELECT created_at, max_runs, delete_ttl_days FROM topologies WHERE topology_id = ? AND version = ?",
                    (topology_id, latest_version),
                ).fetchone()
                created_at = latest_row["created_at"]
                max_runs = latest_row["max_runs"]
                delete_ttl_days = latest_row["delete_ttl_days"]

            if overwrite and latest_version is not None:
                version = latest_version
            elif not overwrite and latest_version is not None:
                version = latest_version + 1
            else:
                version = 1

            spec_to_save = dict(spec)
            spec_to_save["topology_id"] = topology_id
            spec_to_save["version"] = version

            conn.execute(
                """
                INSERT OR REPLACE INTO topologies
                    (topology_id, version, user_id, name, spec_json,
                     created_at, updated_at, deleted_at, delete_ttl_days, max_runs)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    topology_id,
                    version,
                    user_id,
                    resolved_name,
                    json.dumps(spec_to_save),
                    created_at,
                    now,
                    None,
                    delete_ttl_days,
                    max_runs,
                ),
            )
            conn.commit()

            row = conn.execute(
                "SELECT * FROM topologies WHERE topology_id = ? AND version = ?",
                (topology_id, version),
            ).fetchone()
            return self._row_to_topology(row)

    def rename_topology(self, topology_id: str, new_name: str) -> None:
        """Update the name across all versions of a topology."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE topologies SET name = ?, updated_at = ? WHERE topology_id = ?",
                (new_name, time.time(), topology_id),
            )
            conn.commit()

    def set_max_runs(self, topology_id: str, max_runs: int) -> None:
        """Set run-history cap for all versions of a topology."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE topologies SET max_runs = ? WHERE topology_id = ?",
                (max_runs, topology_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Write — runs
    # ------------------------------------------------------------------

    def save_run(
        self,
        topology_id: str,
        topology_version: int,
        user_id: str,
        result: dict,
        user_prompt: str | None = None,
    ) -> TopologyRun:
        """Save one execution run and enforce run-cap by deleting oldest runs."""
        run_id = str(uuid.uuid4())
        now = time.time()

        latency = result.get("latency_seconds") or 0.0
        input_tokens = result.get("total_input_tokens", result.get("input_tokens", 0))
        output_tokens = result.get("total_output_tokens", result.get("output_tokens", 0))

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute(
                """
                INSERT INTO topology_runs
                    (run_id, topology_id, topology_version, user_id, ran_at,
                     user_prompt, result_json, success, latency_seconds,
                     input_tokens, output_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    topology_id,
                    topology_version,
                    user_id,
                    now,
                    user_prompt,
                    json.dumps(result),
                    1 if result.get("success") else 0,
                    float(latency),
                    self._coerce_int(input_tokens),
                    self._coerce_int(output_tokens),
                ),
            )

            cap_row = conn.execute(
                "SELECT max_runs FROM topologies WHERE topology_id = ? ORDER BY version DESC LIMIT 1",
                (topology_id,),
            ).fetchone()
            max_runs = cap_row["max_runs"] if cap_row else 10

            excess_rows = conn.execute(
                """
                SELECT run_id FROM topology_runs
                WHERE topology_id = ?
                ORDER BY ran_at DESC
                LIMIT -1 OFFSET ?
                """,
                (topology_id, max_runs),
            ).fetchall()

            if excess_rows:
                excess_ids = [row["run_id"] for row in excess_rows]
                placeholders = ",".join(["?"] * len(excess_ids))
                conn.execute(
                    f"DELETE FROM topology_runs WHERE run_id IN ({placeholders})",
                    excess_ids,
                )

            conn.commit()

            row = conn.execute(
                "SELECT * FROM topology_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            return self._row_to_run(row)

    # ------------------------------------------------------------------
    # Soft delete / restore / purge
    # ------------------------------------------------------------------

    def delete_topology(self, topology_id: str, ttl_days: int | None = None) -> None:
        """Soft-delete all versions of a topology."""
        effective_ttl = ttl_days if ttl_days is not None else self.retention_days
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE topologies
                SET deleted_at = ?, delete_ttl_days = ?
                WHERE topology_id = ?
                """,
                (time.time(), effective_ttl, topology_id),
            )
            conn.commit()

    def restore_topology(self, topology_id: str) -> None:
        """Restore all versions of a soft-deleted topology."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE topologies SET deleted_at = NULL WHERE topology_id = ?",
                (topology_id,),
            )
            conn.commit()

    def purge_expired(self) -> int:
        """
        Hard-delete topologies past their delete TTL.

        Also removes associated run rows. Returns number of topology rows deleted.
        """
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            expired = conn.execute(
                """
                SELECT DISTINCT topology_id
                FROM topologies
                WHERE deleted_at IS NOT NULL
                  AND (deleted_at + delete_ttl_days * 86400) < ?
                """,
                (now,),
            ).fetchall()
            expired_ids = [row["topology_id"] for row in expired]

            if not expired_ids:
                return 0

            placeholders = ",".join(["?"] * len(expired_ids))
            conn.execute(
                f"DELETE FROM topology_runs WHERE topology_id IN ({placeholders})",
                expired_ids,
            )
            result = conn.execute(
                f"DELETE FROM topologies WHERE topology_id IN ({placeholders})",
                expired_ids,
            )
            conn.commit()

            deleted_count = result.rowcount
            logger.info("Purged %d expired topology rows", deleted_count)
            return deleted_count

    def purge_old_runs(self, retention_days: int | None = None) -> int:
        """Hard-delete run rows older than retention_days."""
        days = retention_days or self.retention_days
        cutoff = time.time() - days * 86400
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM topology_runs WHERE ran_at < ?", (cutoff,))
            conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------
    # Read — topology
    # ------------------------------------------------------------------

    def get_topology(self, topology_id: str, version: int | None = None) -> StoredTopology | None:
        """Fetch a topology by ID; version=None returns latest."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if version is None:
                row = conn.execute(
                    """
                    SELECT * FROM topologies
                    WHERE topology_id = ?
                    ORDER BY version DESC
                    LIMIT 1
                    """,
                    (topology_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM topologies WHERE topology_id = ? AND version = ?",
                    (topology_id, version),
                ).fetchone()
            return self._row_to_topology(row) if row else None

    def list_versions(self, topology_id: str) -> list[StoredTopology]:
        """List all versions for a topology, newest first."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM topologies WHERE topology_id = ? ORDER BY version DESC",
                (topology_id,),
            ).fetchall()
            return [self._row_to_topology(row) for row in rows]

    def count_runs(self, user_id: str) -> int:
        """Count total execution runs for a user."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM topology_runs WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return row[0] if row else 0

    def list_topologies(self, user_id: str, include_deleted: bool = False) -> list[StoredTopology]:
        """
        List one row per topology_id (latest version per topology).

        By default, soft-deleted topologies are excluded.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            deleted_filter = "" if include_deleted else "AND t.deleted_at IS NULL"
            rows = conn.execute(
                f"""
                SELECT t.*
                FROM topologies t
                WHERE t.user_id = ?
                  {deleted_filter}
                  AND t.version = (
                      SELECT MAX(t2.version)
                      FROM topologies t2
                      WHERE t2.topology_id = t.topology_id
                        AND t2.user_id = t.user_id
                  )
                ORDER BY t.updated_at DESC
                """,
                (user_id,),
            ).fetchall()
            return [self._row_to_topology(row) for row in rows]

    # ------------------------------------------------------------------
    # Read — runs
    # ------------------------------------------------------------------

    def get_runs(self, topology_id: str, limit: int | None = None) -> list[TopologyRun]:
        """Return run history for a topology, newest first."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if limit is None:
                rows = conn.execute(
                    """
                    SELECT * FROM topology_runs
                    WHERE topology_id = ?
                    ORDER BY ran_at DESC
                    """,
                    (topology_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM topology_runs
                    WHERE topology_id = ?
                    ORDER BY ran_at DESC
                    LIMIT ?
                    """,
                    (topology_id, limit),
                ).fetchall()
            return [self._row_to_run(row) for row in rows]
