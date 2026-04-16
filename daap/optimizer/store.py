"""
SQLite persistence for bandit state.

Tables:
  bandit_state: stores B matrix and f vector per (user_id, role, arm)
  bandit_observations: log of every update (for debugging + retraining)
"""

import json
import sqlite3
from contextlib import contextmanager
import numpy as np
from pathlib import Path


class BanditStore:
    """
    Persist and load bandit state from SQLite.

    Stores per (user_id, role, arm):
      - B matrix (flattened JSON)
      - f vector (JSON)
      - n_pulls count

    Also logs every observation for debugging.
    """

    def __init__(self, db_path: str = "daap_optimizer.db"):
        self.db_path = str(Path(db_path))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        """Open a connection, commit on success, always close (needed on Windows)."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bandit_state (
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    arm TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    B_flat TEXT NOT NULL,
                    f_vec TEXT NOT NULL,
                    n_pulls INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, role, arm)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bandit_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    context_vec TEXT NOT NULL,
                    role TEXT NOT NULL,
                    arm TEXT NOT NULL,
                    reward REAL NOT NULL,
                    topology_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_obs_user
                ON bandit_observations(user_id)
            """)

    def save_arm_state(
        self,
        user_id: str,
        role: str,
        arm: str,
        dim: int,
        B: np.ndarray,
        f: np.ndarray,
        n_pulls: int,
    ):
        """Save or update one arm's state."""
        B_flat = json.dumps(B.flatten().tolist())
        f_vec = json.dumps(f.tolist())

        with self._connect() as conn:
            conn.execute("""
                INSERT INTO bandit_state (user_id, role, arm, dim, B_flat, f_vec, n_pulls)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, role, arm) DO UPDATE SET
                    B_flat = excluded.B_flat,
                    f_vec = excluded.f_vec,
                    n_pulls = excluded.n_pulls,
                    updated_at = CURRENT_TIMESTAMP
            """, (user_id, role, arm, dim, B_flat, f_vec, n_pulls))

    def load_arm_state(
        self,
        user_id: str,
        role: str,
        arm: str,
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        """
        Load one arm's state.

        Returns:
            (B, f, n_pulls) or None if not found
        """
        with self._connect() as conn:
            row = conn.execute("""
                SELECT dim, B_flat, f_vec, n_pulls
                FROM bandit_state
                WHERE user_id = ? AND role = ? AND arm = ?
            """, (user_id, role, arm)).fetchone()

        if row is None:
            return None

        dim, B_flat_json, f_vec_json, n_pulls = row
        B = np.array(json.loads(B_flat_json)).reshape(dim, dim)
        f = np.array(json.loads(f_vec_json))
        return B, f, n_pulls

    def load_optimizer(self, user_id: str, dim: int) -> dict:
        """
        Load all bandit states for a user.

        Returns:
            {role: {arm: (B, f, n_pulls)}}
        """
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT role, arm, dim, B_flat, f_vec, n_pulls
                FROM bandit_state
                WHERE user_id = ?
            """, (user_id,)).fetchall()

        result = {}
        for role, arm, d, B_flat_json, f_vec_json, n_pulls in rows:
            if role not in result:
                result[role] = {}
            B = np.array(json.loads(B_flat_json)).reshape(d, d)
            f = np.array(json.loads(f_vec_json))
            result[role][arm] = (B, f, n_pulls)

        return result

    def save_optimizer(self, user_id: str, optimizer):
        """Save full optimizer state for a user."""
        for role, bandit in optimizer.bandits.items():
            for arm_name, state in bandit.arm_states.items():
                self.save_arm_state(
                    user_id=user_id,
                    role=role,
                    arm=arm_name,
                    dim=state.dim,
                    B=state.B,
                    f=state.f,
                    n_pulls=state.n_pulls,
                )

    def log_observation(
        self,
        user_id: str,
        context_vec: list[float],
        role: str,
        arm: str,
        reward: float,
        topology_id: str = None,
    ):
        """Log a single observation for debugging."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO bandit_observations
                (user_id, context_vec, role, arm, reward, topology_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, json.dumps(context_vec), role, arm, reward, topology_id))

    def get_user_run_count(self, user_id: str) -> int:
        """Count total observations for a user."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT COUNT(*) FROM bandit_observations WHERE user_id = ?
            """, (user_id,)).fetchone()
        return row[0] if row else 0

    def get_profile_summary(self, user_id: str) -> list[dict]:
        """
        Return per-role optimizer summary for /profile display.

        One entry per role: the arm with the most pulls.
        Pure SQL — no numpy loading.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, arm, n_pulls
                FROM bandit_state
                WHERE user_id = ?
                ORDER BY role ASC, n_pulls DESC
                """,
                (user_id,),
            ).fetchall()

        seen: set[str] = set()
        result: list[dict] = []
        for role, arm, n_pulls in rows:
            if role not in seen:
                seen.add(role)
                result.append({"role": role, "best_arm": arm, "n_pulls": n_pulls})
        return result
