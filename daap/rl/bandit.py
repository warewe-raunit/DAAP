"""Thompson Sampling bandit with SQLite persistence for topology optimization."""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - fallback path
    np = None

logger = logging.getLogger(__name__)

ARM_IDS: tuple[str, ...] = (
    "budget_1",
    "budget_2",
    "budget_3",
    "balanced_1",
    "balanced_2",
    "balanced_3",
    "quality_1",
    "quality_2",
    "quality_3",
)

ARM_CONFIGS: dict[str, dict[str, Any]] = {
    "budget_1": {"profile": "budget", "instance_count": 1},
    "budget_2": {"profile": "budget", "instance_count": 2},
    "budget_3": {"profile": "budget", "instance_count": 3},
    "balanced_1": {"profile": "balanced", "instance_count": 1},
    "balanced_2": {"profile": "balanced", "instance_count": 2},
    "balanced_3": {"profile": "balanced", "instance_count": 3},
    "quality_1": {"profile": "quality", "instance_count": 1},
    "quality_2": {"profile": "quality", "instance_count": 2},
    "quality_3": {"profile": "quality", "instance_count": 3},
}

TIER_PROFILES: dict[str, dict[str, str]] = {
    "budget": {
        "default": "fast",
    },
    "balanced": {
        "tool_or_react": "fast",
        "writing_or_eval": "smart",
        "default": "smart",
    },
    "quality": {
        "tool_or_react": "fast",
        "middle": "smart",
        "synthesis": "powerful",
    },
}


@dataclass
class ArmState:
    arm_id: str
    task_type: str
    alpha: float
    beta: float
    n_pulls: int
    last_updated: float


class ThompsonBandit:
    """Contextual Thompson Sampling over discrete topology configuration arms."""

    def __init__(self, db_path: str = "optimizer.db"):
        self.db_path = str(Path(db_path))
        db_parent = Path(self.db_path).parent
        if db_parent and str(db_parent) not in ("", "."):
            db_parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bandit_arms (
                    arm_id       TEXT NOT NULL,
                    task_type    TEXT NOT NULL,
                    alpha        REAL NOT NULL DEFAULT 1.0,
                    beta         REAL NOT NULL DEFAULT 1.0,
                    n_pulls      INTEGER NOT NULL DEFAULT 0,
                    last_updated REAL NOT NULL,
                    PRIMARY KEY (arm_id, task_type)
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bandit_experiences (
                    exp_id           TEXT PRIMARY KEY,
                    task_fingerprint TEXT NOT NULL,
                    arm_id           TEXT NOT NULL,
                    reward           REAL NOT NULL,
                    topology_json    TEXT,
                    outcome_json     TEXT,
                    timestamp        REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_exp_fingerprint
                ON bandit_experiences(task_fingerprint, timestamp DESC);
                """
            )
            conn.commit()

    def _ensure_arms(self, task_type: str) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO bandit_arms(
                    arm_id, task_type, alpha, beta, n_pulls, last_updated
                )
                VALUES (?, ?, 1.0, 1.0, 0, ?)
                """,
                [(arm_id, task_type, now) for arm_id in ARM_IDS],
            )
            conn.commit()

    def _load_arms(self, task_type: str) -> dict[str, ArmState]:
        self._ensure_arms(task_type)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT arm_id, task_type, alpha, beta, n_pulls, last_updated
                FROM bandit_arms
                WHERE task_type = ?
                """,
                (task_type,),
            ).fetchall()

        states: dict[str, ArmState] = {}
        for row in rows:
            states[row["arm_id"]] = ArmState(
                arm_id=row["arm_id"],
                task_type=row["task_type"],
                alpha=float(row["alpha"]),
                beta=float(row["beta"]),
                n_pulls=int(row["n_pulls"]),
                last_updated=float(row["last_updated"]),
            )
        return states

    @staticmethod
    def _clamp_reward(reward: float) -> float:
        return max(0.0, min(1.0, float(reward)))

    def _sample_beta(self, alpha: float, beta: float) -> float:
        if np is not None:
            return float(np.random.beta(alpha, beta))
        return float(random.betavariate(alpha, beta))

    def select_arm(self, task_type: str) -> str:
        """Sample all arm posteriors and return arm with max sampled theta."""
        normalized_task = (task_type or "general").strip().lower() or "general"

        with self._lock:
            try:
                states = self._load_arms(normalized_task)
                best_arm = ARM_IDS[0]
                best_score = -1.0
                for arm_id in ARM_IDS:
                    state = states.get(arm_id)
                    if state is None:
                        continue
                    theta = self._sample_beta(state.alpha, state.beta)
                    if theta > best_score:
                        best_score = theta
                        best_arm = arm_id
                return best_arm
            except Exception as exc:
                logger.warning("Bandit select_arm failed, using fallback arm: %s", exc)
                return ARM_IDS[0]

    def update(self, task_type: str, arm_id: str, reward: float) -> None:
        """Update Beta posterior for an arm with reward in [0, 1]."""
        normalized_task = (task_type or "general").strip().lower() or "general"
        chosen_arm = arm_id if arm_id in ARM_CONFIGS else ARM_IDS[0]
        reward_clamped = self._clamp_reward(reward)
        now = time.time()

        with self._lock:
            self._ensure_arms(normalized_task)
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE bandit_arms
                    SET alpha = alpha + ?,
                        beta = beta + ?,
                        n_pulls = n_pulls + 1,
                        last_updated = ?
                    WHERE arm_id = ? AND task_type = ?
                    """,
                    (reward_clamped, 1.0 - reward_clamped, now, chosen_arm, normalized_task),
                )
                conn.commit()

    def adjust(self, task_type: str, arm_id: str, delta_reward: float) -> None:
        """Apply a reward delta correction without incrementing pulls."""
        if abs(float(delta_reward)) < 1e-12:
            return

        normalized_task = (task_type or "general").strip().lower() or "general"
        chosen_arm = arm_id if arm_id in ARM_CONFIGS else ARM_IDS[0]
        delta = float(delta_reward)
        now = time.time()

        with self._lock:
            self._ensure_arms(normalized_task)
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE bandit_arms
                    SET alpha = CASE
                            WHEN alpha + ? < 0.001 THEN 0.001
                            ELSE alpha + ?
                        END,
                        beta = CASE
                            WHEN beta - ? < 0.001 THEN 0.001
                            ELSE beta - ?
                        END,
                        last_updated = ?
                    WHERE arm_id = ? AND task_type = ?
                    """,
                    (delta, delta, delta, delta, now, chosen_arm, normalized_task),
                )
                conn.commit()

    @staticmethod
    def _serialize_json(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _maybe_parse_json(value: str | None) -> Any:
        if value is None:
            return None
        try:
            return json.loads(value)
        except Exception:
            return value

    def log_experience(
        self,
        task_fingerprint: str,
        arm_id: str,
        reward: float,
        topology_json: Any,
        outcome_json: Any,
    ) -> str:
        exp_id = uuid.uuid4().hex
        chosen_arm = arm_id if arm_id in ARM_CONFIGS else ARM_IDS[0]
        reward_clamped = self._clamp_reward(reward)
        now = time.time()

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO bandit_experiences(
                        exp_id, task_fingerprint, arm_id, reward,
                        topology_json, outcome_json, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        exp_id,
                        task_fingerprint,
                        chosen_arm,
                        reward_clamped,
                        self._serialize_json(topology_json),
                        self._serialize_json(outcome_json),
                        now,
                    ),
                )
                conn.commit()
        return exp_id

    def update_experience(
        self,
        exp_id: str,
        reward: float,
        outcome_json: Any | None = None,
    ) -> None:
        reward_clamped = self._clamp_reward(reward)
        payload = self._serialize_json(outcome_json)

        with self._lock:
            with self._connect() as conn:
                if payload is None:
                    conn.execute(
                        """
                        UPDATE bandit_experiences
                        SET reward = ?
                        WHERE exp_id = ?
                        """,
                        (reward_clamped, exp_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE bandit_experiences
                        SET reward = ?, outcome_json = ?
                        WHERE exp_id = ?
                        """,
                        (reward_clamped, payload, exp_id),
                    )
                conn.commit()

    def get_experiences(self, task_fingerprint: str, limit: int = 20) -> list[dict[str, Any]]:
        safe_limit = max(1, min(200, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT exp_id, task_fingerprint, arm_id, reward,
                       topology_json, outcome_json, timestamp
                FROM bandit_experiences
                WHERE task_fingerprint = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (task_fingerprint, safe_limit),
            ).fetchall()

        experiences: list[dict[str, Any]] = []
        for row in rows:
            experiences.append(
                {
                    "exp_id": row["exp_id"],
                    "task_fingerprint": row["task_fingerprint"],
                    "arm_id": row["arm_id"],
                    "reward": float(row["reward"]),
                    "topology_json": self._maybe_parse_json(row["topology_json"]),
                    "outcome_json": self._maybe_parse_json(row["outcome_json"]),
                    "timestamp": float(row["timestamp"]),
                }
            )
        return experiences

    def get_experience(self, exp_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT exp_id, task_fingerprint, arm_id, reward,
                       topology_json, outcome_json, timestamp
                FROM bandit_experiences
                WHERE exp_id = ?
                """,
                (exp_id,),
            ).fetchone()

        if row is None:
            return None
        return {
            "exp_id": row["exp_id"],
            "task_fingerprint": row["task_fingerprint"],
            "arm_id": row["arm_id"],
            "reward": float(row["reward"]),
            "topology_json": self._maybe_parse_json(row["topology_json"]),
            "outcome_json": self._maybe_parse_json(row["outcome_json"]),
            "timestamp": float(row["timestamp"]),
        }

    def find_latest_experience_by_topology_id(
        self,
        topology_id: str,
        scan_limit: int = 500,
    ) -> dict[str, Any] | None:
        target = str(topology_id or "").strip()
        if not target:
            return None

        safe_limit = max(1, min(2000, int(scan_limit)))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT exp_id, task_fingerprint, arm_id, reward,
                       topology_json, outcome_json, timestamp
                FROM bandit_experiences
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        for row in rows:
            outcome = self._maybe_parse_json(row["outcome_json"])
            if isinstance(outcome, dict) and str(outcome.get("topology_id", "")) == target:
                return {
                    "exp_id": row["exp_id"],
                    "task_fingerprint": row["task_fingerprint"],
                    "arm_id": row["arm_id"],
                    "reward": float(row["reward"]),
                    "topology_json": self._maybe_parse_json(row["topology_json"]),
                    "outcome_json": outcome,
                    "timestamp": float(row["timestamp"]),
                }
        return None

    def get_arm_stats(self, task_type: str) -> dict[str, dict[str, float | int]]:
        normalized_task = (task_type or "general").strip().lower() or "general"

        with self._lock:
            states = self._load_arms(normalized_task)

        stats: dict[str, dict[str, float | int]] = {}
        for arm_id in ARM_IDS:
            state = states.get(arm_id)
            if state is None:
                continue
            denom = state.alpha + state.beta
            mean = (state.alpha / denom) if denom > 0 else 0.5
            stats[arm_id] = {
                "alpha": round(state.alpha, 6),
                "beta": round(state.beta, 6),
                "n_pulls": state.n_pulls,
                "mean": round(mean, 6),
                "last_updated": state.last_updated,
            }
        return stats
