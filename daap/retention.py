"""
Retention/TTL configuration shared across DAAP persistence stores.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_data_dir() -> Path:
    """Return the DAAP data directory for persistent storage.

    Reads DAAP_DATA_DIR env var; defaults to ~/.daap/.
    All stores should write their SQLite files here so data survives
    container restarts (the Docker volume mounts this path).
    """
    raw = os.environ.get("DAAP_DATA_DIR", "").strip()
    if raw:
        p = Path(raw)
    else:
        p = Path.home() / ".daap"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_positive_int(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        value = int(raw)
        if value > 0:
            return value
    except Exception:
        pass
    return default


def get_retention_days(default: int = 30) -> int:
    """Retention window for persisted run/state data."""
    return _read_positive_int("DAAP_RETENTION_DAYS", default)


def get_session_ttl_hours(default: int = 24) -> int:
    """TTL window for inactive API sessions."""
    return _read_positive_int("DAAP_SESSION_TTL_HOURS", default)

