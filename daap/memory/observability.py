"""
In-process observability state for DAAP memory.

Memory is optional in DAAP, but degraded mode must be explicit and queryable.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _MemoryState:
    available: bool = False
    backend: str = "mem0"
    reason: str = "uninitialized"
    last_error: str | None = None
    last_error_stage: str | None = None
    last_error_ts: float | None = None
    op_counts: dict[str, dict[str, int]] = field(default_factory=dict)


_STATE = _MemoryState()
_LOCK = threading.Lock()


def set_memory_status(available: bool, reason: str, backend: str = "mem0") -> None:
    with _LOCK:
        _STATE.available = available
        _STATE.reason = reason
        _STATE.backend = backend


def record_memory_event(operation: str, success: bool) -> None:
    with _LOCK:
        stats = _STATE.op_counts.setdefault(operation, {"ok": 0, "error": 0})
        stats["ok" if success else "error"] += 1


def record_memory_error(stage: str, error: Exception | str) -> None:
    with _LOCK:
        _STATE.last_error_stage = stage
        _STATE.last_error = str(error)
        _STATE.last_error_ts = time.time()


def get_memory_status() -> dict:
    with _LOCK:
        return {
            "available": _STATE.available,
            "backend": _STATE.backend,
            "reason": _STATE.reason,
            "last_error": _STATE.last_error,
            "last_error_stage": _STATE.last_error_stage,
            "last_error_ts": _STATE.last_error_ts,
            "operations": dict(_STATE.op_counts),
        }

