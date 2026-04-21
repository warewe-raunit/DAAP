"""Runtime capability snapshot helpers for the master agent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _safe_memory_status() -> dict[str, Any]:
    try:
        from daap.memory.observability import get_memory_status

        return get_memory_status()
    except Exception:
        return {"available": False, "reason": "memory observability unavailable"}


def _safe_connected_mcp_servers() -> list[str]:
    try:
        from daap.mcpx.manager import get_mcp_manager

        manager = get_mcp_manager()
        return manager.list_connected()
    except Exception:
        return []


def build_master_runtime_snapshot(
    toolkit,
    *,
    execution_mode: str,
    memory_enabled: bool | None = None,
    optimizer_enabled: bool | None = None,
    topology_store_enabled: bool | None = None,
    feedback_store_enabled: bool | None = None,
    session_store_enabled: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, serializable runtime snapshot for prompt grounding."""
    snapshot: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agent_identity": "DAAP Master Agent",
        "agent_role": "orchestrator",
        "execution_mode": execution_mode,
        "master_tools": sorted(list(getattr(toolkit, "tools", {}).keys())),
        "connected_mcp_servers": _safe_connected_mcp_servers(),
        "memory_status": _safe_memory_status(),
        "feature_flags": {},
    }

    flags = snapshot["feature_flags"]
    if memory_enabled is not None:
        flags["memory_enabled"] = bool(memory_enabled)
    if optimizer_enabled is not None:
        flags["optimizer_enabled"] = bool(optimizer_enabled)
    if topology_store_enabled is not None:
        flags["topology_store_enabled"] = bool(topology_store_enabled)
    if feedback_store_enabled is not None:
        flags["feedback_store_enabled"] = bool(feedback_store_enabled)
    if session_store_enabled is not None:
        flags["session_store_enabled"] = bool(session_store_enabled)

    if isinstance(extra, dict):
        snapshot.update(extra)

    return snapshot
