"""
DAAP Memory Setup — compatibility wrapper for unified memory configuration.

Runtime memory configuration now lives in daap.memory.config so every memory
entry point (palace/client/routes) uses one backend contract.
"""

from daap.memory.config import get_memory_client


def create_memory_client(mode: str = "production"):
    """
    Create and configure a Mem0 Memory instance.

    Args:
        mode: "production" — Qdrant-backed persistent storage.
              "ephemeral"  — in-memory Qdrant (dev, no Docker needed).

    Returns:
        Configured Mem0 Memory instance.

    Raises:
        ImportError: if mem0ai is not installed.
        Exception:   if provider credentials are missing or unreachable.
    """
    normalized = "testing" if mode in {"testing", "ephemeral"} else "production"
    return get_memory_client(normalized)
