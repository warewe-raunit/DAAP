"""
Mem0 configuration.

Uses:
  - text-embedding-3-small via OpenRouter for embeddings
  - Gemini 2.5 Flash Lite via OpenRouter for extraction (cheapest viable)
  - Qdrant local for vector storage (persistent) OR in-memory for testing
  - Async writes (production default since Mem0 v1.0.0)

Required env vars:
  OPENROUTER_API_KEY  - for both embeddings and fact extraction LLM
  QDRANT_HOST         - optional, defaults to localhost
  QDRANT_PORT         - optional, defaults to 6333
"""

import os
from typing import Literal

from daap.memory.observability import (
    record_memory_error,
    record_memory_event,
    set_memory_status,
)


# ============================================================================
# Configuration builders
# ============================================================================

def build_config(mode: Literal["production", "testing"] = "production") -> dict:
    """
    Build Mem0 config dict.

    mode="production": Qdrant persistent store. Survives restarts.
    mode="testing": In-memory. Fast. No persistence. For pytest.
    """
    llm_config = {
        "provider": "openai",  # OpenRouter is OpenAI-compatible
        "config": {
            "model": "google/gemini-2.5-flash-lite",
            "temperature": 0.0,
            "max_tokens": 2000,
            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "openai_base_url": "https://openrouter.ai/api/v1",
        },
    }

    embedder_config = {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "openai_base_url": "https://openrouter.ai/api/v1",
        },
    }

    config = {
        "llm": llm_config,
        "embedder": embedder_config,
        "version": "v1.1",
    }

    if mode == "production":
        config["vector_store"] = {
            "provider": "qdrant",
            "config": {
                "collection_name": "daap_memories",
                "host": os.environ.get("QDRANT_HOST", "localhost"),
                "port": int(os.environ.get("QDRANT_PORT", 6333)),
                "embedding_model_dims": 1536,  # text-embedding-3-small dim
            },
        }
    # For testing mode, omit vector_store → Mem0 uses in-memory default

    return config


# ============================================================================
# Singleton client
# ============================================================================

_memory_client = None


def get_memory_client(mode: Literal["production", "testing"] = "production"):
    """
    Get or create singleton Mem0 client.

    Lazy init: client created on first call, reused thereafter.
    In tests, call reset_memory_client() between tests.
    """
    global _memory_client
    if _memory_client is None:
        try:
            from mem0 import Memory

            _memory_client = Memory.from_config(build_config(mode))
            set_memory_status(True, f"initialized ({mode})")
            record_memory_event("init_client", True)
        except Exception as exc:
            set_memory_status(False, f"init failed ({mode})")
            record_memory_error("init_client", exc)
            record_memory_event("init_client", False)
            raise
    return _memory_client


def reset_memory_client():
    """Reset the singleton. Used in tests and after config changes."""
    global _memory_client
    _memory_client = None


# ============================================================================
# Health check
# ============================================================================

def check_memory_available() -> tuple[bool, str]:
    """
    Check if memory can be initialized.

    Returns (ok, reason). ok=True if memory is available.
    Used by startup to decide whether to enable memory features.
    """
    if not os.environ.get("OPENROUTER_API_KEY"):
        reason = "OPENROUTER_API_KEY not set (required for embeddings and extraction LLM)"
        set_memory_status(False, reason)
        record_memory_event("health_check", False)
        return False, reason
    try:
        get_memory_client("production")
        set_memory_status(True, "ok")
        record_memory_event("health_check", True)
        return True, "ok"
    except Exception as e:
        reason = f"Mem0 init failed: {e}"
        set_memory_status(False, reason)
        record_memory_error("health_check", e)
        record_memory_event("health_check", False)
        return False, reason
