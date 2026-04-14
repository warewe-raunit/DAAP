"""
DAAP Memory Setup — Mem0 client configuration.

Production: OpenRouter Claude Haiku for extraction and OpenRouter embeddings,
            Qdrant for persistence.
Dev/testing: mock at the DaapMemory level — tests never call this.

Memory is an optimization, not a dependency. If setup fails (missing keys,
Qdrant not running) the caller should catch and disable memory gracefully.
"""

import os


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


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
    from mem0 import Memory  # guarded import — mem0 is optional

    base_llm = {
        "provider": "openai",
        "config": {
            "model": "anthropic/claude-haiku-4-5-20251001",
            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "openai_base_url": OPENROUTER_BASE_URL,
            "temperature": 0,
            "max_tokens": 1000,
        },
    }

    base_embedder = {
        "provider": "openai",
        "config": {
            "model": "openai/text-embedding-3-small",
            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "openai_base_url": OPENROUTER_BASE_URL,
        },
    }

    if mode == "ephemeral":
        # In-memory Qdrant — no Docker, no persistence, good for local dev
        config = {
            "llm": base_llm,
            "embedder": base_embedder,
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "daap_memories_dev",
                    "on_disk": False,
                },
            },
        }
    else:
        # Production — persistent Qdrant
        config = {
            "llm": base_llm,
            "embedder": base_embedder,
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "daap_memories",
                    "host": os.environ.get("QDRANT_HOST", "localhost"),
                    "port": int(os.environ.get("QDRANT_PORT", 6333)),
                },
            },
        }

    return Memory.from_config(config)
