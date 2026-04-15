"""
DAAP RL Embedder — cheap/free task embedding for contextual bandit classification.

Model priority:
  1. nvidia/llama-nemotron-embed-vl-1b-v2:free  — $0/token, 50 req/day rate limit
  2. qwen/qwen3-embedding-0.6b                  — ~$0.001/1M tokens, no rate limit
  3. Keyword classification fallback             — $0, always works

All embeddings are cached in SQLite so each unique prompt is only embedded once.
Centroids are pre-defined from seed examples and stored on first computation.

Activation: set DAAP_RL_USE_EMBEDDINGS=1 in environment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_FREE = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
EMBEDDING_MODEL_CHEAP = "qwen/qwen3-embedding-0.6b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Seed examples per task type — averaged to form centroids
# ---------------------------------------------------------------------------

TASK_SEEDS: dict[str, list[str]] = {
    "lead_generation": [
        "Find 50 B2B leads in construction industry",
        "Build a list of prospects matching our ICP",
        "Scrape LinkedIn for VP of Sales contacts",
        "Generate a contact list of SaaS companies with 50-200 employees",
    ],
    "email_outreach": [
        "Write a cold email sequence for our product",
        "Draft personalized outreach emails for each lead",
        "Create follow-up email templates for SDR team",
        "Generate subject lines and email copy for outreach campaign",
    ],
    "qualification": [
        "Score and rank these leads by fit",
        "Qualify prospects against our ICP criteria",
        "Evaluate and filter leads by budget and authority",
        "Rank these contacts by likelihood to convert",
    ],
    "research": [
        "Research the company background and recent news",
        "Find information about this prospect's tech stack",
        "Analyze the market landscape for our product",
        "Investigate competitor pricing and positioning",
    ],
    "content": [
        "Write a blog post about our product features",
        "Create landing page copy for the new campaign",
        "Draft a case study from this customer interview",
        "Generate social media content for product launch",
    ],
    "data_processing": [
        "Parse and clean this CSV of company data",
        "Transform this spreadsheet into structured JSON",
        "Extract and deduplicate email addresses from the file",
        "Normalize company names and domains in this dataset",
    ],
    "general": [
        "Help me with my sales workflow",
        "Automate this business process",
        "Build a multi-step pipeline for my team",
    ],
}


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """SQLite-backed cache: (text_hash, model_id) → embedding vector."""

    def __init__(self, db_path: str = "optimizer.db") -> None:
        self.db_path = db_path
        self._init_table()

    def _init_table(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    text_hash    TEXT NOT NULL,
                    model_id     TEXT NOT NULL,
                    embedding    TEXT NOT NULL,   -- JSON array of floats
                    created_at   REAL NOT NULL,
                    PRIMARY KEY (text_hash, model_id)
                )
            """)
            conn.commit()

    @staticmethod
    def _hash(text: str, model_id: str) -> str:
        return hashlib.sha256(f"{model_id}::{text}".encode()).hexdigest()

    def get(self, text: str, model_id: str) -> list[float] | None:
        key = self._hash(text, model_id)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT embedding FROM embedding_cache WHERE text_hash = ? AND model_id = ?",
                (key, model_id),
            ).fetchone()
        if row:
            try:
                return json.loads(row[0])
            except Exception:
                return None
        return None

    def put(self, text: str, model_id: str, embedding: list[float]) -> None:
        key = self._hash(text, model_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embedding_cache
                    (text_hash, model_id, embedding, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (key, model_id, json.dumps(embedding), time.time()),
            )
            conn.commit()


# ---------------------------------------------------------------------------
# Embedding fetch (with model fallback)
# ---------------------------------------------------------------------------

def _call_embeddings_api(text: str, model_id: str) -> list[float]:
    """
    Call OpenRouter embeddings endpoint via OpenAI-compatible client.
    Raises on failure — caller handles fallback.
    """
    from openai import OpenAI  # transitive dep via agentscope

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={"HTTP-Referer": "https://github.com/warewe-raunit/DAAP"},
    )
    response = client.embeddings.create(model=model_id, input=text)
    return response.data[0].embedding


def get_embedding(text: str, cache: EmbeddingCache | None = None) -> list[float] | None:
    """
    Get embedding for text. Tries free model first, then cheap model.
    Returns None if both fail (caller falls back to keyword classification).

    Args:
        text:  Text to embed (user prompt or seed example).
        cache: Optional EmbeddingCache. If provided, caches result.

    Returns:
        List of floats (embedding vector) or None.
    """
    text = (text or "").strip()
    if not text:
        return None

    for model_id in (EMBEDDING_MODEL_FREE, EMBEDDING_MODEL_CHEAP):
        # Check cache first
        if cache is not None:
            cached = cache.get(text, model_id)
            if cached is not None:
                logger.debug("Embedding cache hit: model=%s", model_id)
                return cached

        try:
            embedding = _call_embeddings_api(text, model_id)
            if cache is not None:
                cache.put(text, model_id, embedding)
            logger.debug("Embedded via %s (%d dims)", model_id, len(embedding))
            return embedding
        except Exception as exc:
            logger.warning("Embedding model %s failed: %s — trying fallback", model_id, exc)

    return None


# ---------------------------------------------------------------------------
# Centroid classification
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / denom)


class CentroidClassifier:
    """
    Classifies task types by cosine similarity to pre-computed centroids.

    Centroids are computed by averaging seed embeddings per task type.
    They are computed once and cached in EmbeddingCache (same db).
    """

    # Special prefix used to distinguish centroid cache entries from prompt entries
    _CENTROID_PREFIX = "__centroid__"

    def __init__(self, cache: EmbeddingCache) -> None:
        self.cache = cache
        self._centroids: dict[str, list[float]] | None = None  # lazy-loaded

    def _build_centroid(self, task_type: str) -> list[float] | None:
        """Average embeddings of all seed examples for a task type."""
        seeds = TASK_SEEDS.get(task_type, [])
        vectors = []
        for seed in seeds:
            emb = get_embedding(seed, self.cache)
            if emb is not None:
                vectors.append(emb)

        if not vectors:
            return None

        arr = np.array(vectors, dtype=np.float32)
        centroid = arr.mean(axis=0).tolist()
        return centroid

    def _load_centroids(self) -> dict[str, list[float]]:
        """
        Load or compute centroids for all task types.
        Centroid vectors are stored in the embedding cache using a special key prefix.
        """
        centroids: dict[str, list[float]] = {}

        for task_type in TASK_SEEDS:
            # Try to load pre-computed centroid from cache
            centroid_key = f"{self._CENTROID_PREFIX}{task_type}"
            cached = self.cache.get(centroid_key, "centroid_v1")
            if cached is not None:
                centroids[task_type] = cached
                continue

            # Compute and store
            centroid = self._build_centroid(task_type)
            if centroid is not None:
                self.cache.put(centroid_key, "centroid_v1", centroid)
                centroids[task_type] = centroid
                logger.info("Built centroid for task type: %s", task_type)

        return centroids

    def classify(self, user_prompt: str) -> str | None:
        """
        Classify user_prompt to a task type by nearest centroid.

        Returns task_type string or None if embedding fails.
        """
        embedding = get_embedding(user_prompt, self.cache)
        if embedding is None:
            return None

        if self._centroids is None:
            self._centroids = self._load_centroids()

        if not self._centroids:
            return None

        best_type = max(
            self._centroids,
            key=lambda t: cosine_similarity(embedding, self._centroids[t]),
        )
        best_sim = cosine_similarity(embedding, self._centroids[best_type])
        logger.debug("Embedding classify: %s → %s (sim=%.3f)", user_prompt[:50], best_type, best_sim)
        return best_type


# ---------------------------------------------------------------------------
# Module-level singleton (one per db_path)
# ---------------------------------------------------------------------------

_classifiers: dict[str, CentroidClassifier] = {}


def get_centroid_classifier(db_path: str = "optimizer.db") -> CentroidClassifier:
    """Return (or create) a CentroidClassifier for the given db_path."""
    if db_path not in _classifiers:
        cache = EmbeddingCache(db_path=db_path)
        _classifiers[db_path] = CentroidClassifier(cache)
    return _classifiers[db_path]
