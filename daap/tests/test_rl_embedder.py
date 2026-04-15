"""
Tests for daap/rl/embedder.py — cache, cosine similarity, centroid classifier.
No real API calls — all embedding calls are mocked.
"""

from __future__ import annotations

import json
import math
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from daap.rl.embedder import (
    EmbeddingCache,
    cosine_similarity,
    CentroidClassifier,
    TASK_SEEDS,
    EMBEDDING_MODEL_FREE,
    EMBEDDING_MODEL_CHEAP,
)


# ---------------------------------------------------------------------------
# EmbeddingCache
# ---------------------------------------------------------------------------

class TestEmbeddingCache:
    def test_miss_returns_none(self, tmp_path):
        cache = EmbeddingCache(db_path=str(tmp_path / "e.db"))
        assert cache.get("hello world", "model-x") is None

    def test_put_and_get_roundtrip(self, tmp_path):
        cache = EmbeddingCache(db_path=str(tmp_path / "e.db"))
        vec = [0.1, 0.2, 0.3]
        cache.put("hello", "model-a", vec)
        result = cache.get("hello", "model-a")
        assert result == pytest.approx(vec)

    def test_different_models_isolated(self, tmp_path):
        cache = EmbeddingCache(db_path=str(tmp_path / "e.db"))
        cache.put("text", "model-a", [1.0, 0.0])
        cache.put("text", "model-b", [0.0, 1.0])
        assert cache.get("text", "model-a") == pytest.approx([1.0, 0.0])
        assert cache.get("text", "model-b") == pytest.approx([0.0, 1.0])

    def test_different_texts_isolated(self, tmp_path):
        cache = EmbeddingCache(db_path=str(tmp_path / "e.db"))
        cache.put("alpha", "model-x", [1.0])
        cache.put("beta",  "model-x", [2.0])
        assert cache.get("alpha", "model-x") == pytest.approx([1.0])
        assert cache.get("beta",  "model-x") == pytest.approx([2.0])

    def test_put_replaces_existing(self, tmp_path):
        cache = EmbeddingCache(db_path=str(tmp_path / "e.db"))
        cache.put("text", "model-x", [1.0, 0.0])
        cache.put("text", "model-x", [0.5, 0.5])
        assert cache.get("text", "model-x") == pytest.approx([0.5, 0.5])

    def test_persists_across_instances(self, tmp_path):
        db = str(tmp_path / "e.db")
        EmbeddingCache(db_path=db).put("q", "m", [9.0, 8.0])
        assert EmbeddingCache(db_path=db).get("q", "m") == pytest.approx([9.0, 8.0])


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == pytest.approx(0.0)

    def test_result_in_minus_one_to_one(self):
        import random
        random.seed(7)
        for _ in range(20):
            a = [random.gauss(0, 1) for _ in range(16)]
            b = [random.gauss(0, 1) for _ in range(16)]
            sim = cosine_similarity(a, b)
            assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# CentroidClassifier
# ---------------------------------------------------------------------------

def _make_unit(dim: int, idx: int) -> list[float]:
    """Return a unit vector with all zeros except position idx = 1."""
    v = [0.0] * dim
    v[idx] = 1.0
    return v


class TestCentroidClassifier:
    def _make_classifier(self, tmp_path, fake_embeddings: dict[str, list[float]]) -> CentroidClassifier:
        """
        Build a CentroidClassifier with mocked get_embedding.
        fake_embeddings: {text → embedding vector}
        """
        cache = EmbeddingCache(db_path=str(tmp_path / "e.db"))

        def _fake_get_embedding(text, cache_arg=None):
            if text in fake_embeddings:
                if cache_arg is not None:
                    cache_arg.put(text, "mock", fake_embeddings[text])
                return fake_embeddings[text]
            return None

        classifier = CentroidClassifier(cache)
        with patch("daap.rl.embedder.get_embedding", side_effect=_fake_get_embedding):
            classifier._centroids = classifier._load_centroids()
        return classifier

    def test_classify_returns_nearest_task_type(self, tmp_path):
        dim = len(TASK_SEEDS)
        # Give each task type a unique orthogonal dimension
        task_types = list(TASK_SEEDS.keys())
        fake_embs: dict[str, list[float]] = {}

        for ti, task_type in enumerate(task_types):
            for seed in TASK_SEEDS[task_type]:
                fake_embs[seed] = _make_unit(dim, ti)

        # Query vector close to "lead_generation" (index 0)
        query_vec = _make_unit(dim, task_types.index("lead_generation"))
        fake_embs["find leads for me"] = query_vec

        classifier = self._make_classifier(tmp_path, fake_embs)

        with patch("daap.rl.embedder.get_embedding", return_value=query_vec):
            result = classifier.classify("find leads for me")

        assert result == "lead_generation"

    def test_classify_returns_none_when_embedding_fails(self, tmp_path):
        cache = EmbeddingCache(db_path=str(tmp_path / "e.db"))
        classifier = CentroidClassifier(cache)
        classifier._centroids = {"lead_generation": [1.0, 0.0]}

        with patch("daap.rl.embedder.get_embedding", return_value=None):
            result = classifier.classify("anything")

        assert result is None

    def test_centroid_cached_after_build(self, tmp_path):
        dim = 4
        task_types = list(TASK_SEEDS.keys())
        fake_embs = {}
        for ti, tt in enumerate(task_types):
            for seed in TASK_SEEDS[tt]:
                vec = [0.0] * dim
                vec[ti % dim] = 1.0
                fake_embs[seed] = vec

        cache = EmbeddingCache(db_path=str(tmp_path / "e.db"))
        classifier = CentroidClassifier(cache)

        with patch("daap.rl.embedder.get_embedding", side_effect=lambda t, c=None: fake_embs.get(t)):
            centroids = classifier._load_centroids()

        # At least some task types got centroids
        assert len(centroids) > 0

        # Cached centroid entries exist
        for task_type in centroids:
            cached = cache.get(f"__centroid__{task_type}", "centroid_v1")
            assert cached is not None


# ---------------------------------------------------------------------------
# Optimizer.classify_task_type with DAAP_RL_USE_EMBEDDINGS=1
# ---------------------------------------------------------------------------

class TestOptimizerEmbeddingPath:
    def test_embedding_path_used_when_env_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DAAP_RL_USE_EMBEDDINGS", "1")

        from daap.rl.optimizer import TopologyOptimizer
        opt = TopologyOptimizer(db_path=str(tmp_path / "opt.db"))

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = "research"

        with patch("daap.rl.embedder.get_centroid_classifier", return_value=mock_classifier):
            result = opt.classify_task_type("investigate the market")

        assert result == "research"
        mock_classifier.classify.assert_called_once()

    def test_falls_back_to_keywords_when_embedding_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DAAP_RL_USE_EMBEDDINGS", "1")

        from daap.rl.optimizer import TopologyOptimizer
        opt = TopologyOptimizer(db_path=str(tmp_path / "opt.db"))

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = None

        with patch("daap.rl.embedder.get_centroid_classifier", return_value=mock_classifier):
            result = opt.classify_task_type("find leads in construction")

        # Falls back to keyword — "lead" → lead_generation
        assert result == "lead_generation"

    def test_falls_back_to_keywords_when_env_not_set(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DAAP_RL_USE_EMBEDDINGS", raising=False)

        from daap.rl.optimizer import TopologyOptimizer
        opt = TopologyOptimizer(db_path=str(tmp_path / "opt.db"))

        # Even if embedding would work, should use keywords
        with patch("daap.rl.embedder.get_centroid_classifier") as mock_gc:
            result = opt.classify_task_type("write cold email sequence")

        mock_gc.assert_not_called()
        assert result == "email_outreach"

    def test_falls_back_to_keywords_on_exception(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DAAP_RL_USE_EMBEDDINGS", "1")

        from daap.rl.optimizer import TopologyOptimizer
        opt = TopologyOptimizer(db_path=str(tmp_path / "opt.db"))

        with patch("daap.rl.embedder.get_centroid_classifier", side_effect=RuntimeError("boom")):
            result = opt.classify_task_type("qualify these leads")

        assert result == "qualification"
