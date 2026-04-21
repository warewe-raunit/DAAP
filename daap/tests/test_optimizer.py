"""Tests for the contextual bandit optimizer (LinTS)."""

import numpy as np
import pytest
import tempfile
import os
import sqlite3


# --- Context Tests ---

def test_extract_context_lead_gen():
    from daap.optimizer.context import extract_context
    ctx = extract_context(
        user_prompt="Find B2B leads for my SaaS",
        node_count=4,
        has_parallel=True,
        user_total_runs=10,
    )
    assert ctx.task_category == 0  # "lead"
    assert ctx.has_parallel == 1.0
    assert ctx.user_maturity == 0.2  # 10/50


def test_extract_context_email():
    from daap.optimizer.context import extract_context
    ctx = extract_context(
        user_prompt="Draft cold emails for my prospects",
        node_count=2,
        has_parallel=False,
        user_total_runs=100,
    )
    assert ctx.task_category == 2  # "email"
    assert ctx.has_parallel == 0.0
    assert ctx.user_maturity == 1.0  # capped at 1.0


def test_context_vector_dimension():
    from daap.optimizer.context import extract_context, CONTEXT_DIM
    ctx = extract_context("Find leads", 3, True, 5)
    vec = ctx.to_vector()
    assert len(vec) == CONTEXT_DIM
    assert vec[-1] == 1.0  # bias term


def test_context_vector_one_hot():
    from daap.optimizer.context import extract_context
    ctx = extract_context("Find leads", 3, True, 5)
    vec = ctx.to_vector()
    one_hot = vec[:6]
    assert sum(one_hot) == 1.0
    assert one_hot[0] == 1.0  # "lead" category


# --- Bandit Tests ---

def test_bandit_select_arm_returns_valid():
    from daap.optimizer.bandit import LinTSBandit, ARMS
    bandit = LinTSBandit(dim=10)
    ctx = np.ones(10)
    arm = bandit.select_arm(ctx)
    assert arm in ARMS


def test_bandit_select_arm_deterministic_with_seed():
    from daap.optimizer.bandit import LinTSBandit
    bandit = LinTSBandit(dim=10)
    ctx = np.ones(10)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    assert bandit.select_arm(ctx, rng=rng1) == bandit.select_arm(ctx, rng=rng2)


def test_bandit_update_increases_pulls():
    from daap.optimizer.bandit import LinTSBandit
    bandit = LinTSBandit(dim=10)
    ctx = np.ones(10)
    bandit.update(ctx, "fast", 0.8)
    assert bandit.arm_states["fast"].n_pulls == 1
    assert bandit.arm_states["smart"].n_pulls == 0


def test_bandit_learns_high_reward_arm():
    """After many high rewards on 'smart', bandit should prefer it."""
    from daap.optimizer.bandit import LinTSBandit
    bandit = LinTSBandit(dim=10)
    ctx = np.ones(10)

    for _ in range(50):
        bandit.update(ctx, "smart", 0.9)
        bandit.update(ctx, "fast", 0.2)
        bandit.update(ctx, "powerful", 0.3)

    rng = np.random.default_rng(123)
    picks = [bandit.select_arm(ctx, rng=rng) for _ in range(100)]
    smart_count = picks.count("smart")
    assert smart_count > 70, f"Expected smart >70 picks, got {smart_count}"


def test_bandit_explores_initially():
    """With no data, all arms should be picked roughly equally."""
    from daap.optimizer.bandit import LinTSBandit
    bandit = LinTSBandit(dim=10)
    ctx = np.ones(10)

    rng = np.random.default_rng(42)
    picks = [bandit.select_arm(ctx, rng=rng) for _ in range(300)]

    for arm in ["fast", "smart", "powerful"]:
        count = picks.count(arm)
        assert count > 30, f"{arm} only picked {count}/300 times"


# --- TopologyOptimizer Tests ---

def test_optimizer_recommend_returns_all_roles():
    from daap.optimizer.bandit import TopologyOptimizer
    opt = TopologyOptimizer(dim=10)
    ctx = [0] * 6 + [0.3, 1.0, 0.2, 1.0]
    recs = opt.recommend(ctx, roles=["researcher", "writer"])
    assert "researcher" in recs
    assert "writer" in recs
    assert recs["researcher"] in ["fast", "smart", "powerful"]


def test_optimizer_update_and_persist():
    from daap.optimizer.bandit import TopologyOptimizer
    opt = TopologyOptimizer(dim=10)
    ctx = [0] * 6 + [0.3, 1.0, 0.2, 1.0]

    opt.update(ctx, {"researcher": "fast", "writer": "smart"}, reward=0.9)

    assert opt.bandits["researcher"].arm_states["fast"].n_pulls == 1
    assert opt.bandits["writer"].arm_states["smart"].n_pulls == 1


# --- Reward Tests ---

def test_reward_perfect_rating():
    from daap.optimizer.context import compute_reward
    r = compute_reward(
        user_rating=5, actual_cost_usd=0.01, budget_usd=1.0,
        latency_seconds=10, timeout_seconds=120,
    )
    assert r > 0.9


def test_reward_terrible_rating():
    from daap.optimizer.context import compute_reward
    r = compute_reward(
        user_rating=1, actual_cost_usd=0.5, budget_usd=1.0,
        latency_seconds=100, timeout_seconds=120,
    )
    assert r < 0.2


def test_reward_clamped():
    from daap.optimizer.context import compute_reward
    r = compute_reward(
        user_rating=1, actual_cost_usd=10.0, budget_usd=1.0,
        latency_seconds=500, timeout_seconds=120,
    )
    assert r >= 0.0


# --- Store Tests ---

def test_store_save_and_load():
    from daap.optimizer.store import BanditStore
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = BanditStore(db_path=db_path)

        B = np.eye(10)
        f = np.ones(10) * 0.5

        store.save_arm_state("user1", "researcher", "fast", 10, B, f, 5)
        loaded = store.load_arm_state("user1", "researcher", "fast")

        assert loaded is not None
        B_loaded, f_loaded, n = loaded
        assert n == 5
        np.testing.assert_array_almost_equal(B_loaded, B)
        np.testing.assert_array_almost_equal(f_loaded, f)


def test_store_observation_count():
    from daap.optimizer.store import BanditStore
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = BanditStore(db_path=db_path)

        store.log_observation("user1", [0] * 10, "researcher", "fast", 0.8)
        store.log_observation("user1", [0] * 10, "writer", "smart", 0.7)

        assert store.get_user_run_count("user1") == 2
        assert store.get_user_run_count("user2") == 0


def test_store_load_optimizer_round_trip():
    from daap.optimizer.store import BanditStore
    from daap.optimizer.bandit import TopologyOptimizer
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = BanditStore(db_path=db_path)

        opt = TopologyOptimizer(dim=10)
        ctx = np.ones(10)
        opt.bandits["researcher"] = opt._get_or_create_bandit("researcher")
        opt.bandits["researcher"].update(ctx, "smart", 0.9)

        store.save_optimizer("user1", opt)

        saved = store.load_optimizer("user1", dim=10)
        assert "researcher" in saved
        assert "smart" in saved["researcher"]
        _, _, n_pulls = saved["researcher"]["smart"]
        assert n_pulls == 1


# --- Integration Tests ---

def test_get_tier_recommendations_returns_valid_tiers(monkeypatch, tmp_path):
    from daap.optimizer import integration as integ
    # Use a temp db
    monkeypatch.setattr(integ, "_store", __import__("daap.optimizer.store", fromlist=["BanditStore"]).BanditStore(db_path=str(tmp_path / "test.db")))
    monkeypatch.setattr(integ, "_optimizer_cache", {})

    from daap.optimizer.integration import get_tier_recommendations
    recs = get_tier_recommendations(
        user_id="test-user",
        user_prompt="Find B2B leads",
        proposed_roles=["researcher", "writer"],
        node_count=2,
        has_parallel=False,
    )
    assert set(recs.keys()) == {"researcher", "writer"}
    for tier in recs.values():
        assert tier in ["fast", "smart", "powerful"]


def test_record_run_outcome_persists(monkeypatch, tmp_path):
    from daap.optimizer import integration as integ
    from daap.optimizer.store import BanditStore

    store = BanditStore(db_path=str(tmp_path / "test.db"))
    monkeypatch.setattr(integ, "_store", store)
    monkeypatch.setattr(integ, "_optimizer_cache", {})

    from daap.optimizer.integration import record_run_outcome
    record_run_outcome(
        user_id="test-user",
        user_prompt="Find B2B leads",
        node_configs={"researcher": "fast", "writer": "smart"},
        user_rating=4,
        actual_cost_usd=0.05,
        budget_usd=1.0,
        latency_seconds=30.0,
        timeout_seconds=120.0,
        topology_id="topo-test",
        node_count=2,
        has_parallel=False,
    )

    assert store.get_user_run_count("test-user") == 2  # one per role
    loaded = store.load_arm_state("test-user", "researcher", "fast")
    assert loaded is not None
    _, _, n_pulls = loaded
    assert n_pulls == 1


def test_bandit_store_get_profile_summary_empty(tmp_path):
    from daap.optimizer.store import BanditStore
    store = BanditStore(db_path=str(tmp_path / "opt.db"))
    assert store.get_profile_summary("alice") == []


def test_bandit_store_get_profile_summary_returns_best_arm_per_role(tmp_path):
    import numpy as np
    from daap.optimizer.store import BanditStore

    store = BanditStore(db_path=str(tmp_path / "opt.db"))
    dim = 4
    B = np.eye(dim)
    f = np.zeros(dim)

    store.save_arm_state("alice", "master", "fast", dim, B, f, n_pulls=2)
    store.save_arm_state("alice", "master", "smart", dim, B, f, n_pulls=8)
    store.save_arm_state("alice", "researcher", "fast", dim, B, f, n_pulls=3)

    summary = store.get_profile_summary("alice")
    roles = {s["role"]: s for s in summary}

    assert "master" in roles
    assert roles["master"]["best_arm"] == "smart"
    assert roles["master"]["n_pulls"] == 8

    assert "researcher" in roles
    assert roles["researcher"]["best_arm"] == "fast"
    assert roles["researcher"]["n_pulls"] == 3


def test_bandit_store_purge_expired_observations(tmp_path):
    from daap.optimizer.store import BanditStore

    store = BanditStore(db_path=str(tmp_path / "opt.db"))
    store.log_observation("alice", [0.0] * 10, "researcher", "smart", 0.8)

    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE bandit_observations SET created_at = ?",
            ("2000-01-01 00:00:00",),
        )
        conn.commit()

    purged = store.purge_expired(retention_days=1)
    assert purged == 1
    assert store.get_user_run_count("alice") == 0
