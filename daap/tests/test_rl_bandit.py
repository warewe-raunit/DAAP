"""Unit tests for ThompsonBandit persistence and update behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from daap.rl.bandit import ARM_IDS, ThompsonBandit


def _make_bandit(tmp_path) -> ThompsonBandit:
    db_path = tmp_path / "optimizer.db"
    return ThompsonBandit(db_path=str(db_path))


def test_init_creates_db_file(tmp_path):
    db_path = tmp_path / "optimizer.db"
    ThompsonBandit(db_path=str(db_path))
    assert Path(db_path).exists()


def test_cold_start_is_beta_one_one(tmp_path):
    bandit = _make_bandit(tmp_path)
    stats = bandit.get_arm_stats("lead_generation")

    assert len(stats) == 9
    for arm_id in ARM_IDS:
        assert stats[arm_id]["alpha"] == 1.0
        assert stats[arm_id]["beta"] == 1.0
        assert stats[arm_id]["n_pulls"] == 0


def test_first_select_returns_valid_arm(tmp_path):
    bandit = _make_bandit(tmp_path)
    arm = bandit.select_arm("research")
    assert arm in ARM_IDS


def test_thompson_sampling_is_seed_deterministic(tmp_path):
    bandit = _make_bandit(tmp_path)

    np.random.seed(11)
    arm_a = bandit.select_arm("qualification")

    np.random.seed(11)
    arm_b = bandit.select_arm("qualification")

    assert arm_a == arm_b


def test_high_reward_arm_dominates_with_many_updates(tmp_path):
    bandit = _make_bandit(tmp_path)
    task_type = "qualification"

    for _ in range(80):
        bandit.update(task_type, "quality_2", 1.0)

    np.random.seed(42)
    counts = {arm_id: 0 for arm_id in ARM_IDS}
    for _ in range(100):
        counts[bandit.select_arm(task_type)] += 1

    assert counts["quality_2"] >= 70


def test_update_increments_pulls_and_alpha_on_high_reward(tmp_path):
    bandit = _make_bandit(tmp_path)
    task_type = "email_outreach"

    before = bandit.get_arm_stats(task_type)["balanced_1"]
    bandit.update(task_type, "balanced_1", 0.9)
    after = bandit.get_arm_stats(task_type)["balanced_1"]

    assert after["n_pulls"] == before["n_pulls"] + 1
    assert after["alpha"] > before["alpha"]


def test_update_increments_beta_on_low_reward(tmp_path):
    bandit = _make_bandit(tmp_path)
    task_type = "email_outreach"

    before = bandit.get_arm_stats(task_type)["balanced_2"]
    bandit.update(task_type, "balanced_2", 0.1)
    after = bandit.get_arm_stats(task_type)["balanced_2"]

    assert after["beta"] > before["beta"]


def test_update_clamps_out_of_bounds_reward(tmp_path):
    bandit = _make_bandit(tmp_path)
    task_type = "general"

    bandit.update(task_type, "budget_1", 9.0)
    high = bandit.get_arm_stats(task_type)["budget_1"]
    assert high["alpha"] == 2.0
    assert high["beta"] == 1.0

    bandit.update(task_type, "budget_1", -4.0)
    low = bandit.get_arm_stats(task_type)["budget_1"]
    assert low["alpha"] == 2.0
    assert low["beta"] == 2.0


def test_log_experience_returns_exp_id_and_can_be_fetched(tmp_path):
    bandit = _make_bandit(tmp_path)

    exp_id = bandit.log_experience(
        task_fingerprint="research:researcher",
        arm_id="budget_2",
        reward=0.7,
        topology_json={"topology_id": "topo-1"},
        outcome_json={"topology_id": "topo-1", "success": True},
    )

    exps = bandit.get_experiences("research:researcher", limit=5)
    assert len(exps) == 1
    assert exps[0]["exp_id"] == exp_id
    assert exps[0]["outcome_json"]["topology_id"] == "topo-1"


def test_get_experiences_ordered_by_recency(tmp_path):
    bandit = _make_bandit(tmp_path)
    fp = "lead_generation:researcher|writer"

    first_id = bandit.log_experience(fp, "budget_1", 0.2, {}, {"topology_id": "topo-a"})
    second_id = bandit.log_experience(fp, "budget_2", 0.9, {}, {"topology_id": "topo-b"})

    rows = bandit.get_experiences(fp, limit=10)
    assert rows[0]["exp_id"] == second_id
    assert rows[1]["exp_id"] == first_id


def test_persistence_survives_across_instances(tmp_path):
    db_path = tmp_path / "optimizer.db"

    b1 = ThompsonBandit(db_path=str(db_path))
    b1.update("lead_generation", "quality_3", 1.0)

    b2 = ThompsonBandit(db_path=str(db_path))
    stats = b2.get_arm_stats("lead_generation")

    assert stats["quality_3"]["n_pulls"] == 1
    assert stats["quality_3"]["alpha"] == 2.0
