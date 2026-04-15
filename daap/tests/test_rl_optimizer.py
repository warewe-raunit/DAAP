"""Integration-style unit tests for TopologyOptimizer behavior."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from daap.rl.optimizer import TopologyOptimizer


def _optimizer(tmp_path) -> TopologyOptimizer:
    return TopologyOptimizer(db_path=str(tmp_path / "optimizer.db"))


def _topology(topology_id: str = "topo-abc12345") -> dict:
    return {
        "topology_id": topology_id,
        "user_prompt": "Find leads and draft outreach emails",
        "nodes": [
            {
                "node_id": "researcher",
                "role": "Research Leads",
                "agent_mode": "react",
                "tools": [{"name": "WebSearch"}],
                "model_tier": "smart",
                "instance_config": {"parallel_instances": 3, "consolidation": "merge"},
            },
            {
                "node_id": "qualifier",
                "role": "Lead Evaluator",
                "agent_mode": "single",
                "tools": [],
                "model_tier": "smart",
                "instance_config": {"parallel_instances": 1},
            },
            {
                "node_id": "writer",
                "role": "Email Writer",
                "agent_mode": "single",
                "tools": [],
                "model_tier": "smart",
                "instance_config": {"parallel_instances": 2, "consolidation": "merge"},
            },
        ],
    }


def _result_payload(topology_id: str) -> dict:
    return {
        "topology_id": topology_id,
        "success": True,
        "latency_seconds": 20.0,
        "total_input_tokens": 1400,
        "total_output_tokens": 900,
        "final_output": "Completed run",
        "error": None,
    }


def test_classify_task_type_lead_generation(tmp_path):
    opt = _optimizer(tmp_path)
    assert opt.classify_task_type("Find 50 leads in construction") == "lead_generation"


def test_classify_task_type_email_outreach(tmp_path):
    opt = _optimizer(tmp_path)
    assert opt.classify_task_type("Draft cold email follow-up sequence") == "email_outreach"


def test_classify_task_type_general_fallback(tmp_path):
    opt = _optimizer(tmp_path)
    assert opt.classify_task_type("Hello there") == "general"


def test_fingerprint_sorts_roles_stably_across_node_order(tmp_path):
    opt = _optimizer(tmp_path)

    topo_a = _topology("topo-a")
    topo_b = _topology("topo-b")
    topo_b["nodes"] = list(reversed(topo_b["nodes"]))

    fp_a = opt.build_task_fingerprint("lead_generation", topo_a)
    fp_b = opt.build_task_fingerprint("lead_generation", topo_b)

    assert fp_a == fp_b


def test_recommend_overrides_returns_new_dict_and_does_not_mutate_original(tmp_path, monkeypatch):
    opt = _optimizer(tmp_path)
    bandit = opt._get_bandit()
    monkeypatch.setattr(bandit, "select_arm", lambda _task: "balanced_2")

    topo = _topology()
    original = copy.deepcopy(topo)

    updated = opt.recommend_overrides(topo, task_type="lead_generation")

    assert updated is not topo
    assert topo == original
    assert isinstance(updated, dict)


def test_recommend_overrides_applies_valid_model_tiers_and_tracks_pending(tmp_path, monkeypatch):
    opt = _optimizer(tmp_path)
    bandit = opt._get_bandit()
    monkeypatch.setattr(bandit, "select_arm", lambda _task: "quality_3")

    topo = _topology("topo-track")
    updated = opt.recommend_overrides(topo, task_type="lead_generation")

    for node in updated["nodes"]:
        assert node["model_tier"] in {"fast", "smart", "powerful"}

    pending = opt._pending["topo-track"]
    assert pending["arm_id"] == "quality_3"
    assert pending["task_type"] == "lead_generation"


def test_recommend_overrides_respects_operator_override_model_map(tmp_path, monkeypatch):
    opt = _optimizer(tmp_path)
    bandit = opt._get_bandit()
    monkeypatch.setattr(bandit, "select_arm", lambda _task: "budget_3")

    topo = _topology("topo-op")
    topo["nodes"][1]["model_tier"] = "powerful"
    topo["nodes"][1]["operator_override"] = {
        "provider": "openrouter",
        "model_map": {"smart": "deepseek/deepseek-v3.2"},
    }

    updated = opt.recommend_overrides(topo, task_type="qualification")

    assert updated["nodes"][1]["model_tier"] == "powerful"


def test_recommend_overrides_only_changes_parallel_nodes_with_parallel_gt_one(tmp_path, monkeypatch):
    opt = _optimizer(tmp_path)
    bandit = opt._get_bandit()
    monkeypatch.setattr(bandit, "select_arm", lambda _task: "budget_2")

    topo = _topology("topo-par")
    updated = opt.recommend_overrides(topo, task_type="lead_generation")

    assert updated["nodes"][1]["instance_config"]["parallel_instances"] == 1
    assert updated["nodes"][0]["instance_config"]["parallel_instances"] == 2
    assert updated["nodes"][2]["instance_config"]["parallel_instances"] == 2


def test_recommend_overrides_graceful_on_internal_error(tmp_path, monkeypatch):
    opt = _optimizer(tmp_path)
    topo = _topology("topo-fail")

    def _boom(_task):
        raise RuntimeError("boom")

    monkeypatch.setattr(opt._get_bandit(), "select_arm", _boom)
    out = opt.recommend_overrides(topo, task_type="general")

    assert out == topo


def test_record_outcome_updates_bandit_and_clears_pending(tmp_path, monkeypatch):
    opt = _optimizer(tmp_path)
    bandit = opt._get_bandit()
    monkeypatch.setattr(bandit, "select_arm", lambda _task: "quality_1")

    topo = _topology("topo-outcome")
    updated = opt.recommend_overrides(topo, task_type="lead_generation")

    before = bandit.get_arm_stats("lead_generation")["quality_1"]["n_pulls"]
    opt.record_outcome(updated, _result_payload("topo-outcome"), task_type="lead_generation")
    after = bandit.get_arm_stats("lead_generation")["quality_1"]["n_pulls"]

    assert after == before + 1
    assert "topo-outcome" not in opt._pending


def test_record_outcome_graceful_without_pending(tmp_path):
    opt = _optimizer(tmp_path)
    bandit = opt._get_bandit()

    opt.record_outcome(_topology("topo-missing"), _result_payload("topo-missing"), task_type="general")

    stats = bandit.get_arm_stats("general")
    assert all(v["n_pulls"] == 0 for v in stats.values())


def test_record_outcome_accepts_execution_result_object(tmp_path, monkeypatch):
    opt = _optimizer(tmp_path)
    bandit = opt._get_bandit()
    monkeypatch.setattr(bandit, "select_arm", lambda _task: "balanced_1")

    topo = _topology("topo-object")
    updated = opt.recommend_overrides(topo, task_type="research")

    result_obj = SimpleNamespace(
        topology_id="topo-object",
        success=True,
        error=None,
        final_output="ok",
        total_latency_seconds=11.2,
        total_input_tokens=320,
        total_output_tokens=140,
    )

    before = bandit.get_arm_stats("research")["balanced_1"]["n_pulls"]
    opt.record_outcome(updated, result_obj, task_type="research")
    after = bandit.get_arm_stats("research")["balanced_1"]["n_pulls"]

    assert after == before + 1


def test_record_rating_stores_pending_rating(tmp_path):
    opt = _optimizer(tmp_path)
    opt.record_rating("topo-rate-1", 4)
    assert opt._pending_ratings["topo-rate-1"] == 4


def test_record_rating_consumed_by_record_outcome(tmp_path, monkeypatch):
    opt = _optimizer(tmp_path)
    bandit = opt._get_bandit()
    monkeypatch.setattr(bandit, "select_arm", lambda _task: "balanced_3")

    topo = _topology("topo-rate-2")
    updated = opt.recommend_overrides(topo, task_type="email_outreach")

    opt.record_rating("topo-rate-2", 5)
    opt.record_outcome(updated, _result_payload("topo-rate-2"), task_type="email_outreach")

    assert "topo-rate-2" not in opt._pending_ratings

    fingerprint = opt.build_task_fingerprint("email_outreach", updated)
    exps = bandit.get_experiences(fingerprint, limit=1)
    assert exps[0]["outcome_json"]["rating"] == 5
    assert exps[0]["outcome_json"]["rating_applied"] is True


def test_record_rating_graceful_on_missing_topology_id(tmp_path):
    opt = _optimizer(tmp_path)
    opt.record_rating("", 3)
    assert opt._pending_ratings == {}


def test_retroactive_rating_updates_existing_experience_without_extra_pull(tmp_path, monkeypatch):
    opt = _optimizer(tmp_path)
    bandit = opt._get_bandit()
    monkeypatch.setattr(bandit, "select_arm", lambda _task: "budget_1")

    topo = _topology("topo-retro")
    updated = opt.recommend_overrides(topo, task_type="general")
    opt.record_outcome(updated, _result_payload("topo-retro"), task_type="general")

    exp_id = opt._topology_to_exp["topo-retro"]
    before_exp = bandit.get_experience(exp_id)
    before_stats = bandit.get_arm_stats("general")["budget_1"]

    opt.record_rating("topo-retro", 1)

    after_exp = bandit.get_experience(exp_id)
    after_stats = bandit.get_arm_stats("general")["budget_1"]

    assert after_exp is not None
    assert before_exp is not None
    assert after_exp["outcome_json"]["rating_applied"] is True
    assert after_stats["n_pulls"] == before_stats["n_pulls"]
    assert after_exp["reward"] <= before_exp["reward"]


def test_get_rl_prompt_section_returns_none_when_insufficient_experience(tmp_path):
    opt = _optimizer(tmp_path)
    section = opt.get_rl_prompt_section("lead_generation", _topology("topo-ctx"))
    assert section is None
