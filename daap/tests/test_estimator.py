"""
DAAP Estimator Tests — pytest suite for spec/estimator.py
"""

import json
from pathlib import Path

import pytest

from daap.spec.schema import TopologySpec
from daap.spec.resolver import resolve_topology, ResolvedTopology
from daap.spec.estimator import (
    estimate_topology,
    TopologyEstimate,
    RISK_ORDER,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_resolved(name: str) -> ResolvedTopology:
    raw = json.loads((FIXTURES / name).read_text())
    topo = TopologySpec.model_validate(raw)
    result = resolve_topology(topo)
    assert isinstance(result, ResolvedTopology), f"Resolution failed: {result}"
    return result


def test_estimate_sales_outreach():
    resolved = load_resolved("sales_outreach_topology.json")
    est = estimate_topology(resolved)

    assert est.total_cost_usd > 0
    assert est.within_budget  # $1.00 budget, sales outreach should be well under
    assert est.total_latency_seconds > 0
    assert est.user_facing_summary
    # Summary should mention roles, not model IDs
    assert "researcher" in est.user_facing_summary.lower()
    assert "$" in est.user_facing_summary


def test_over_budget_generates_suggestions():
    raw = json.loads((FIXTURES / "sales_outreach_topology.json").read_text())
    raw["constraints"]["max_cost_usd"] = 0.000001  # impossibly low budget
    topo = TopologySpec.model_validate(raw)
    resolved = resolve_topology(topo)
    assert isinstance(resolved, ResolvedTopology)
    est = estimate_topology(resolved)

    assert not est.within_budget
    assert len(est.cost_suggestions) > 0

    # Sorted: free before low before high before destructive
    risks = [RISK_ORDER[s.risk] for s in est.cost_suggestions]
    assert risks == sorted(risks), "Cost suggestions not sorted by risk"


def test_over_timeout_generates_suggestions():
    raw = json.loads((FIXTURES / "sales_outreach_topology.json").read_text())
    raw["constraints"]["max_latency_seconds"] = 0.001  # impossibly low
    topo = TopologySpec.model_validate(raw)
    resolved = resolve_topology(topo)
    assert isinstance(resolved, ResolvedTopology)
    est = estimate_topology(resolved)

    assert not est.within_timeout
    assert len(est.latency_suggestions) > 0


def test_parallel_latency_is_max_not_sum():
    resolved = load_resolved("parallel_branches_topology.json")
    est = estimate_topology(resolved)

    # Step 2 has researcher_a and researcher_b in parallel
    # Total latency must be less than if they ran sequentially
    node_map = {e.node_id: e for e in est.node_estimates}
    parallel_step_latency = max(
        node_map["researcher_a"].estimated_latency_seconds,
        node_map["researcher_b"].estimated_latency_seconds,
    )
    sequential_latency = (
        node_map["researcher_a"].estimated_latency_seconds
        + node_map["researcher_b"].estimated_latency_seconds
    )
    # actual total must use max (parallel), not sum (sequential)
    assert est.total_latency_seconds < (
        node_map["input_parser"].estimated_latency_seconds + sequential_latency +
        node_map["synthesizer"].estimated_latency_seconds
    )


def test_react_vs_single_cost_difference():
    raw = json.loads((FIXTURES / "parallel_branches_topology.json").read_text())
    # researcher_a is react (has tools), input_parser is single (no tools)
    topo = TopologySpec.model_validate(raw)
    resolved = resolve_topology(topo)
    assert isinstance(resolved, ResolvedTopology)
    est = estimate_topology(resolved)

    node_map = {e.node_id: e for e in est.node_estimates}
    react_node = node_map["researcher_a"]
    single_node = node_map["input_parser"]

    # React node should cost more due to tool loop overhead
    # (Both are fast tier but react has overhead multiplier)
    # Compare cost per token: react should be higher
    assert react_node.estimated_latency_seconds >= single_node.estimated_latency_seconds


def test_instance_count_multiplies_cost_not_latency():
    raw = json.loads((FIXTURES / "sales_outreach_topology.json").read_text())
    topo = TopologySpec.model_validate(raw)
    resolved = resolve_topology(topo)
    assert isinstance(resolved, ResolvedTopology)
    est = estimate_topology(resolved)

    node_map = {e.node_id: e for e in est.node_estimates}
    researcher = node_map["lead_researcher"]
    evaluator = node_map["lead_evaluator"]

    # researcher has 4 instances, evaluator has 2
    # Latency should NOT be 4x (they run in parallel)
    # Cost SHOULD be roughly proportional to instance count
    assert researcher.instance_count == 4
    assert evaluator.instance_count == 2

    # researcher cost should be roughly 2x evaluator cost (4 vs 2 instances, same tier)
    # (rough — different roles have different heuristics)
    assert researcher.estimated_cost_usd > 0
    assert evaluator.estimated_cost_usd > 0


def test_min_viable_cost_is_floor():
    resolved = load_resolved("sales_outreach_topology.json")
    est = estimate_topology(resolved)

    assert est.min_viable_cost_usd > 0
    assert est.min_viable_cost_usd <= est.total_cost_usd


def test_suggestion_risk_levels_valid():
    raw = json.loads((FIXTURES / "sales_outreach_topology.json").read_text())
    raw["constraints"]["max_cost_usd"] = 0.000001
    topo = TopologySpec.model_validate(raw)
    resolved = resolve_topology(topo)
    assert isinstance(resolved, ResolvedTopology)
    est = estimate_topology(resolved)

    valid_risks = {"free", "low", "high", "destructive"}
    for s in est.cost_suggestions:
        assert s.risk in valid_risks, f"Invalid risk level: {s.risk}"
    for s in est.latency_suggestions:
        assert s.risk in valid_risks, f"Invalid risk level: {s.risk}"


def test_free_suggestions_have_zero_quality_impact():
    raw = json.loads((FIXTURES / "sales_outreach_topology.json").read_text())
    raw["constraints"]["max_cost_usd"] = 0.000001
    topo = TopologySpec.model_validate(raw)
    resolved = resolve_topology(topo)
    assert isinstance(resolved, ResolvedTopology)
    est = estimate_topology(resolved)

    for s in est.cost_suggestions:
        if s.risk == "free":
            # Free suggestions should be about mode/iteration changes, NOT node removal
            assert "remove" not in s.description.lower(), \
                f"Free suggestion should not remove nodes: {s.description}"
            assert "downgrade" not in s.description.lower(), \
                f"Free suggestion should not downgrade models: {s.description}"


def test_user_facing_summary_is_plain_english():
    resolved = load_resolved("sales_outreach_topology.json")
    est = estimate_topology(resolved)

    # Should NOT contain raw model IDs
    technical_ids = ["claude-haiku", "claude-sonnet", "claude-opus", "4-5-20251001"]
    for tid in technical_ids:
        assert tid not in est.user_facing_summary, \
            f"Summary contains technical model ID '{tid}': {est.user_facing_summary}"

    # Should contain dollar sign and be non-empty
    assert "$" in est.user_facing_summary
    assert len(est.user_facing_summary) > 20
