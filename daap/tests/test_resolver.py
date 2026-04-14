"""
DAAP Resolver Tests — pytest suite for spec/resolver.py
"""

import json
from pathlib import Path

import pytest

from daap.spec.schema import TopologySpec
from daap.spec.resolver import (
    MODEL_REGISTRY,
    TOOL_REGISTRY,
    CONSOLIDATION_REGISTRY,
    ResolvedTopology,
    ResolutionError,
    resolve_topology,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> TopologySpec:
    raw = json.loads((FIXTURES / name).read_text())
    return TopologySpec.model_validate(raw)


def test_resolve_sales_outreach():
    topo = load_fixture("sales_outreach_topology.json")
    result = resolve_topology(topo)
    assert isinstance(result, ResolvedTopology), f"Expected ResolvedTopology, got errors: {result}"

    node_map = {n.node_id: n for n in result.nodes}

    # Model tiers resolved correctly (fixture has anthropic operator_config)
    assert "haiku" in node_map["lead_researcher"].concrete_model_id
    assert "sonnet" in node_map["lead_evaluator"].concrete_model_id

    # Tools resolved
    researcher_tools = node_map["lead_researcher"].concrete_tool_ids
    assert any("WebSearch" in t for t in researcher_tools)

    # Linear execution order for sequential topology
    assert result.execution_order == [
        ["lead_researcher"],
        ["lead_evaluator"],
        ["personalization_researcher"],
        ["email_drafter"],
    ]


def test_resolve_parallel_topology():
    topo = load_fixture("parallel_branches_topology.json")
    result = resolve_topology(topo)
    assert isinstance(result, ResolvedTopology)

    # Step 1: input_parser alone, Step 2: both researchers in parallel, Step 3: synthesizer
    assert result.execution_order[0] == ["input_parser"]
    assert set(result.execution_order[1]) == {"researcher_a", "researcher_b"}
    assert result.execution_order[2] == ["synthesizer"]


def test_mcp_tool_passthrough():
    topo = load_fixture("sales_outreach_topology.json")
    result = resolve_topology(topo)
    assert isinstance(result, ResolvedTopology)

    researcher = next(n for n in result.nodes if n.node_id == "lead_researcher")
    assert "mcp://linkedin" in researcher.concrete_tool_ids


def test_unknown_model_tier_fails(monkeypatch):
    import daap.spec.resolver as resolver_module
    original = dict(resolver_module.MODEL_REGISTRY)
    # Remove "fast" from registry and clear operator_config so it must fall back
    monkeypatch.setitem(resolver_module.MODEL_REGISTRY, "fast", None)
    # None return from _resolve_model should trigger ResolutionError

    topo = load_fixture("sales_outreach_topology.json")
    # Remove operator_config so it uses MODEL_REGISTRY
    topo_data = json.loads((FIXTURES / "sales_outreach_topology.json").read_text())
    topo_data.pop("operator_config", None)
    topo = TopologySpec.model_validate(topo_data)

    result = resolve_topology(topo)
    assert isinstance(result, list), "Expected list of ResolutionErrors"
    assert any(e.field == "model_tier" for e in result)


def test_unknown_tool_fails():
    raw = json.loads((FIXTURES / "sales_outreach_topology.json").read_text())
    # Replace a tool with an unknown one (not mcp://, not in registry)
    raw["nodes"][0]["tools"] = [{"name": "FakeToolXYZ"}]
    raw["nodes"][0]["agent_mode"] = "react"
    topo = TopologySpec.model_validate(raw)
    result = resolve_topology(topo)
    assert isinstance(result, list)
    assert any(e.field == "tools" and "FakeToolXYZ" in e.abstract_name for e in result)


def test_consolidation_resolved():
    topo = load_fixture("sales_outreach_topology.json")
    result = resolve_topology(topo)
    assert isinstance(result, ResolvedTopology)

    researcher = next(n for n in result.nodes if n.node_id == "lead_researcher")
    assert researcher.consolidation_func == CONSOLIDATION_REGISTRY["deduplicate"]

    evaluator = next(n for n in result.nodes if n.node_id == "lead_evaluator")
    assert evaluator.consolidation_func == CONSOLIDATION_REGISTRY["rank"]


def test_execution_order_complex_dag():
    topo = load_fixture("diamond_topology.json")
    result = resolve_topology(topo)
    assert isinstance(result, ResolvedTopology)

    assert result.execution_order[0] == ["node_a"]
    assert set(result.execution_order[1]) == {"node_b", "node_c"}
    assert result.execution_order[2] == ["node_d"]
