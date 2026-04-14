"""
DAAP Validator Tests — pytest suite for spec/validator.py
"""

import json
from pathlib import Path

import pytest

from daap.spec.schema import TopologySpec
from daap.spec.validator import (
    ValidationError,
    ValidationResult,
    validate_topology,
    DEFAULT_AVAILABLE_TOOLS,
    DEFAULT_AVAILABLE_MODELS,
)

FIXTURES = Path(__file__).parent / "fixtures"

AVAILABLE_TOOLS = DEFAULT_AVAILABLE_TOOLS | {"mcp://linkedin"}
AVAILABLE_MODELS = DEFAULT_AVAILABLE_MODELS


def load_fixture(name: str) -> TopologySpec:
    raw = json.loads((FIXTURES / name).read_text())
    return TopologySpec.model_validate(raw)


def minimal_node(node_id: str, with_tools: bool = False) -> dict:
    node = {
        "node_id": node_id,
        "role": "Test Node",
        "model_tier": "fast",
        "system_prompt": "You are a test agent.",
        "outputs": [{"data_key": "result", "data_type": "string", "description": "output"}],
        "instance_config": {"parallel_instances": 1},
        "agent_mode": "single",
        "tools": [],
    }
    if with_tools:
        node["tools"] = [{"name": "WebSearch"}]
        node["agent_mode"] = "react"
    return node


def minimal_topology(nodes: list[dict], edges: list[dict] | None = None) -> TopologySpec:
    return TopologySpec.model_validate({
        "topology_id": "test-001",
        "version": 1,
        "created_at": "2026-04-13T10:00:00Z",
        "user_prompt": "Test.",
        "nodes": nodes,
        "edges": edges or [],
    })


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------

def test_valid_topology_passes():
    topo = load_fixture("sales_outreach_topology.json")
    result = validate_topology(topo, AVAILABLE_TOOLS, AVAILABLE_MODELS)
    assert result.is_valid
    assert len(result.errors) == 0


def test_cycle_detection():
    topo = load_fixture("cyclic_topology.json")
    result = validate_topology(topo, AVAILABLE_TOOLS, AVAILABLE_MODELS)
    assert not result.is_valid
    cycle_errors = [e for e in result.errors if e.category == "structural" and "cycle" in e.message.lower()]
    assert cycle_errors, f"Expected cycle error, got: {result.error_summary}"
    assert any("node_a" in e.message or "node_b" in e.message or "node_c" in e.message
               for e in cycle_errors)


def test_orphan_node_detected():
    nodes = [
        minimal_node("node_a"),
        minimal_node("node_b"),
        minimal_node("orphan"),
    ]
    edges = [{"source_node_id": "node_a", "target_node_id": "node_b",
              "data_key": "result", "description": ""}]
    # Fix I/O: node_b needs to declare the input
    nodes[1]["inputs"] = [{"data_key": "result", "data_type": "string", "description": "input"}]
    topo = minimal_topology(nodes, edges)
    result = validate_topology(topo, AVAILABLE_TOOLS, AVAILABLE_MODELS)
    orphan_errors = [e for e in result.errors if e.category == "structural" and "orphan" in e.message.lower()]
    assert orphan_errors, f"Expected orphan error, got: {result.error_summary}"


def test_exceeds_max_nodes():
    nodes = [minimal_node(f"node_{i}") for i in range(3)]
    topo = TopologySpec.model_validate({
        "topology_id": "test-001",
        "version": 1,
        "created_at": "2026-04-13T10:00:00Z",
        "user_prompt": "Test.",
        "nodes": nodes,
        "edges": [],
        "constraints": {"max_nodes": 2, "max_cost_usd": 1.0, "max_latency_seconds": 120.0,
                        "max_total_instances": 20, "max_retries_per_node": 2},
    })
    result = validate_topology(topo, AVAILABLE_TOOLS, AVAILABLE_MODELS)
    node_errors = [e for e in result.errors if e.category == "structural" and "exceeds limit" in e.message]
    assert node_errors


def test_exceeds_total_instances():
    nodes = [
        {**minimal_node("node_a"), "instance_config": {"parallel_instances": 10, "consolidation": "merge"}},
        {**minimal_node("node_b"), "instance_config": {"parallel_instances": 10, "consolidation": "merge"}},
        {**minimal_node("node_c"), "instance_config": {"parallel_instances": 10, "consolidation": "merge"}},
    ]
    topo = TopologySpec.model_validate({
        "topology_id": "test-001",
        "version": 1,
        "created_at": "2026-04-13T10:00:00Z",
        "user_prompt": "Test.",
        "nodes": nodes,
        "edges": [],
        "constraints": {"max_nodes": 10, "max_cost_usd": 1.0, "max_latency_seconds": 120.0,
                        "max_total_instances": 5, "max_retries_per_node": 2},
    })
    result = validate_topology(topo, AVAILABLE_TOOLS, AVAILABLE_MODELS)
    inst_errors = [e for e in result.errors if "instances" in e.message and "exceeds" in e.message]
    assert inst_errors


# ---------------------------------------------------------------------------
# I/O compatibility
# ---------------------------------------------------------------------------

def test_edge_references_nonexistent_output():
    nodes = [
        {**minimal_node("node_a"), "outputs": [{"data_key": "real_output", "data_type": "string", "description": ""}]},
        {**minimal_node("node_b"), "inputs": [{"data_key": "missing_key", "data_type": "string", "description": ""}]},
    ]
    edges = [{"source_node_id": "node_a", "target_node_id": "node_b",
              "data_key": "missing_key", "description": ""}]
    topo = minimal_topology(nodes, edges)
    result = validate_topology(topo, AVAILABLE_TOOLS, AVAILABLE_MODELS)
    io_errors = [e for e in result.errors if e.category == "io" and "doesn't produce" in e.message]
    assert io_errors


def test_edge_references_nonexistent_input():
    nodes = [
        {**minimal_node("node_a"), "outputs": [{"data_key": "real_output", "data_type": "string", "description": ""}]},
        {**minimal_node("node_b"), "inputs": []},
    ]
    edges = [{"source_node_id": "node_a", "target_node_id": "node_b",
              "data_key": "real_output", "description": ""}]
    topo = minimal_topology(nodes, edges)
    result = validate_topology(topo, AVAILABLE_TOOLS, AVAILABLE_MODELS)
    io_errors = [e for e in result.errors if e.category == "io" and "doesn't consume" in e.message]
    assert io_errors


def test_type_mismatch_on_edge():
    nodes = [
        {**minimal_node("node_a"), "outputs": [{"data_key": "data", "data_type": "list[Lead]", "description": ""}]},
        {**minimal_node("node_b"), "inputs": [{"data_key": "data", "data_type": "string", "description": ""}]},
    ]
    edges = [{"source_node_id": "node_a", "target_node_id": "node_b",
              "data_key": "data", "description": ""}]
    topo = minimal_topology(nodes, edges)
    result = validate_topology(topo, AVAILABLE_TOOLS, AVAILABLE_MODELS)
    type_errors = [e for e in result.errors if e.category == "io" and "Type mismatch" in e.message]
    assert type_errors


def test_unsatisfied_input():
    nodes = [
        minimal_node("node_a"),
        {**minimal_node("node_b"), "inputs": [{"data_key": "required_key", "data_type": "string", "description": ""}]},
    ]
    edges = [{"source_node_id": "node_a", "target_node_id": "node_b",
              "data_key": "result", "description": ""}]
    topo = minimal_topology(nodes, edges)
    result = validate_topology(topo, AVAILABLE_TOOLS, AVAILABLE_MODELS)
    input_errors = [e for e in result.errors if e.category == "io" and "declares input" in e.message]
    assert input_errors


# ---------------------------------------------------------------------------
# Tool validation
# ---------------------------------------------------------------------------

def test_unknown_tool_fails():
    nodes = [{**minimal_node("node_a"), "tools": [{"name": "FakeToolThatDoesNotExist"}],
              "agent_mode": "react"}]
    topo = minimal_topology(nodes)
    result = validate_topology(topo, AVAILABLE_TOOLS, AVAILABLE_MODELS)
    tool_errors = [e for e in result.errors if e.category == "tool" and "not available" in e.message]
    assert tool_errors
    assert "FakeToolThatDoesNotExist" in tool_errors[0].message


def test_mcp_tool_valid_format():
    tools_with_linkedin = AVAILABLE_TOOLS | {"mcp://linkedin"}
    nodes = [{**minimal_node("node_a"), "tools": [{"name": "mcp://linkedin"}], "agent_mode": "react"}]
    topo = minimal_topology(nodes)
    result = validate_topology(topo, tools_with_linkedin, AVAILABLE_MODELS)
    tool_errors = [e for e in result.errors if e.category == "tool" and "mcp://linkedin" in e.message]
    assert not tool_errors, f"Valid MCP tool should not fail: {result.error_summary}"


def test_mcp_tool_malformed():
    nodes = [{**minimal_node("node_a"), "tools": [{"name": "mcp://"}], "agent_mode": "react"}]
    topo = minimal_topology(nodes)
    result = validate_topology(topo, AVAILABLE_TOOLS | {"mcp://"}, AVAILABLE_MODELS)
    tool_errors = [e for e in result.errors if e.category == "tool" and "Malformed MCP" in e.message]
    assert tool_errors


def test_duplicate_tools_in_node():
    nodes = [{**minimal_node("node_a"),
              "tools": [{"name": "WebSearch"}, {"name": "WebSearch"}],
              "agent_mode": "react"}]
    topo = minimal_topology(nodes)
    result = validate_topology(topo, AVAILABLE_TOOLS, AVAILABLE_MODELS)
    dup_errors = [e for e in result.errors if e.category == "tool" and "times" in e.message]
    assert dup_errors


def test_react_mode_no_tools_caught_by_validator():
    # Bypass Pydantic by using SINGLE in schema, then check validator separately
    # The schema itself rejects react+no tools, so test validator's category 4 check
    # by using a node that arrives with empty tools and react mode (mock post-parse)
    from daap.spec.schema import AgentMode, HandoffMode, InstanceConfig, IOSchema, ModelTier, NodeSpec
    from daap.spec.schema import TopologySpec, ConstraintSpec

    # Build a TopologySpec manually, bypassing the NodeSpec validator
    node = object.__new__(NodeSpec)
    node.__dict__.update({
        "node_id": "test_node",
        "role": "Test",
        "model_tier": ModelTier.FAST,
        "system_prompt": "test",
        "tools": [],
        "inputs": [],
        "outputs": [IOSchema(data_key="result", data_type="string", description="")],
        "instance_config": InstanceConfig(parallel_instances=1),
        "handoff_mode": HandoffMode.NEVER,
        "operator_override": None,
        "agent_mode": AgentMode.REACT,
        "max_react_iterations": 10,
    })
    topo = object.__new__(TopologySpec)
    topo.__dict__.update({
        "topology_id": "t1",
        "version": 1,
        "created_at": "2026-04-13T10:00:00Z",
        "user_prompt": "test",
        "nodes": [node],
        "edges": [],
        "constraints": ConstraintSpec(),
        "operator_config": None,
        "metadata": {},
    })
    result = validate_topology(topo, AVAILABLE_TOOLS, AVAILABLE_MODELS)
    tool_errors = [e for e in result.errors if e.category == "tool" and "ReAct mode" in e.message]
    assert tool_errors


# ---------------------------------------------------------------------------
# Pattern validation
# ---------------------------------------------------------------------------

def test_coordinator_with_outgoing_edges_warns():
    nodes = [
        {**minimal_node("node_a"), "handoff_mode": "always",
         "outputs": [{"data_key": "result", "data_type": "string", "description": ""}]},
        {**minimal_node("node_b"), "inputs": [{"data_key": "result", "data_type": "string", "description": ""}]},
    ]
    edges = [{"source_node_id": "node_a", "target_node_id": "node_b",
              "data_key": "result", "description": ""}]
    topo = minimal_topology(nodes, edges)
    result = validate_topology(topo, AVAILABLE_TOOLS, AVAILABLE_MODELS)
    pattern_errors = [e for e in result.errors if e.category == "pattern" and "ALWAYS" in e.message]
    assert pattern_errors


def test_parallel_node_missing_consolidation():
    # Already caught by Pydantic, so test that validator also catches it defensively
    # via manual object construction
    from daap.spec.schema import (AgentMode, ConsolidationStrategy, HandoffMode,
                                   InstanceConfig, IOSchema, ModelTier, NodeSpec,
                                   TopologySpec, ConstraintSpec)

    ic = object.__new__(InstanceConfig)
    ic.__dict__.update({"parallel_instances": 3, "consolidation": None})

    node = object.__new__(NodeSpec)
    node.__dict__.update({
        "node_id": "para_node",
        "role": "Para",
        "model_tier": ModelTier.FAST,
        "system_prompt": "test",
        "tools": [],
        "inputs": [],
        "outputs": [IOSchema(data_key="r", data_type="string", description="")],
        "instance_config": ic,
        "handoff_mode": HandoffMode.NEVER,
        "operator_override": None,
        "agent_mode": AgentMode.SINGLE,
        "max_react_iterations": 10,
    })
    topo = object.__new__(TopologySpec)
    topo.__dict__.update({
        "topology_id": "t1", "version": 1,
        "created_at": "2026-04-13T10:00:00Z",
        "user_prompt": "test",
        "nodes": [node], "edges": [],
        "constraints": ConstraintSpec(),
        "operator_config": None, "metadata": {},
    })
    result = validate_topology(topo, AVAILABLE_TOOLS, AVAILABLE_MODELS)
    errors = [e for e in result.errors
              if "consolidation" in e.message.lower() and "3" in e.message]
    assert errors


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def test_all_categories_run_even_if_early_failure():
    # 3 nodes: node_a (unknown tool) + node_b connected, node_c is orphan (structural)
    nodes = [
        {**minimal_node("node_a"), "tools": [{"name": "FakeTool"}], "agent_mode": "react",
         "outputs": [{"data_key": "result", "data_type": "string", "description": ""}]},
        {**minimal_node("node_b"),
         "inputs": [{"data_key": "result", "data_type": "string", "description": ""}]},
        minimal_node("node_c"),  # orphan — no edges connect it
    ]
    edges = [{"source_node_id": "node_a", "target_node_id": "node_b",
              "data_key": "result", "description": ""}]
    topo = minimal_topology(nodes, edges)
    result = validate_topology(topo, AVAILABLE_TOOLS, AVAILABLE_MODELS)
    categories = {e.category for e in result.errors}
    assert "structural" in categories, f"Expected structural error, got: {result.error_summary}"
    assert "tool" in categories, f"Expected tool error, got: {result.error_summary}"


def test_error_summary_format():
    nodes = [
        {**minimal_node("node_a"), "tools": [{"name": "FakeTool1"}, {"name": "FakeTool2"}],
         "agent_mode": "react"},
    ]
    topo = minimal_topology(nodes)
    result = validate_topology(topo, AVAILABLE_TOOLS, AVAILABLE_MODELS)
    summary = result.error_summary
    assert "FakeTool1" in summary or "FakeTool2" in summary
    assert "[tool]" in summary
    assert "→ Fix:" in summary
