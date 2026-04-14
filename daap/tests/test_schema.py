"""
DAAP Schema Tests — pytest suite for spec/schema.py

Covers: happy path parsing, validation errors, serialization roundtrip.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from daap.spec.schema import (
    AgentMode,
    ConsolidationStrategy,
    ConstraintSpec,
    EdgeSpec,
    HandoffMode,
    IOSchema,
    InstanceConfig,
    ModelTier,
    NodeSpec,
    OperatorConfig,
    TopologySpec,
    get_topology_json_schema,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def minimal_node(node_id: str = "test_node") -> dict:
    return {
        "node_id": node_id,
        "role": "Test Node",
        "model_tier": "fast",
        "system_prompt": "You are a test agent.",
        "outputs": [{"data_key": "result", "data_type": "string", "description": "output"}],
        "instance_config": {"parallel_instances": 1},
        "agent_mode": "single",  # no tools → must use single mode
    }


def minimal_topology(nodes: list[dict] | None = None, edges: list[dict] | None = None) -> dict:
    return {
        "topology_id": "test-001",
        "version": 1,
        "created_at": "2026-04-13T10:00:00Z",
        "user_prompt": "Test task.",
        "nodes": nodes if nodes is not None else [minimal_node()],
        "edges": edges if edges is not None else [],
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_parse_sales_outreach_topology():
    raw = json.loads((FIXTURES / "sales_outreach_topology.json").read_text())
    topo = TopologySpec.model_validate(raw)

    assert len(topo.nodes) == 4
    assert len(topo.edges) == 3

    researcher = topo.nodes[0]
    assert researcher.instance_config.parallel_instances == 4
    assert researcher.instance_config.consolidation == ConsolidationStrategy.DEDUPLICATE


def test_single_node_topology():
    topo = TopologySpec.model_validate(minimal_topology())
    assert len(topo.nodes) == 1
    assert len(topo.edges) == 0


def test_default_constraints():
    topo = TopologySpec.model_validate(minimal_topology())
    assert topo.constraints.max_cost_usd == 1.00
    assert topo.constraints.max_latency_seconds == 120.0
    assert topo.constraints.max_nodes == 10
    assert topo.constraints.max_total_instances == 20
    assert topo.constraints.max_retries_per_node == 2


def test_handoff_mode_default():
    topo = TopologySpec.model_validate(minimal_topology())
    assert topo.nodes[0].handoff_mode == HandoffMode.NEVER


def test_json_schema_export():
    schema = get_topology_json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "nodes" in schema["properties"]


# ---------------------------------------------------------------------------
# Operator config
# ---------------------------------------------------------------------------

def test_operator_config_on_topology():
    data = minimal_topology()
    data["operator_config"] = {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_map": {
            "fast": "meta-llama/llama-3-8b-instruct",
            "smart": "anthropic/claude-3-5-sonnet",
            "powerful": "anthropic/claude-opus-4",
        },
    }
    topo = TopologySpec.model_validate(data)
    assert topo.operator_config.provider == "openrouter"
    assert topo.operator_config.model_map["fast"] == "meta-llama/llama-3-8b-instruct"


def test_operator_override_per_node():
    node = minimal_node()
    node["operator_override"] = {
        "provider": "opencode",
        "api_key_env": "OPENCODE_API_KEY",
        "model_map": {"fast": "opencode/fast-v1"},
    }
    topo = TopologySpec.model_validate(minimal_topology(nodes=[node]))
    assert topo.nodes[0].operator_override.provider == "opencode"


def test_no_operator_config_ok():
    """operator_config is optional — topology without it parses fine."""
    topo = TopologySpec.model_validate(minimal_topology())
    assert topo.operator_config is None


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_parallel_without_consolidation_fails():
    with pytest.raises(ValidationError, match="consolidation strategy required"):
        InstanceConfig(parallel_instances=3, consolidation=None)


def test_empty_nodes_fails():
    with pytest.raises(ValidationError):
        TopologySpec.model_validate(minimal_topology(nodes=[]))


def test_duplicate_node_ids_fails():
    nodes = [minimal_node("researcher"), minimal_node("researcher")]
    with pytest.raises(ValidationError, match="duplicate node_ids"):
        TopologySpec.model_validate(minimal_topology(nodes=nodes))


def test_broken_edge_reference_fails():
    edges = [{"source_node_id": "nonexistent_node", "target_node_id": "test_node", "data_key": "x"}]
    with pytest.raises(ValidationError, match="non-existent nodes"):
        TopologySpec.model_validate(minimal_topology(edges=edges))


def test_invalid_node_id_format_fails():
    node = minimal_node("Lead Researcher")  # spaces + uppercase
    with pytest.raises(ValidationError, match="node_id"):
        TopologySpec.model_validate(minimal_topology(nodes=[node]))


def test_node_without_outputs_fails():
    node = minimal_node()
    node["outputs"] = []
    with pytest.raises(ValidationError, match="at least one output"):
        TopologySpec.model_validate(minimal_topology(nodes=[node]))


def test_react_without_tools_fails():
    node = minimal_node()
    node["agent_mode"] = "react"
    node["tools"] = []
    with pytest.raises(ValidationError, match="no tools provided"):
        TopologySpec.model_validate(minimal_topology(nodes=[node]))


def test_single_mode_no_tools_ok():
    node = minimal_node()
    node["agent_mode"] = "single"
    node["tools"] = []
    topo = TopologySpec.model_validate(minimal_topology(nodes=[node]))
    assert topo.nodes[0].agent_mode == AgentMode.SINGLE


def test_default_agent_mode_is_react():
    # Remove explicit agent_mode so Pydantic uses the default, add tool so react validator passes
    node = minimal_node()
    node["tools"] = [{"name": "WebSearch"}]
    del node["agent_mode"]  # let schema default kick in
    topo = TopologySpec.model_validate(minimal_topology(nodes=[node]))
    assert topo.nodes[0].agent_mode == AgentMode.REACT


def test_invalid_operator_provider_fails():
    data = minimal_topology()
    data["operator_config"] = {"provider": "   ", "api_key_env": "KEY"}
    with pytest.raises(ValidationError, match="provider must not be empty"):
        TopologySpec.model_validate(data)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_roundtrip_serialization():
    raw = json.loads((FIXTURES / "sales_outreach_topology.json").read_text())
    topo1 = TopologySpec.model_validate(raw)
    serialized = topo1.model_dump_json()
    topo2 = TopologySpec.model_validate_json(serialized)
    assert topo1 == topo2


def test_topology_to_dict():
    raw = json.loads((FIXTURES / "sales_outreach_topology.json").read_text())
    topo = TopologySpec.model_validate(raw)
    d = topo.model_dump()
    assert isinstance(d, dict)
    json.dumps(d)  # must not raise
