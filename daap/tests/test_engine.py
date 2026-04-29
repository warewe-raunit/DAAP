"""
DAAP Engine Tests — pytest suite for executor/engine.py

Uses MockAgent and patches build_node — no real API calls.
"""

import json
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from agentscope.message import Msg

from daap.spec.schema import TopologySpec
from daap.spec.resolver import resolve_topology, ResolvedTopology
from daap.executor.engine import ExecutionResult, NodeResult, execute_topology
from daap.executor.node_builder import BuiltNode

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockAgent:
    def __init__(self, name: str, response: str = "mock output"):
        self.name = name
        self._response = response
        self.call_count = 0

    async def __call__(self, msg):
        self.call_count += 1
        return Msg(name=self.name, content=self._response, role="assistant")

    async def reply(self, msg, structured_model=None):
        # Production interface (AgentScope ReActAgent) uses .reply() with a
        # structured_model kwarg. Single-mode patterns.py calls reply with
        # structured_model=None; react-mode passes NodeResult and reads
        # result.metadata['result']. Mock both shapes.
        self.call_count += 1
        if structured_model is not None:
            return Msg(
                name=self.name,
                content=self._response,
                role="assistant",
                metadata={"result": self._response},
            )
        return Msg(name=self.name, content=self._response, role="assistant")


class FailOnceAgent:
    """Fails on first call, succeeds on subsequent calls."""
    def __init__(self, name: str):
        self.name = name
        self.call_count = 0

    async def __call__(self, msg):
        return await self.reply(msg)

    async def reply(self, msg, structured_model=None):
        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError(f"Node {self.name} failed (attempt 1)")
        return Msg(name=self.name, content="recovered output", role="assistant")


class AlwaysFailAgent:
    def __init__(self, name: str):
        self.name = name

    async def __call__(self, msg):
        raise RuntimeError(f"Node {self.name} always fails")

    async def reply(self, msg, structured_model=None):
        raise RuntimeError(f"Node {self.name} always fails")


def make_built_node(node_id, response="mock output", parallel_instances=1) -> BuiltNode:
    return BuiltNode(
        node_id=node_id,
        agent=MockAgent(node_id, response),
        parallel_instances=parallel_instances,
        consolidation_func=None,
        consolidation_strategy=None,
        agent_mode="single",
        operator_provider="openrouter",
        operator_base_url=None,
        operator_api_key_env="OPENROUTER_API_KEY",
    )


def load_resolved(name: str) -> ResolvedTopology:
    raw = json.loads((FIXTURES / name).read_text())
    topo = TopologySpec.model_validate(raw)
    result = resolve_topology(topo)
    assert isinstance(result, ResolvedTopology)
    return result


def patch_build_node(responses: dict[str, str]):
    """
    Patch build_node to return MockAgents keyed by node_id.
    responses: {node_id: response_text}
    """
    async def _mock_build(resolved_node, tool_registry, daap_memory=None, tracker=None, user_id=None, today=None):
        resp = responses.get(resolved_node.node_id, "mock output")
        return make_built_node(resolved_node.node_id, resp,
                               resolved_node.parallel_instances)
    return patch("daap.executor.engine.build_node", side_effect=_mock_build)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_simple_topology():
    resolved = load_resolved("parallel_branches_topology.json")
    # Use only input_parser and synthesizer for simplicity
    responses = {
        "input_parser": "parsed data",
        "researcher_a": "research A",
        "researcher_b": "research B",
        "synthesizer": "final synthesis",
    }
    with patch_build_node(responses):
        result = await execute_topology(resolved, "test prompt")

    assert result.success, f"Expected success, got error: {result.error}"
    assert result.final_output == "final synthesis"
    assert len(result.node_results) == 4


@pytest.mark.asyncio
async def test_execute_sales_outreach_topology():
    resolved = load_resolved("sales_outreach_topology.json")
    responses = {
        "lead_researcher": "raw leads data",
        "lead_evaluator": "qualified leads",
        "personalization_researcher": "personalization briefs",
        "email_drafter": "final emails",
    }
    with patch_build_node(responses):
        result = await execute_topology(resolved, "Find leads for SaaS")

    assert result.success
    assert result.final_output == "final emails"
    assert len(result.node_results) == 4

    executed_ids = [r.node_id for r in result.node_results]
    assert executed_ids == [
        "lead_researcher", "lead_evaluator",
        "personalization_researcher", "email_drafter",
    ]


@pytest.mark.asyncio
async def test_execute_parallel_topology():
    resolved = load_resolved("parallel_branches_topology.json")
    responses = {
        "input_parser": "parsed",
        "researcher_a": "branch A result",
        "researcher_b": "branch B result",
        "synthesizer": "synthesized",
    }
    with patch_build_node(responses):
        result = await execute_topology(resolved, "test")

    assert result.success
    executed_ids = {r.node_id for r in result.node_results}
    assert "researcher_a" in executed_ids
    assert "researcher_b" in executed_ids
    assert "synthesizer" in executed_ids


@pytest.mark.asyncio
async def test_node_failure_returns_error():
    resolved = load_resolved("parallel_branches_topology.json")

    async def _failing_build(resolved_node, tool_registry, daap_memory=None, tracker=None, user_id=None, today=None):
        if resolved_node.node_id == "researcher_a":
            node = BuiltNode(
                node_id=resolved_node.node_id,
                agent=AlwaysFailAgent(resolved_node.node_id),
                parallel_instances=1,
                consolidation_func=None,
                consolidation_strategy=None,
                agent_mode="single",
                operator_provider="openrouter",
                operator_base_url=None,
                operator_api_key_env="OPENROUTER_API_KEY",
            )
            return node
        return make_built_node(resolved_node.node_id)

    with patch("daap.executor.engine.build_node", side_effect=_failing_build):
        # Set max_retries to 0 so it fails fast
        resolved.constraints.__dict__["max_retries_per_node"] = 0
        result = await execute_topology(resolved, "test")

    assert not result.success
    assert result.error is not None
    assert "researcher" in result.error.lower() or "failed" in result.error.lower()


@pytest.mark.asyncio
async def test_retry_on_failure():
    resolved = load_resolved("parallel_branches_topology.json")
    fail_once = FailOnceAgent("input_parser")

    async def _build_with_fail_once(resolved_node, tool_registry, daap_memory=None, tracker=None, user_id=None, today=None):
        if resolved_node.node_id == "input_parser":
            return BuiltNode(
                node_id="input_parser",
                agent=fail_once,
                parallel_instances=1,
                consolidation_func=None,
                consolidation_strategy=None,
                agent_mode="single",
                operator_provider="openrouter",
                operator_base_url=None,
                operator_api_key_env="OPENROUTER_API_KEY",
            )
        return make_built_node(resolved_node.node_id)

    with patch("daap.executor.engine.build_node", side_effect=_build_with_fail_once):
        resolved.constraints.__dict__["max_retries_per_node"] = 1
        result = await execute_topology(resolved, "test")

    assert result.success, f"Should have recovered: {result.error}"
    assert fail_once.call_count >= 2  # failed once, retried


@pytest.mark.asyncio
async def test_timing_recorded():
    resolved = load_resolved("parallel_branches_topology.json")
    responses = {n: "output" for n in
                 ["input_parser", "researcher_a", "researcher_b", "synthesizer"]}
    with patch_build_node(responses):
        result = await execute_topology(resolved, "test")

    assert result.total_latency_seconds >= 0  # mocks run in microseconds, may round to 0
    assert all(r.latency_seconds >= 0 for r in result.node_results)


@pytest.mark.asyncio
async def test_data_flows_between_nodes():
    """node_b should receive node_a's output, not the original prompt."""
    resolved = load_resolved("parallel_branches_topology.json")

    received_inputs: dict[str, str] = {}

    class RecordingAgent:
        def __init__(self, name, response):
            self.name = name
            self._response = response

        async def __call__(self, msg):
            return await self.reply(msg)

        async def reply(self, msg, structured_model=None):
            received_inputs[self.name] = msg.content
            if structured_model is not None:
                return Msg(
                    name=self.name,
                    content=self._response,
                    role="assistant",
                    metadata={"result": self._response},
                )
            return Msg(name=self.name, content=self._response, role="assistant")

    async def _recording_build(resolved_node, tool_registry, daap_memory=None, tracker=None, user_id=None, today=None):
        resp = {
            "input_parser": "PARSED_DATA",
            "researcher_a": "RESEARCH_A",
            "researcher_b": "RESEARCH_B",
            "synthesizer": "FINAL",
        }.get(resolved_node.node_id, "output")
        return BuiltNode(
            node_id=resolved_node.node_id,
            agent=RecordingAgent(resolved_node.node_id, resp),
            parallel_instances=resolved_node.parallel_instances,
            consolidation_func=None,
            consolidation_strategy=None,
            agent_mode="single",
            operator_provider="openrouter",
            operator_base_url=None,
            operator_api_key_env="OPENROUTER_API_KEY",
        )

    with patch("daap.executor.engine.build_node", side_effect=_recording_build):
        result = await execute_topology(resolved, "ORIGINAL_PROMPT")

    assert result.success
    # input_parser is first — should get user prompt
    assert "ORIGINAL_PROMPT" in received_inputs.get("input_parser", "")
    # researcher_a/b should get input_parser's output, NOT the original prompt
    assert "PARSED_DATA" in received_inputs.get("researcher_a", "")
    assert "ORIGINAL_PROMPT" not in received_inputs.get("researcher_a", "")
