"""
DAAP Patterns Tests — pytest suite for executor/patterns.py

Uses MockAgent (simple async callable) — no real API calls.
"""

import asyncio
import pytest

from agentscope.message import Msg

from daap.executor.node_builder import BuiltNode
from daap.executor.patterns import (
    consolidate_outputs,
    run_execution_step,
    run_parallel_instances,
)


# ---------------------------------------------------------------------------
# MockAgent — async callable returning a fixed Msg
# ---------------------------------------------------------------------------

class MockAgent:
    """Lightweight test agent — no AgentScope internals."""
    def __init__(self, name: str, response: str):
        self.name = name
        self._response = response

    async def __call__(self, msg):
        return Msg(name=self.name, content=self._response, role="assistant")


def make_built_node(
    node_id="node_a",
    response="agent output",
    parallel_instances=1,
    consolidation_func=None,
    agent_mode="single",
    agent_factory=None,
) -> BuiltNode:
    return BuiltNode(
        node_id=node_id,
        agent=MockAgent(node_id, response),
        parallel_instances=parallel_instances,
        consolidation_func=consolidation_func,
        agent_mode=agent_mode,
        operator_provider="anthropic",
        operator_base_url=None,
        operator_api_key_env="OPENROUTER_API_KEY",
        agent_factory=agent_factory,
    )


# ---------------------------------------------------------------------------
# run_parallel_instances
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_instance_no_fanout():
    node = make_built_node(parallel_instances=1)
    input_msg = Msg(name="user", content="do the thing", role="user")
    results = await run_parallel_instances(node, input_msg)
    assert len(results) == 1
    assert results[0].content == "agent output"


@pytest.mark.asyncio
async def test_parallel_instances():
    node = make_built_node(parallel_instances=3)
    input_msg = Msg(name="user", content="do the thing", role="user")
    results = await run_parallel_instances(node, input_msg)
    assert len(results) == 3
    assert all(r.content == "agent output" for r in results)


@pytest.mark.asyncio
async def test_parallel_instances_use_isolated_agents_with_factory():
    created_agent_ids = []

    class IsolatedAgent:
        def __init__(self):
            self.agent_id = len(created_agent_ids) + 1
            self.calls = 0
            created_agent_ids.append(self.agent_id)

        async def __call__(self, msg):
            self.calls += 1
            return Msg(
                name=f"agent_{self.agent_id}",
                content=f"instance={self.agent_id};calls={self.calls}",
                role="assistant",
            )

    def _factory():
        return IsolatedAgent()

    node = make_built_node(
        parallel_instances=3,
        agent_factory=_factory,
    )
    input_msg = Msg(name="user", content="do the thing", role="user")

    results = await run_parallel_instances(node, input_msg)

    assert len(results) == 3
    assert len(created_agent_ids) == 3
    assert len({r.content for r in results}) == 3


# ---------------------------------------------------------------------------
# consolidate_outputs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consolidation_merge():
    outputs = [
        Msg(name="a", content="Result A", role="assistant"),
        Msg(name="b", content="Result B", role="assistant"),
        Msg(name="c", content="Result C", role="assistant"),
    ]
    result = await consolidate_outputs(outputs, "merge")
    assert "Result A" in result.content
    assert "Result B" in result.content
    assert "Result C" in result.content


@pytest.mark.asyncio
async def test_consolidation_vote():
    outputs = [
        Msg(name="a", content="A", role="assistant"),
        Msg(name="b", content="B", role="assistant"),
        Msg(name="c", content="A", role="assistant"),
    ]
    result = await consolidate_outputs(outputs, "vote")
    assert result.content.strip() == "A"


@pytest.mark.asyncio
async def test_consolidation_single_output_passthrough():
    outputs = [Msg(name="a", content="Only one", role="assistant")]
    result = await consolidate_outputs(outputs, "merge")
    assert result.content == "Only one"


@pytest.mark.asyncio
async def test_consolidation_empty_outputs():
    result = await consolidate_outputs([], "merge")
    assert result.content == ""


@pytest.mark.asyncio
async def test_consolidation_deduplicate_falls_back_without_model():
    """Without a model_id, deduplicate falls back to merge."""
    outputs = [
        Msg(name="a", content="Alpha", role="assistant"),
        Msg(name="b", content="Beta", role="assistant"),
    ]
    result = await consolidate_outputs(outputs, "deduplicate", consolidation_model_id=None)
    assert "Alpha" in result.content
    assert "Beta" in result.content


# ---------------------------------------------------------------------------
# run_execution_step
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execution_step_single_node():
    node = make_built_node("writer", "final draft")
    initial = Msg(name="user", content="write something", role="user")
    result = await run_execution_step([node], {}, [], initial)
    assert "writer" in result
    assert result["writer"].content == "final draft"


@pytest.mark.asyncio
async def test_execution_step_parallel_nodes():
    node_b = make_built_node("node_b", "output B")
    node_c = make_built_node("node_c", "output C")
    initial = Msg(name="user", content="start", role="user")
    result = await run_execution_step([node_b, node_c], {}, [], initial)
    assert "node_b" in result
    assert "node_c" in result
    assert result["node_b"].content == "output B"
    assert result["node_c"].content == "output C"


@pytest.mark.asyncio
async def test_execution_step_uses_prior_output():
    """Node should receive prior step's output, not the original prompt."""

    class EchoAgent:
        """Returns whatever it receives."""
        def __init__(self, name):
            self.name = name
        async def __call__(self, msg):
            return Msg(name=self.name, content=f"echo: {msg.content}", role="assistant")

    node = BuiltNode(
        node_id="node_b",
        agent=EchoAgent("node_b"),
        parallel_instances=1,
        consolidation_func=None,
        agent_mode="single",
        operator_provider="anthropic",
        operator_base_url=None,
        operator_api_key_env="OPENROUTER_API_KEY",
    )

    # Simulate prior step output keyed by data_key
    prior_data = {"raw_leads": Msg(name="node_a", content="lead data", role="assistant")}

    # Edge: node_a → node_b via raw_leads
    class FakeEdge:
        source_node_id = "node_a"
        target_node_id = "node_b"
        data_key = "raw_leads"

    initial = Msg(name="user", content="original prompt", role="user")
    result = await run_execution_step([node], prior_data, [FakeEdge()], initial)

    assert "echo: lead data" in result["node_b"].content
    assert "original prompt" not in result["node_b"].content


@pytest.mark.asyncio
async def test_first_node_gets_user_prompt():
    """First node (no incoming edges) gets original user prompt."""

    class EchoAgent:
        def __init__(self, name):
            self.name = name
        async def __call__(self, msg):
            return Msg(name=self.name, content=f"echo: {msg.content}", role="assistant")

    node = BuiltNode(
        node_id="first_node",
        agent=EchoAgent("first_node"),
        parallel_instances=1,
        consolidation_func=None,
        agent_mode="single",
        operator_provider="anthropic",
        operator_base_url=None,
        operator_api_key_env="OPENROUTER_API_KEY",
    )

    initial = Msg(name="user", content="find leads", role="user")
    result = await run_execution_step([node], {}, [], initial)
    assert "echo: find leads" in result["first_node"].content
