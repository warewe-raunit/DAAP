"""
DAAP Node Builder Tests — pytest suite for executor/node_builder.py

All tests mock AgentScope model/agent creation — no real API calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from daap.spec.resolver import ResolvedNode
from daap.executor.node_builder import BuiltNode, build_node
from daap.tools.registry import get_tool_registry


def make_resolved_node(
    node_id="test_node",
    agent_mode="react",
    max_react_iterations=10,
    concrete_model_id="claude-haiku-4-5-20251001",
    tool_ids=None,
    operator_provider="openrouter",
    operator_base_url=None,
    operator_api_key_env="OPENROUTER_API_KEY",
):
    from daap.spec.schema import IOSchema
    return ResolvedNode(
        node_id=node_id,
        role="Test Node",
        concrete_model_id=concrete_model_id,
        system_prompt="You are a test agent.",
        concrete_tool_ids=tool_ids or ["agentscope.tools.WebSearch"],
        inputs=[],
        outputs=[IOSchema(data_key="result", data_type="string", description="output")],
        parallel_instances=1,
        consolidation_func=None,
        handoff_mode="never",
        agent_mode=agent_mode,
        max_react_iterations=max_react_iterations,
        operator_provider=operator_provider,
        operator_base_url=operator_base_url,
        operator_api_key_env=operator_api_key_env,
    )


@pytest.mark.asyncio
@patch("daap.executor.node_builder.TerminatingReActAgent")
@patch("daap.executor.node_builder.TrackedOpenAIChatModel")
async def test_build_react_node(mock_model_cls, mock_agent_cls):
    mock_agent_cls.return_value = MagicMock()
    mock_model_cls.return_value = MagicMock()

    node = make_resolved_node(agent_mode="react", max_react_iterations=10)
    registry = get_tool_registry()
    built = await build_node(node, registry)

    assert isinstance(built, BuiltNode)
    assert built.node_id == "test_node"
    assert built.agent_mode == "react"
    assert built.parallel_instances == 1

    # ReActAgent called with correct max_iters
    call_kwargs = mock_agent_cls.call_args.kwargs
    assert call_kwargs["max_iters"] == 10
    assert call_kwargs["name"] == "test_node"


@pytest.mark.asyncio
@patch("daap.executor.node_builder.TerminatingReActAgent")
@patch("daap.executor.node_builder.TrackedOpenAIChatModel")
async def test_build_single_node(mock_model_cls, mock_agent_cls):
    mock_agent_cls.return_value = MagicMock()
    mock_model_cls.return_value = MagicMock()

    node = make_resolved_node(agent_mode="single", tool_ids=[])
    registry = get_tool_registry()
    built = await build_node(node, registry)

    assert built.agent_mode == "single"
    # Single mode → max_iters=1
    call_kwargs = mock_agent_cls.call_args.kwargs
    assert call_kwargs["max_iters"] == 1


@pytest.mark.asyncio
@patch("daap.executor.node_builder.TrackedOpenAIChatModel")
async def test_build_node_with_unknown_tool_raises(mock_model_cls):
    mock_model_cls.return_value = MagicMock()
    node = make_resolved_node(
        agent_mode="react",
        tool_ids=["agentscope.tools.NonExistentTool"],
    )
    with pytest.raises(ValueError, match="not found in registry"):
        await build_node(node, get_tool_registry())


@pytest.mark.asyncio
@patch("daap.executor.node_builder.TerminatingReActAgent")
@patch("daap.executor.node_builder.TrackedOpenAIChatModel")
async def test_build_node_anthropic_model_selection(mock_model_cls, mock_agent_cls):
    mock_agent_cls.return_value = MagicMock()
    mock_model_cls.return_value = MagicMock()

    node = make_resolved_node(
        concrete_model_id="claude-sonnet-4-6",
        operator_provider="anthropic",
    )
    await build_node(node, get_tool_registry())

    mock_model_cls.assert_called_once()
    call_kwargs = mock_model_cls.call_args.kwargs
    assert call_kwargs["model_name"] == "anthropic/claude-sonnet-4-6"
    assert call_kwargs["client_kwargs"]["base_url"] == "https://openrouter.ai/api/v1"


@pytest.mark.asyncio
@patch("daap.executor.node_builder.TerminatingReActAgent")
@patch("daap.executor.node_builder.TrackedOpenAIChatModel")
async def test_build_node_openrouter_model_selection(mock_model_cls, mock_agent_cls):
    mock_agent_cls.return_value = MagicMock()
    mock_model_cls.return_value = MagicMock()

    node = make_resolved_node(
        concrete_model_id="anthropic/claude-3-5-sonnet",
        operator_provider="openrouter",
        operator_base_url="https://openrouter.ai/api/v1",
        operator_api_key_env="OPENROUTER_API_KEY",
    )
    await build_node(node, get_tool_registry())

    mock_model_cls.assert_called_once()
    call_kwargs = mock_model_cls.call_args.kwargs
    assert call_kwargs["model_name"] == "anthropic/claude-3-5-sonnet"
    assert call_kwargs["client_kwargs"]["base_url"] == "https://openrouter.ai/api/v1"


@pytest.mark.asyncio
@patch("daap.executor.node_builder.TerminatingReActAgent")
@patch("daap.executor.node_builder.TrackedOpenAIChatModel")
async def test_mcp_tool_skipped_not_raised(mock_model_cls, mock_agent_cls):
    """MCP tools should be skipped (with warning), not raise errors."""
    mock_agent_cls.return_value = MagicMock()
    mock_model_cls.return_value = MagicMock()

    node = make_resolved_node(
        agent_mode="react",
        tool_ids=["mcp://linkedin", "agentscope.tools.WebSearch"],
    )
    built = await build_node(node, get_tool_registry())
    assert built is not None  # didn't raise


@pytest.mark.asyncio
@patch("daap.executor.node_builder.TerminatingReActAgent")
@patch("daap.executor.node_builder.TrackedOpenAIChatModel")
async def test_build_node_exposes_isolated_agent_factory(mock_model_cls, mock_agent_cls):
    mock_model_cls.return_value = MagicMock()

    created_agents = []

    def _make_agent(*args, **kwargs):
        agent = MagicMock()
        created_agents.append(agent)
        return agent

    mock_agent_cls.side_effect = _make_agent

    node = make_resolved_node(agent_mode="react", tool_ids=["agentscope.tools.WebSearch"])
    built = await build_node(node, get_tool_registry())

    assert callable(built.agent_factory)
    a1 = built.agent_factory()
    a2 = built.agent_factory()
    assert a1 is not a2
    assert len(created_agents) >= 3  # initial agent + 2 factory spawns


@pytest.mark.asyncio
@patch("daap.executor.node_builder.TerminatingReActAgent")
@patch("daap.executor.node_builder.TrackedOpenAIChatModel")
async def test_build_node_applies_subagent_skills(mock_model_cls, mock_agent_cls):
    mock_agent_cls.return_value = MagicMock()
    mock_model_cls.return_value = MagicMock()

    seen_targets: list[str] = []

    def _fake_apply(toolkit, target):
        seen_targets.append(target)
        return []

    node = make_resolved_node(agent_mode="react", tool_ids=["agentscope.tools.WebSearch"])
    with patch("daap.executor.node_builder.apply_configured_skills", side_effect=_fake_apply):
        await build_node(node, get_tool_registry())

    assert seen_targets
    assert set(seen_targets) == {"subagent"}
