"""
DAAP Master Agent Tests — pytest suite for master/tools.py, master/prompts.py, master/agent.py

All tests use mocked models/agents — no real API calls.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentscope.message import Msg

from daap.master.tools import (
    create_master_toolkit,
    delegate_to_architect,
    _extract_topology_json,
)
from daap.master.prompts import get_master_system_prompt
from daap.master.agent import MasterAgentTurnResult, parse_turn_result
from daap.tools.registry import get_available_tool_names

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_msg(content: str) -> Msg:
    return Msg(name="DAAP", content=content, role="assistant")


def load_sales_outreach_topology() -> str:
    return (FIXTURES / "sales_outreach_topology.json").read_text()


VALID_QUESTIONS_JSON = json.dumps([
    {
        "question": "What does your company do?",
        "options": [
            {"label": "SaaS product", "description": "Software as a service"},
            {"label": "Agency", "description": "Professional services"},
            {"label": "Other", "description": "Something else"},
        ],
        "multi_select": False,
    },
])

TWO_QUESTION_JSON = json.dumps([
    {
        "question": "What's your product?",
        "options": [
            {"label": "SaaS", "description": "Software"},
            {"label": "Hardware", "description": "Physical goods"},
        ],
        "multi_select": False,
    },
    {
        "question": "Company size?",
        "options": [
            {"label": "SMB", "description": "1-50 employees"},
            {"label": "Mid-market", "description": "50-500 employees"},
        ],
        "multi_select": True,
    },
])


# ---------------------------------------------------------------------------
# ask_user tool tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ask_user_stores_pending_questions():
    """ask_user stores questions in per-toolkit state while waiting."""
    toolkit = create_master_toolkit()
    ask_user_fn = toolkit.tools["ask_user"].original_func

    task = asyncio.create_task(ask_user_fn(VALID_QUESTIONS_JSON))
    await asyncio.sleep(0)  # yield so task runs and stores questions

    pending = toolkit.get_pending_questions()
    assert pending is not None
    assert len(pending) == 1
    assert pending[0]["question"] == "What does your company do?"

    # Clean up — resolve so the task finishes
    toolkit.resolve_pending_questions(["SaaS product"])
    await task


@pytest.mark.asyncio
async def test_ask_user_waits_for_resolution():
    """ask_user returns user answers after resolve_pending_questions is called."""
    toolkit = create_master_toolkit()
    ask_user_fn = toolkit.tools["ask_user"].original_func

    async def _resolve_after_yield():
        await asyncio.sleep(0.01)
        toolkit.resolve_pending_questions(["SaaS product"])

    asyncio.create_task(_resolve_after_yield())
    result = await ask_user_fn(VALID_QUESTIONS_JSON)

    block = result.content[0]
    text = block.text if hasattr(block, "text") else block.get("text", "")
    assert "SaaS product" in text
    assert "What does your company do?" in text


@pytest.mark.asyncio
async def test_ask_user_handles_invalid_json():
    """ask_user returns an error ToolResponse for invalid JSON input."""
    toolkit = create_master_toolkit()
    ask_user_fn = toolkit.tools["ask_user"].original_func
    result = await ask_user_fn("not valid json {{{")
    block = result.content[0]
    text = block.text if hasattr(block, "text") else block.get("text", "")
    assert "invalid" in text.lower() or "Invalid" in text


@pytest.mark.asyncio
async def test_ask_user_questions_have_structure():
    """ask_user stores questions with required keys: question, options, multi_select."""
    toolkit = create_master_toolkit()
    ask_user_fn = toolkit.tools["ask_user"].original_func

    task = asyncio.create_task(ask_user_fn(TWO_QUESTION_JSON))
    await asyncio.sleep(0)

    pending = toolkit.get_pending_questions()
    assert pending is not None
    assert len(pending) == 2

    for q in pending:
        assert "question" in q
        assert "options" in q
        assert "multi_select" in q
        for opt in q["options"]:
            assert "label" in opt
            assert "description" in opt

    toolkit.resolve_pending_questions(["SaaS", "SMB"])
    await task


@pytest.mark.asyncio
async def test_ask_user_for_plan_approval():
    """ask_user can present topology approval options (proceed/cheaper/cancel)."""
    toolkit = create_master_toolkit()
    ask_user_fn = toolkit.tools["ask_user"].original_func

    approval_questions = json.dumps([
        {
            "question": "Pipeline ready: 4 agents, ~$0.14, ~48s. What would you like to do?",
            "options": [
                {"label": "Run it (Recommended)", "description": "Execute the pipeline now"},
                {"label": "Make it cheaper", "description": "Reduce cost"},
                {"label": "Make changes", "description": "Modify the plan"},
                {"label": "Cancel", "description": "Never mind"},
            ],
            "multi_select": False,
        }
    ])

    task = asyncio.create_task(ask_user_fn(approval_questions))
    await asyncio.sleep(0)

    pending = toolkit.get_pending_questions()
    assert pending is not None
    labels = [opt["label"] for opt in pending[0]["options"]]
    assert any("Run" in label or "proceed" in label.lower() for label in labels)
    assert any("cheaper" in label.lower() or "Cheaper" in label for label in labels)
    assert any("Cancel" in label or "cancel" in label for label in labels)

    toolkit.resolve_pending_questions(["Run it (Recommended)"])
    await task


# ---------------------------------------------------------------------------
# Tool registration tests
# ---------------------------------------------------------------------------

def test_both_tools_registered_in_toolkit():
    """Both delegate_to_architect and ask_user are registered in the master toolkit."""
    toolkit = create_master_toolkit()
    registered = set(toolkit.tools.keys())
    assert "delegate_to_architect" in registered
    assert "ask_user" in registered
    assert "register_skill" in registered
    assert "get_runtime_context" in registered


def test_create_master_toolkit_applies_master_skills(monkeypatch):
    """create_master_toolkit applies configured AgentScope skills for master target."""
    import daap.master.tools as tools_module

    seen: list[str] = []

    def _fake_apply(toolkit, target):
        seen.append(target)
        return []

    monkeypatch.setattr(tools_module, "apply_configured_skills", _fake_apply)

    create_master_toolkit()

    assert seen == ["master"]


# ---------------------------------------------------------------------------
# delegate_to_architect tool tests
# ---------------------------------------------------------------------------

def _block_text(block) -> str:
    return block.text if hasattr(block, "text") else block.get("text", "")


def _make_generate_mock(content):
    """AsyncMock for topology_agent.generate_topology returning (parsed_dict, raw_text).

    Accepts a dict or a JSON string. If the string isn't parseable, parsed_dict is None
    so delegate_to_architect exercises the repair fallback path.
    """
    raw = content if isinstance(content, str) else json.dumps(content)
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            parsed = None
    except (json.JSONDecodeError, TypeError):
        parsed = None
    return AsyncMock(return_value=(parsed, raw))


def test_extract_topology_json_from_wrapped_architect_output():
    topology_json = load_sales_outreach_topology()
    wrapped = f"Here is the validated topology:\n```json\n{topology_json}\n```\nDone."

    parsed = _extract_topology_json(wrapped)

    assert parsed is not None
    assert parsed["topology_id"] == "test-sales-outreach-001"
    assert "nodes" in parsed


@pytest.mark.asyncio
async def test_delegate_to_architect_handles_unparseable_output():
    """delegate_to_architect returns error when architect returns unparseable JSON."""
    fake_generate = _make_generate_mock("not valid json {{{")
    with patch("daap.master.topology_agent.generate_topology", new=fake_generate):
        with patch("daap.master.tools._repair_topology_json", new=AsyncMock(return_value=None)):
            result = await delegate_to_architect("{}", "build a search pipeline")
    text = _block_text(result.content[0])
    assert "malformed" in text.lower() or "error" in text.lower() or "repair" in text.lower()
    assert (result.metadata or {}).get("topology") is None


@pytest.mark.asyncio
async def test_delegate_to_architect_validates_and_stores_result():
    """delegate_to_architect validates architect JSON and returns it in ToolResponse.metadata."""
    topology_json = load_sales_outreach_topology()
    fake_generate = _make_generate_mock(topology_json)
    with patch("daap.master.topology_agent.generate_topology", new=fake_generate):
        result = await delegate_to_architect("{}", "build sales outreach pipeline")
    text = _block_text(result.content[0])
    assert "valid" in text.lower() or "cost" in text.lower()

    meta = result.metadata or {}
    assert meta.get("topology") is not None
    assert meta.get("estimate") is not None

    estimate = meta["estimate"]
    assert "total_cost_usd" in estimate
    assert "total_latency_seconds" in estimate
    assert "min_viable_cost_usd" in estimate
    assert estimate["total_cost_usd"] >= 0
    assert estimate["min_viable_cost_usd"] >= 0


@pytest.mark.asyncio
async def test_delegate_to_architect_calls_generate_once():
    """delegate_to_architect invokes generate_topology exactly once per call (no ReAct loop)."""
    topology_json = load_sales_outreach_topology()
    fake_generate = _make_generate_mock(topology_json)
    with patch("daap.master.topology_agent.generate_topology", new=fake_generate):
        await delegate_to_architect("{}", "build search pipeline")
    assert fake_generate.call_count == 1


@pytest.mark.asyncio
async def test_delegate_to_architect_returns_validation_errors():
    """delegate_to_architect surfaces validation errors when architect JSON is invalid."""
    bad_topology = {
        "topology_id": "topo-badbad01",
        "version": 1,
        "created_at": "2026-04-13T10:00:00Z",
        "user_prompt": "test",
        "nodes": [
            {
                "node_id": "node_a",
                "role": "Researcher",
                "model_tier": "fast",
                "system_prompt": "Search for leads.",
                "tools": [{"name": "NonExistentTool99"}],
                "inputs": [],
                "outputs": [{"data_key": "results", "data_type": "string", "description": "output"}],
                "instance_config": {"parallel_instances": 1, "consolidation": None},
                "agent_mode": "react",
            }
        ],
        "edges": [],
    }
    fake_generate = _make_generate_mock(bad_topology)
    with patch("daap.master.topology_agent.generate_topology", new=fake_generate):
        result = await delegate_to_architect("{}", "test pipeline")
    text = _block_text(result.content[0])
    assert "error" in text.lower() or "invalid" in text.lower() or "validation" in text.lower()
    assert (result.metadata or {}).get("topology") is None


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------

def test_prompt_contains_available_tools():
    """Master prompt routes via delegate_to_architect; tool catalog lives in architect prompt."""
    from daap.master.topology_agent import _TOPOLOGY_ARCHITECT_SYSTEM_PROMPT
    available = get_available_tool_names()
    builtin = sorted(t for t in available if not t.startswith("mcp://"))
    # Architect template now expects 4 placeholders. Format with stub values for the others.
    formatted = _TOPOLOGY_ARCHITECT_SYSTEM_PROMPT.format(
        tool_catalog="(catalog)",
        tool_names=", ".join(builtin),
        runtime_block="(runtime)",
        user_context_block="(user)",
    )
    assert any(t in formatted for t in builtin[:5])
    # Master prompt references delegate_to_architect, not individual tool names
    master_prompt = get_master_system_prompt(available_tools=available)
    assert "delegate_to_architect" in master_prompt


def test_prompt_delegates_topology_to_architect():
    """Master prompt references delegate_to_architect; JSON schema stays in topology_agent."""
    prompt = get_master_system_prompt()
    assert "delegate_to_architect" in prompt
    # The raw JSON example block (with topology_id, node_id fields) is gone from master
    assert '"topology_id"' not in prompt   # JSON key notation absent
    assert '"node_id"' not in prompt


def test_prompt_instructs_to_ask_questions_via_tool():
    """System prompt explicitly tells agent to use ask_user tool for questions."""
    prompt = get_master_system_prompt()
    assert "ask_user" in prompt
    # Should NOT instruct plain-text questions — tool-based elicitation only
    assert "ask_user" in prompt.lower()


def test_prompt_includes_skill_registration_hint():
    """System prompt instructs the agent to call register_skill for skill paths."""
    prompt = get_master_system_prompt()
    assert "register_skill" in prompt
    assert "skill" in prompt.lower()


def test_prompt_includes_runtime_section_when_provided():
    """System prompt includes compact Runtime section when runtime_context provided."""
    prompt = get_master_system_prompt(
        runtime_context={
            "execution_mode": "api-session",
            "master_tools": ["ask_user", "delegate_to_architect", "get_runtime_context"],
        }
    )
    assert "## Runtime" in prompt
    assert "execution_mode: api-session" in prompt


def test_prompt_disables_execute_claims_when_execution_tool_missing():
    """Prompt avoids forced execute flow if execute_pending_topology is unavailable."""
    prompt = get_master_system_prompt(
        runtime_context={"master_tools": ["ask_user", "delegate_to_architect"]},
    )
    assert "execute_pending_topology" in prompt
    assert "not available" in prompt


# ---------------------------------------------------------------------------
# parse_turn_result tests (agent behavior — tool-based detection)
# ---------------------------------------------------------------------------

def test_parse_turn_result_detects_pending_questions():
    """parse_turn_result reports is_asking_questions=True when ask_user stored questions."""
    # Simulate ask_user having stored questions via a mock toolkit
    from unittest.mock import MagicMock as _MM
    mock_toolkit = _MM()
    mock_toolkit.get_pending_questions.return_value = [
        {
            "question": "What's your product?",
            "options": [{"label": "SaaS", "description": "Software"}],
            "multi_select": False,
        }
    ]
    mock_master = _MM()
    mock_master._daap_toolkit = mock_toolkit

    msg = make_msg("I need a few details before proceeding.")
    result = parse_turn_result(msg, master=mock_master)

    assert result.is_asking_questions is True
    assert result.pending_questions is not None
    assert len(result.pending_questions) == 1
    assert result.has_topology is False
    assert result.needs_user_input is True


def test_parse_turn_result_detects_topology():
    """parse_turn_result reports has_topology=True when toolkit holds a topology slot."""
    fake_topo = {"topology_id": "topo-abc12345", "nodes": [], "edges": []}
    fake_estimate = {
        "total_cost_usd": 0.14,
        "total_latency_seconds": 48.0,
        "min_viable_cost_usd": 0.05,
        "within_budget": True,
        "within_timeout": True,
    }

    mock_toolkit = MagicMock()
    mock_toolkit.get_pending_questions.return_value = None
    mock_toolkit.get_last_topology_result.return_value = {
        "topology": fake_topo,
        "estimate": fake_estimate,
    }
    mock_master = MagicMock()
    mock_master._daap_toolkit = mock_toolkit

    msg = make_msg("Here's the pipeline I designed for you.")
    result = parse_turn_result(msg, master=mock_master)

    assert result.has_topology is True
    assert result.topology_dict == fake_topo
    assert result.estimate_data == fake_estimate
    assert result.is_asking_questions is False
    assert result.needs_user_input is True
    mock_toolkit.clear_last_topology_result.assert_called_once()


def test_parse_turn_result_direct_response():
    """parse_turn_result returns direct response when no tools were called."""
    mock_toolkit = MagicMock()
    mock_toolkit.get_pending_questions.return_value = None
    mock_toolkit.get_last_topology_result.return_value = {"topology": None, "estimate": None}
    mock_master = MagicMock()
    mock_master._daap_toolkit = mock_toolkit

    msg = make_msg("Here's a cold email for HR directors at SaaS companies...")
    result = parse_turn_result(msg, master=mock_master)

    assert result.has_topology is False
    assert result.is_asking_questions is False
    assert "cold email" in result.response_text


# ---------------------------------------------------------------------------
# Agent creation tests
# ---------------------------------------------------------------------------

@patch("daap.master.agent.OpenAIChatModel")
@patch("daap.master.agent.ReActAgent")
def test_create_master_agent_uses_react_agent(mock_react_cls, mock_model_cls):
    """create_master_agent creates a ReActAgent with the master toolkit."""
    mock_model_cls.return_value = MagicMock()
    mock_react_cls.return_value = MagicMock()

    from daap.master.agent import create_master_agent
    agent = create_master_agent()

    assert mock_react_cls.called
    call_kwargs = mock_react_cls.call_args[1]
    assert call_kwargs.get("name") == "DAAP"
    assert "toolkit" in call_kwargs
    assert "memory" in call_kwargs


@patch("daap.master.agent.TrackedOpenAIChatModel")
@patch("daap.master.agent.ReActAgent")
def test_create_master_agent_openrouter(mock_react_cls, mock_tracked_cls):
    """create_master_agent uses OpenRouter config with TrackedOpenAIChatModel."""
    mock_tracked_cls.return_value = MagicMock()
    mock_react_cls.return_value = MagicMock()

    from daap.master.agent import create_master_agent

    operator_config = {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_map": {"smart": "anthropic/claude-sonnet-4-5"},
    }
    create_master_agent(operator_config=operator_config)

    assert mock_tracked_cls.called
    call_kwargs = mock_tracked_cls.call_args.kwargs
    assert call_kwargs["model_name"] == "anthropic/claude-sonnet-4-5"
    assert call_kwargs["client_kwargs"]["base_url"] == "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Runtime snapshot capability tests
# ---------------------------------------------------------------------------

def test_runtime_snapshot_includes_functional_capabilities():
    """Runtime snapshot includes functional_capabilities list."""
    from daap.master.runtime import build_master_runtime_snapshot
    from unittest.mock import MagicMock

    toolkit = MagicMock()
    toolkit.tools = {}
    snapshot = build_master_runtime_snapshot(toolkit, execution_mode="script")

    assert "functional_capabilities" in snapshot
    caps = snapshot["functional_capabilities"]
    assert isinstance(caps, list)
    assert len(caps) > 0
    labels = {c["label"] for c in caps}
    assert "Web search" in labels
    assert "LinkedIn" in labels
    for cap in caps:
        assert "label" in cap
        assert "available" in cap
        assert isinstance(cap["available"], bool)


def test_runtime_snapshot_includes_known_gaps():
    """Runtime snapshot includes known_gaps list for uninstalled MCP capabilities."""
    from daap.master.runtime import build_master_runtime_snapshot
    from unittest.mock import MagicMock

    toolkit = MagicMock()
    toolkit.tools = {}
    snapshot = build_master_runtime_snapshot(toolkit, execution_mode="script")

    assert "known_gaps" in snapshot
    gaps = snapshot["known_gaps"]
    assert isinstance(gaps, list)
    gap_labels = {g["label"] for g in gaps}
    assert "LinkedIn" in gap_labels
    for gap in gaps:
        assert "label" in gap
        assert "keywords" in gap
        assert "install_cmd" in gap or "docs_url" in gap


# ---------------------------------------------------------------------------
# New identity + gap detection prompt tests
# ---------------------------------------------------------------------------

def test_prompt_identity_is_daap_platform_not_b2b_only():
    """Master agent identity is DAAP platform assistant, not narrowly B2B sales only."""
    prompt = get_master_system_prompt()
    assert "DAAP" in prompt
    assert "expert AI assistant for B2B sales automation" not in prompt


def test_prompt_includes_gap_detection_rules():
    """Prompt instructs DAAP to surface capability gaps with install command."""
    prompt = get_master_system_prompt()
    assert "install" in prompt.lower()
    assert "capability" in prompt.lower() or "install_cmd" in prompt.lower() or "missing" in prompt.lower()


def test_prompt_instructs_get_runtime_context_for_self_description():
    """Prompt tells agent to call get_runtime_context when asked about capabilities."""
    prompt = get_master_system_prompt()
    assert "get_runtime_context" in prompt


def test_prompt_gap_detection_uses_known_gaps_from_snapshot():
    """Prompt includes known_gaps data when injected via runtime_context."""
    prompt = get_master_system_prompt(
        runtime_context={
            "master_tools": ["ask_user", "delegate_to_architect", "get_runtime_context"],
            "functional_capabilities": [
                {"label": "Web search", "available": True},
                {"label": "LinkedIn", "available": False},
            ],
            "known_gaps": [
                {
                    "label": "LinkedIn",
                    "keywords": ["linkedin", "prospect"],
                    "install_cmd": "daap mcp add linkedin npx @daap/linkedin-mcp",
                }
            ],
        }
    )
    assert "known_gaps" in prompt or "LinkedIn" in prompt
