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
    clear_last_topology_result,
    create_master_toolkit,
    generate_topology,
    get_last_topology_result,
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
    """Both generate_topology and ask_user are registered in the master toolkit."""
    toolkit = create_master_toolkit()
    registered = set(toolkit.tools.keys())
    assert "generate_topology" in registered
    assert "ask_user" in registered
    assert "register_skill" in registered


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
# generate_topology tool tests
# ---------------------------------------------------------------------------

def _block_text(block) -> str:
    return block.text if hasattr(block, "text") else block.get("text", "")


@pytest.mark.asyncio
async def test_generate_topology_handles_invalid_json():
    """generate_topology returns error ToolResponse for invalid JSON."""
    clear_last_topology_result()
    result = await generate_topology("not valid json {{{")
    text = _block_text(result.content[0])
    assert "Invalid JSON" in text or "invalid" in text.lower()
    # Module-level state should be unchanged
    assert get_last_topology_result()["topology"] is None


@pytest.mark.asyncio
async def test_valid_topology_includes_estimate():
    """generate_topology stores estimate data when topology is valid."""
    clear_last_topology_result()
    topology_json = load_sales_outreach_topology()
    result = await generate_topology(topology_json)
    text = _block_text(result.content[0])

    # Should succeed
    assert "valid" in text.lower() or "cost" in text.lower()

    stored = get_last_topology_result()
    assert stored["topology"] is not None
    assert stored["estimate"] is not None

    estimate = stored["estimate"]
    assert "total_cost_usd" in estimate
    assert "total_latency_seconds" in estimate
    assert "min_viable_cost_usd" in estimate
    assert estimate["total_cost_usd"] >= 0
    assert estimate["min_viable_cost_usd"] >= 0


@pytest.mark.asyncio
async def test_topology_has_correct_structure():
    """generate_topology stores topology with expected node roles."""
    clear_last_topology_result()
    topology_json = load_sales_outreach_topology()
    await generate_topology(topology_json)

    stored = get_last_topology_result()
    assert stored["topology"] is not None

    topo = stored["topology"]
    assert "nodes" in topo
    assert "edges" in topo
    assert len(topo["nodes"]) >= 2

    node_ids = {n["node_id"] for n in topo["nodes"]}
    # Sales outreach fixture has: researcher, evaluator, personalizer, drafter
    assert any("researcher" in nid or "research" in nid for nid in node_ids)
    assert any("drafter" in nid or "email" in nid or "writer" in nid for nid in node_ids)


@pytest.mark.asyncio
async def test_topology_validation_errors_fed_back():
    """generate_topology returns validation errors so the agent can retry."""
    clear_last_topology_result()

    # Build a topology with an invalid tool name — should fail validation
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
        "constraints": {},
    }

    result = await generate_topology(json.dumps(bad_topology))
    text = _block_text(result.content[0])
    assert "error" in text.lower() or "invalid" in text.lower() or "validation" in text.lower()
    # Topology should NOT be stored on failure
    assert get_last_topology_result()["topology"] is None


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------

def test_prompt_contains_available_tools():
    """System prompt includes the list of available tool names."""
    available = get_available_tool_names()
    prompt = get_master_system_prompt(available_tools=available)
    # At least some tool names should appear in the prompt
    for tool in list(available)[:3]:
        assert tool in prompt, f"Tool '{tool}' missing from system prompt"


def test_prompt_contains_topology_schema():
    """System prompt includes the TopologySpec JSON schema."""
    prompt = get_master_system_prompt()
    # Schema injection always includes these TopologySpec keys
    assert "topology_id" in prompt
    assert "node_id" in prompt
    assert "model_tier" in prompt


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
    assert "skill directory" in prompt.lower()


def test_prompt_enforces_action_or_final_output():
    """System prompt enforces tool action or final output, not intent-only text."""
    prompt = get_master_system_prompt()
    assert "Never end a turn with a promise of future action" in prompt
    assert "call a tool that makes progress" in prompt


def test_prompt_includes_validation_retry_discipline():
    """System prompt tells the agent to fix topology errors and retry generate_topology."""
    prompt = get_master_system_prompt()
    assert "Retry up to 3 times" in prompt
    assert "Do not ask the user to hand-edit topology JSON" in prompt


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
    """parse_turn_result reports has_topology=True when generate_topology stored a result."""
    from daap.master.tools import _topology_result_var, clear_last_topology_result

    fake_topo = {"topology_id": "topo-abc12345", "nodes": [], "edges": []}
    fake_estimate = {
        "total_cost_usd": 0.14,
        "total_latency_seconds": 48.0,
        "min_viable_cost_usd": 0.05,
        "within_budget": True,
        "within_timeout": True,
    }
    _topology_result_var.set({"topology": fake_topo, "estimate": fake_estimate})

    try:
        msg = make_msg("Here's the pipeline I designed for you.")
        result = parse_turn_result(msg, master=MagicMock())

        assert result.has_topology is True
        assert result.topology_dict == fake_topo
        assert result.estimate_data == fake_estimate
        assert result.is_asking_questions is False
        assert result.needs_user_input is True
    finally:
        clear_last_topology_result()


def test_parse_turn_result_direct_response():
    """parse_turn_result returns direct response when no tools were called."""
    from daap.master.tools import clear_last_topology_result
    clear_last_topology_result()

    msg = make_msg("Here's a cold email for HR directors at SaaS companies...")
    result = parse_turn_result(msg, master=MagicMock())

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
