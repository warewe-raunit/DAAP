"""
DAAP Memory Tests — pytest suite for memory/client.py, reader.py, writer.py

All tests mock mem0.Memory — no real API calls, no Qdrant, no OpenAI key needed.
"""

from unittest.mock import MagicMock, patch

import pytest

from daap.executor.engine import ExecutionResult, NodeResult


# ---------------------------------------------------------------------------
# Mock factory
# ---------------------------------------------------------------------------

def make_mock_mem0(search_results: list[dict] | None = None):
    """Return a MagicMock that mimics the mem0.Memory interface."""
    mock = MagicMock()
    mock.add.return_value = {"results": [{"id": "test-mem-1", "memory": "stored"}]}

    results = search_results if search_results is not None else [
        {"memory": "User sells project management SaaS", "id": "m1"},
        {"memory": "Target: mid-size construction companies", "id": "m2"},
    ]
    mock.search.return_value = {"results": results}
    mock.get_all.return_value = [{"memory": "User sells SaaS", "id": "m1"}]
    return mock


def make_daap_memory(mock_mem0=None):
    """Create a DaapMemory with injected mock mem0 instance."""
    from daap.memory.client import DaapMemory
    mem = DaapMemory.__new__(DaapMemory)
    mem.mem = mock_mem0 or make_mock_mem0()
    return mem


def make_execution_result(success: bool = True) -> ExecutionResult:
    return ExecutionResult(
        topology_id="topo-test0001",
        final_output="10 leads found and emails drafted.",
        node_results=[
            NodeResult(node_id="researcher", output_text="Company A, Company B...", latency_seconds=1.5),
            NodeResult(node_id="email_drafter", output_text="Dear John...", latency_seconds=0.8),
        ],
        total_latency_seconds=2.3,
        success=success,
        error=None if success else "Timeout on researcher node",
    )


# ---------------------------------------------------------------------------
# DaapMemory — user profile
# ---------------------------------------------------------------------------

def test_store_and_retrieve_user_profile():
    """store_user_profile calls mem.add with correct user_id and category."""
    mock_mem = make_mock_mem0()
    mem = make_daap_memory(mock_mem)

    mem.store_user_profile("user-001", "User sells SaaS to construction companies.")

    mock_mem.add.assert_called_once()
    call_kwargs = mock_mem.add.call_args[1]
    assert call_kwargs["user_id"] == "user-001"
    assert call_kwargs["metadata"]["category"] == "profile"


def test_store_and_retrieve_preferences():
    """store_user_preference calls mem.add with preferences category."""
    mock_mem = make_mock_mem0()
    mem = make_daap_memory(mock_mem)

    mem.store_user_preference("user-002", "User prefers short emails under 100 words.")

    mock_mem.add.assert_called_once()
    call_kwargs = mock_mem.add.call_args[1]
    assert call_kwargs["user_id"] == "user-002"
    assert call_kwargs["metadata"]["category"] == "preferences"


def test_get_user_context_returns_structured():
    """get_user_context returns dict with profile, preferences, recent_runs keys."""
    mock_mem = make_mock_mem0(search_results=[
        {"memory": "User sells project management SaaS", "id": "m1"},
    ])
    mem = make_daap_memory(mock_mem)

    ctx = mem.get_user_context("user-003", query="SaaS construction")

    assert "profile" in ctx
    assert "preferences" in ctx
    assert "recent_runs" in ctx
    assert isinstance(ctx["profile"], list)
    assert isinstance(ctx["preferences"], list)
    assert isinstance(ctx["recent_runs"], list)


def test_get_user_context_empty_for_new_user():
    """get_user_context with no search results → all lists empty."""
    mock_mem = make_mock_mem0(search_results=[])
    mem = make_daap_memory(mock_mem)

    ctx = mem.get_user_context("brand-new-user")

    assert ctx["profile"] == []
    assert ctx["preferences"] == []
    assert ctx["recent_runs"] == []


# ---------------------------------------------------------------------------
# DaapMemory — run history
# ---------------------------------------------------------------------------

def test_store_run_result():
    """store_run_result calls mem.add with runs category and run_id."""
    mock_mem = make_mock_mem0()
    mem = make_daap_memory(mock_mem)

    mem.store_run_result(
        user_id="user-004",
        run_summary="Completed pipeline. Found 10 leads. Cost: $0.11.",
        run_id="topo-abc12345",
    )

    mock_mem.add.assert_called_once()
    call_kwargs = mock_mem.add.call_args[1]
    assert call_kwargs["user_id"] == "user-004"
    assert call_kwargs["run_id"] == "topo-abc12345"
    assert call_kwargs["metadata"]["category"] == "runs"


# ---------------------------------------------------------------------------
# DaapMemory — agent diary
# ---------------------------------------------------------------------------

def test_store_agent_learning():
    """store_agent_learning scopes to agent_id daap_{role}."""
    mock_mem = make_mock_mem0()
    mem = make_daap_memory(mock_mem)

    mem.store_agent_learning(
        "researcher",
        "Keyword 'field operations management' works better than 'construction PM'.",
    )

    mock_mem.add.assert_called_once()
    call_kwargs = mock_mem.add.call_args[1]
    assert call_kwargs["agent_id"] == "daap_researcher"
    assert call_kwargs["metadata"]["category"] == "agent_diary"


def test_agent_learnings_scoped_by_role():
    """get_agent_learnings searches by agent_id — different roles don't mix."""
    mock_mem = make_mock_mem0(search_results=[
        {"memory": "Use 'field operations' as search keyword", "id": "r1"},
    ])
    mem = make_daap_memory(mock_mem)

    learnings = mem.get_agent_learnings("researcher", "construction lead search")

    mock_mem.search.assert_called_once()
    call_kwargs = mock_mem.search.call_args[1]
    assert call_kwargs["agent_id"] == "daap_researcher"
    assert len(learnings) == 1
    assert "field operations" in learnings[0]


def test_user_memories_scoped_by_user():
    """Separate user_id values produce separate search calls with correct user_id."""
    mock_mem = make_mock_mem0()
    mem = make_daap_memory(mock_mem)

    mem.get_user_context("user-A", query="test")
    mem.get_user_context("user-B", query="test")

    # Each context call makes 3 searches — verify user_id isolation
    all_calls = mock_mem.search.call_args_list
    user_ids_used = {call[1]["user_id"] for call in all_calls}
    assert "user-A" in user_ids_used
    assert "user-B" in user_ids_used


# ---------------------------------------------------------------------------
# reader.py
# ---------------------------------------------------------------------------

def test_format_user_context_for_prompt():
    """format_user_context_for_prompt produces a readable string with sections."""
    from daap.memory.reader import format_user_context_for_prompt

    context = {
        "profile": ["User sells SaaS", "Target: construction companies"],
        "preferences": ["Prefer short emails"],
        "recent_runs": ["Last run found 10 leads"],
    }

    result = format_user_context_for_prompt(context)

    assert "User Profile" in result
    assert "User Preferences" in result
    assert "Recent Run History" in result
    assert "construction companies" in result
    assert "short emails" in result


def test_load_agent_context_for_node_empty():
    """load_agent_context_for_node returns empty string when no learnings."""
    from daap.memory.reader import load_agent_context_for_node

    mock_mem = make_mock_mem0(search_results=[])
    mem = make_daap_memory(mock_mem)

    result = load_agent_context_for_node(mem, "researcher", "find leads")

    assert result == ""


def test_load_user_context_returns_none_for_new_user():
    """load_user_context_for_master returns None when user has no memories."""
    from daap.memory.reader import load_user_context_for_master

    mock_mem = make_mock_mem0(search_results=[])
    mem = make_daap_memory(mock_mem)

    result = load_user_context_for_master(mem, "brand-new-user", "I need leads")

    assert result is None


# ---------------------------------------------------------------------------
# writer.py
# ---------------------------------------------------------------------------

def test_write_run_to_memory_formats_correctly():
    """write_run_to_memory stores a summary containing key metrics."""
    from daap.memory.writer import write_run_to_memory

    mock_mem = make_mock_mem0()
    mem = make_daap_memory(mock_mem)
    result = make_execution_result(success=True)

    write_run_to_memory(
        memory=mem,
        user_id="user-005",
        topology_summary="4-agent sales pipeline for construction SaaS",
        execution_result=result,
        user_rating=4,
        user_comment="Emails were slightly too long.",
    )

    mock_mem.add.assert_called_once()
    stored_text = mock_mem.add.call_args[0][0]
    assert "successful" in stored_text
    assert "4/5" in stored_text
    assert "Emails were slightly too long" in stored_text
    assert "2" in stored_text  # latency seconds


def test_write_user_feedback():
    """write_user_feedback calls store_user_rating with correct user_id."""
    from daap.memory.writer import write_user_feedback

    mock_mem = make_mock_mem0()
    mem = make_daap_memory(mock_mem)

    write_user_feedback(mem, "user-006", "User rated run 5/5. Great results.")

    mock_mem.add.assert_called_once()
    call_kwargs = mock_mem.add.call_args[1]
    assert call_kwargs["user_id"] == "user-006"
    assert call_kwargs["metadata"]["category"] == "feedback"


def test_write_agent_learnings_from_run():
    """write_agent_learnings_from_run stores a learning per node result."""
    from daap.memory.writer import write_agent_learnings_from_run

    mock_mem = make_mock_mem0()
    mem = make_daap_memory(mock_mem)
    result = make_execution_result()

    topology_nodes = [
        {"node_id": "researcher", "role": "Lead Researcher"},
        {"node_id": "email_drafter", "role": "Email Writer"},
    ]

    write_agent_learnings_from_run(mem, result, topology_nodes)

    # One add() call per node result
    assert mock_mem.add.call_count == 2

    all_agent_ids = {call[1]["agent_id"] for call in mock_mem.add.call_args_list}
    assert "daap_researcher" in all_agent_ids
    assert "daap_writer" in all_agent_ids
