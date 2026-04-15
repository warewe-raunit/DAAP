"""
Tests for P1 memory wiring — verifies DaapMemory is threaded correctly
through the execution stack.

Strategy: avoid importing agentscope-dependent modules (engine, sessions).
Use source-code inspection for signature checks, and test memory
read/write functions directly (they have no agentscope dependency).
"""

import ast
import importlib.util


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_daap_memory(search_results=None):
    from daap.memory.client import DaapMemory
    from unittest.mock import MagicMock
    mock_mem = MagicMock()
    mock_mem.add.return_value = {"results": []}
    mock_mem.search.return_value = {
        "results": search_results or [
            {"memory": "User sells project management SaaS", "id": "m1"},
        ]
    }
    mem = DaapMemory.__new__(DaapMemory)
    mem.mem = mock_mem
    return mem, mock_mem


def _get_source(module_dotpath: str) -> str:
    spec = importlib.util.find_spec(module_dotpath)
    return open(spec.origin).read()


def _function_params(source: str, func_name: str) -> list[str]:
    """Parse function signature from source and return param names."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                params = [arg.arg for arg in node.args.args]
                params += [arg.arg for arg in node.args.kwonlyargs]
                if node.args.vararg:
                    params.append(node.args.vararg.arg)
                if node.args.kwarg:
                    params.append(node.args.kwarg.arg)
                return params
    return []


# ---------------------------------------------------------------------------
# engine.py — daap_memory param present in execute_topology
# ---------------------------------------------------------------------------

def test_execute_topology_has_daap_memory_param():
    """execute_topology must declare a daap_memory parameter."""
    source = _get_source("daap.executor.engine")
    params = _function_params(source, "execute_topology")
    assert "daap_memory" in params, (
        f"execute_topology is missing daap_memory param. Found: {params}"
    )


def test_execute_topology_passes_memory_to_build_node():
    """engine.py must pass daap_memory= when calling build_node."""
    source = _get_source("daap.executor.engine")
    assert "daap_memory=daap_memory" in source, (
        "engine.py must pass daap_memory=daap_memory to build_node()"
    )


# ---------------------------------------------------------------------------
# node_builder.py — daap_memory param and enrichment logic present
# ---------------------------------------------------------------------------

def test_build_node_has_daap_memory_param():
    """build_node must declare a daap_memory parameter."""
    source = _get_source("daap.executor.node_builder")
    params = _function_params(source, "build_node")
    assert "daap_memory" in params, (
        f"build_node is missing daap_memory param. Found: {params}"
    )


def test_build_node_calls_load_agent_context():
    """build_node source must call load_agent_context_for_node."""
    source = _get_source("daap.executor.node_builder")
    assert "load_agent_context_for_node" in source, (
        "build_node must call load_agent_context_for_node for prompt enrichment"
    )


# ---------------------------------------------------------------------------
# sessions.py — daap_memory param present, memory writes present
# ---------------------------------------------------------------------------

def test_create_session_scoped_toolkit_has_daap_memory_param():
    """create_session_scoped_toolkit must declare a daap_memory parameter."""
    source = _get_source("daap.api.sessions")
    params = _function_params(source, "create_session_scoped_toolkit")
    assert "daap_memory" in params, (
        f"create_session_scoped_toolkit missing daap_memory param. Found: {params}"
    )


def test_sessions_has_logger():
    """sessions.py must define a module-level logger (fixes runtime crash)."""
    source = _get_source("daap.api.sessions")
    assert "logger = logging.getLogger" in source, (
        "sessions.py must define logger = logging.getLogger(__name__)"
    )


def test_execute_pending_topology_calls_write_run_to_memory():
    """execute_pending_topology must call write_run_to_memory after success."""
    source = _get_source("daap.api.sessions")
    assert "write_run_to_memory" in source, (
        "execute_pending_topology must call write_run_to_memory post-execution"
    )


def test_execute_pending_topology_calls_write_agent_learnings():
    """execute_pending_topology must call write_agent_learnings_from_run."""
    source = _get_source("daap.api.sessions")
    assert "write_agent_learnings_from_run" in source, (
        "execute_pending_topology must call write_agent_learnings_from_run"
    )


def test_execute_pending_topology_passes_memory_to_engine():
    """execute_pending_topology must pass daap_memory= to execute_topology call."""
    source = _get_source("daap.api.sessions")
    assert "daap_memory=daap_memory" in source, (
        "execute_pending_topology must forward daap_memory to execute_topology()"
    )


# ---------------------------------------------------------------------------
# routes.py — memory passed to create_session_scoped_toolkit
# ---------------------------------------------------------------------------

def test_routes_passes_memory_to_toolkit():
    """routes.py must pass daap_memory= to create_session_scoped_toolkit."""
    source = _get_source("daap.api.routes")
    assert "daap_memory=memory" in source, (
        "routes.py must pass daap_memory=memory to create_session_scoped_toolkit()"
    )


def test_routes_models_endpoint_no_deprecated_models():
    """routes.py /models endpoint must not reference deprecated gemini-2.0-flash-001 as default."""
    source = _get_source("daap.api.routes")
    # Ensure it references MODEL_REGISTRY (dynamic) not hardcoded stale ID as default
    assert "MODEL_REGISTRY" in source, (
        "routes.py must use MODEL_REGISTRY for /models endpoint defaults"
    )


# ---------------------------------------------------------------------------
# Memory reader / writer — functional tests (no agentscope dependency)
# ---------------------------------------------------------------------------

def test_load_agent_context_returns_formatted_learnings():
    """load_agent_context_for_node formats learnings into a readable block."""
    from daap.memory.reader import load_agent_context_for_node

    daap_memory, _ = make_daap_memory(search_results=[
        {"memory": "Use keyword 'field operations' for construction searches", "id": "r1"},
    ])

    result = load_agent_context_for_node(daap_memory, "researcher", "construction leads")

    assert "Learnings from past runs" in result
    assert "field operations" in result


def test_load_agent_context_empty_when_no_learnings():
    """load_agent_context_for_node returns empty string when no results."""
    from daap.memory.reader import load_agent_context_for_node
    from unittest.mock import MagicMock
    from daap.memory.client import DaapMemory

    mock_mem = MagicMock()
    mock_mem.search.return_value = {"results": []}
    mem = DaapMemory.__new__(DaapMemory)
    mem.mem = mock_mem

    result = load_agent_context_for_node(mem, "researcher", "test")
    assert result == ""


def test_load_user_context_returns_none_for_new_user():
    """load_user_context_for_master returns None when no memories exist."""
    from daap.memory.reader import load_user_context_for_master
    from unittest.mock import MagicMock
    from daap.memory.client import DaapMemory

    mock_mem = MagicMock()
    mock_mem.search.return_value = {"results": []}
    mem = DaapMemory.__new__(DaapMemory)
    mem.mem = mock_mem

    result = load_user_context_for_master(mem, "brand-new-user", "")
    assert result is None


def test_load_user_context_returns_dict_for_existing_user():
    """load_user_context_for_master returns dict when memories found."""
    from daap.memory.reader import load_user_context_for_master

    daap_memory, _ = make_daap_memory(search_results=[
        {"memory": "User sells SaaS", "id": "m1"},
    ])
    result = load_user_context_for_master(daap_memory, "existing-user", "leads")
    assert result is not None
    assert "profile" in result


def test_write_run_to_memory_stores_outcome():
    """write_run_to_memory stores a summary with success/failure info."""
    from daap.memory.writer import write_run_to_memory
    from types import SimpleNamespace

    daap_memory, mock_mem = make_daap_memory()

    result = SimpleNamespace(
        topology_id="topo-abc00001",
        final_output="10 leads found.",
        node_results=[],
        total_latency_seconds=2.3,
        success=True,
        error=None,
        total_input_tokens=100,
        total_output_tokens=50,
    )

    write_run_to_memory(
        memory=daap_memory,
        user_id="user-001",
        topology_summary="Find B2B leads",
        execution_result=result,
    )

    mock_mem.add.assert_called_once()
    stored_text = mock_mem.add.call_args[0][0]
    assert "successful" in stored_text


def test_write_run_to_memory_stores_failure():
    """write_run_to_memory stores 'failed' for unsuccessful runs."""
    from daap.memory.writer import write_run_to_memory
    from types import SimpleNamespace

    daap_memory, mock_mem = make_daap_memory()

    result = SimpleNamespace(
        topology_id="topo-abc00001",
        final_output="",
        node_results=[],
        total_latency_seconds=0.5,
        success=False,
        error="Timeout on researcher node",
        total_input_tokens=10,
        total_output_tokens=0,
    )

    write_run_to_memory(
        memory=daap_memory,
        user_id="user-001",
        topology_summary="Find leads",
        execution_result=result,
    )

    stored_text = mock_mem.add.call_args[0][0]
    assert "failed" in stored_text


def test_write_agent_learnings_stores_per_node():
    """write_agent_learnings_from_run stores one learning per node result."""
    from daap.memory.writer import write_agent_learnings_from_run
    from types import SimpleNamespace

    daap_memory, mock_mem = make_daap_memory()

    result = SimpleNamespace(
        topology_id="topo-abc00001",
        node_results=[
            SimpleNamespace(node_id="researcher", output_text="Company A, B, C", latency_seconds=1.5),
            SimpleNamespace(node_id="email_drafter", output_text="Dear John...", latency_seconds=0.8),
        ],
    )
    topology_nodes = [
        {"node_id": "researcher", "role": "Lead Researcher"},
        {"node_id": "email_drafter", "role": "Email Writer"},
    ]

    write_agent_learnings_from_run(daap_memory, result, topology_nodes)

    assert mock_mem.add.call_count == 2
    agent_ids = {call[1]["agent_id"] for call in mock_mem.add.call_args_list}
    assert "daap_researcher" in agent_ids
    assert "daap_writer" in agent_ids


def test_format_user_context_none_returns_empty():
    """format_user_context_for_prompt returns empty string for None or empty dict."""
    from daap.memory.reader import format_user_context_for_prompt
    assert format_user_context_for_prompt(None) == ""
    assert format_user_context_for_prompt({}) == ""


def test_format_user_context_renders_all_sections():
    """format_user_context_for_prompt renders profile, preferences, recent_runs."""
    from daap.memory.reader import format_user_context_for_prompt

    ctx = {
        "profile": ["User sells SaaS"],
        "preferences": ["Prefer short emails"],
        "recent_runs": ["Found 10 leads last run"],
    }
    result = format_user_context_for_prompt(ctx)
    assert "User Profile" in result
    assert "User Preferences" in result
    assert "Recent Run History" in result
