"""Unit tests for topology agent tools (load, persist, rerun)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentscope.tool import ToolResponse

from daap.api.sessions import Session, create_session_scoped_toolkit
from daap.topology.store import TopologyStore
from daap.tools.token_tracker import TokenTracker


SAMPLE_SPEC = {
    "topology_id": "topo-abc12345",
    "version": 1,
    "created_at": "2026-04-14T00:00:00Z",
    "user_prompt": "find b2b leads",
    "nodes": [],
    "edges": [],
    "constraints": {},
}


def _block_text(block) -> str:
    if hasattr(block, "text"):
        return block.text
    if isinstance(block, dict):
        return block.get("text", "")
    return str(block)


@pytest.fixture
def store(tmp_path):
    topology_store = TopologyStore(db_path=str(tmp_path / "test.db"))
    topology_store.save_topology(SAMPLE_SPEC, user_id="user-1")
    return topology_store


@pytest.fixture
def session():
    sess = Session(session_id="sess-001", created_at=0.0, user_id="user-1")
    sess.token_tracker = TokenTracker()
    return sess


@pytest.fixture
def toolkit(session, store):
    return create_session_scoped_toolkit(session, topology_store=store)


@pytest.mark.asyncio
async def test_load_topology_sets_pending(session, toolkit):
    tool_fn = toolkit.tools["load_topology"].original_func
    result = await tool_fn(topology_id="topo-abc12345")

    assert isinstance(result, ToolResponse)
    assert session.pending_topology is not None
    assert session.pending_topology["topology_id"] == "topo-abc12345"
    assert "loaded" in _block_text(result.content[0]).lower()


@pytest.mark.asyncio
async def test_load_topology_not_found(session, toolkit):
    tool_fn = toolkit.tools["load_topology"].original_func
    result = await tool_fn(topology_id="topo-nope")

    assert "not found" in _block_text(result.content[0]).lower()
    assert session.pending_topology is None


@pytest.mark.asyncio
async def test_persist_topology_saves(session, toolkit):
    session.pending_topology = SAMPLE_SPEC
    tool_fn = toolkit.tools["persist_topology"].original_func
    result = await tool_fn(topology_id="topo-abc12345", save_mode="overwrite")

    assert isinstance(result, ToolResponse)
    assert "saved" in _block_text(result.content[0]).lower()


@pytest.mark.asyncio
async def test_persist_topology_no_pending(toolkit):
    tool_fn = toolkit.tools["persist_topology"].original_func
    result = await tool_fn(topology_id="topo-abc12345", save_mode="overwrite")

    assert "no pending topology" in _block_text(result.content[0]).lower()


@pytest.mark.asyncio
async def test_rerun_topology_not_found(toolkit):
    tool_fn = toolkit.tools["rerun_topology"].original_func
    result = await tool_fn(topology_id="topo-nope")

    assert "not found" in _block_text(result.content[0]).lower()


@pytest.mark.asyncio
async def test_rerun_topology_saves_run(session, store, toolkit):
    tool_fn = toolkit.tools["rerun_topology"].original_func
    resolved = SimpleNamespace(nodes=[], execution_order=[[]])
    run_result = SimpleNamespace(
        topology_id="topo-abc12345",
        final_output="ok",
        success=True,
        error=None,
        total_latency_seconds=2.0,
        total_input_tokens=10,
        total_output_tokens=20,
    )

    with (
        patch("daap.spec.schema.TopologySpec.model_validate", return_value=MagicMock()),
        patch("daap.spec.resolver.resolve_topology", return_value=resolved),
        patch("daap.executor.engine.execute_topology", new=AsyncMock(return_value=run_result)),
    ):
        result = await tool_fn(topology_id="topo-abc12345", user_prompt="run again")

    assert "rerun complete" in _block_text(result.content[0]).lower()
    runs = store.get_runs("topo-abc12345")
    assert len(runs) == 1
    assert session.execution_result is not None
