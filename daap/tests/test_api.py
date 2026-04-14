"""
DAAP API Tests — pytest suite for api/routes.py, api/sessions.py, api/ws_handler.py

Uses FastAPI TestClient. Master agent is mocked — no real API calls.
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentscope.message import Msg
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from daap.api.routes import app
from daap.api.sessions import Session, SessionManager, create_session_scoped_toolkit
import daap.api.routes as routes_module
from daap.feedback.store import FeedbackStore


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def mock_agent_returning(text: str = "Mock agent response"):
    """AsyncMock that returns a Msg with the given text content."""
    return AsyncMock(
        return_value=Msg(name="DAAP", content=text, role="assistant")
    )


@pytest.fixture(autouse=True)
def reset_globals(tmp_path):
    """Fresh SessionManager and FeedbackStore (temp DB) per test."""
    routes_module.session_manager = SessionManager()
    routes_module.feedback_store = FeedbackStore(db_path=str(tmp_path / "test.db"))
    yield


@pytest.fixture()
def client():
    return TestClient(app)


def _create_session_with_mock_agent(client, response_text="Mock response"):
    """POST /session with mocked master agent. Returns session_id."""
    mock_agent = mock_agent_returning(response_text)
    with patch("daap.api.routes.create_master_agent_with_toolkit", return_value=mock_agent):
        resp = client.post("/session")
    assert resp.status_code == 200
    return resp.json()["session_id"], mock_agent


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    """GET /health returns 200 with status ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_create_session(client):
    """POST /session returns a session_id."""
    mock_agent = mock_agent_returning()
    with patch("daap.api.routes.create_master_agent_with_toolkit", return_value=mock_agent):
        resp = client.post("/session")
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert len(data["session_id"]) > 0


def test_create_session_with_model_selection(client):
    """POST /session forwards master model config and stores subagent config on session."""
    mock_agent = mock_agent_returning()
    with patch("daap.api.routes.create_master_agent_with_toolkit", return_value=mock_agent) as mock_create:
        resp = client.post(
            "/session"
            "?user_id=test-user"
            "&master_model=anthropic/claude-opus-4-6"
            "&subagent_fast_model=anthropic/claude-haiku-4-5-20251001"
            "&subagent_smart_model=anthropic/claude-sonnet-4-6"
            "&subagent_powerful_model=anthropic/claude-opus-4-6"
        )

    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["operator_config"]["model_map"]["smart"] == "anthropic/claude-opus-4-6"

    session = routes_module.session_manager.get_session(session_id)
    assert session is not None
    assert session.subagent_operator_config is not None
    model_map = session.subagent_operator_config["model_map"]
    assert model_map["fast"] == "anthropic/claude-haiku-4-5-20251001"
    assert model_map["smart"] == "anthropic/claude-sonnet-4-6"
    assert model_map["powerful"] == "anthropic/claude-opus-4-6"


def test_get_session_config_no_session(client):
    """GET /session/{session_id}/config returns 404 for unknown sessions."""
    resp = client.get("/session/nonexistent/config")
    assert resp.status_code == 404


def test_get_session_config_returns_selected_models(client):
    """GET /session/{session_id}/config returns stored model selections."""
    mock_agent = mock_agent_returning()
    with patch("daap.api.routes.create_master_agent_with_toolkit", return_value=mock_agent):
        create_resp = client.post(
            "/session"
            "?user_id=test-user"
            "&master_model=anthropic/claude-sonnet-4-6"
            "&subagent_fast_model=anthropic/claude-haiku-4-5-20251001"
            "&subagent_smart_model=anthropic/claude-sonnet-4-6"
        )

    assert create_resp.status_code == 200
    session_id = create_resp.json()["session_id"]

    resp = client.get(f"/session/{session_id}/config")
    assert resp.status_code == 200
    data = resp.json()

    assert data["session_id"] == session_id
    assert data["user_id"] == "test-user"
    assert data["master_operator_config"]["model_map"]["smart"] == "anthropic/claude-sonnet-4-6"
    sub_map = data["subagent_operator_config"]["model_map"]
    assert sub_map["fast"] == "anthropic/claude-haiku-4-5-20251001"
    assert sub_map["smart"] == "anthropic/claude-sonnet-4-6"


def test_list_sessions(client):
    """GET /sessions lists created sessions."""
    mock_agent = mock_agent_returning()
    with patch("daap.api.routes.create_master_agent_with_toolkit", return_value=mock_agent):
        client.post("/session")
        client.post("/session")

    resp = client.get("/sessions")
    assert resp.status_code == 200
    sessions = resp.json()["sessions"]
    assert len(sessions) == 2


def test_delete_session(client):
    """DELETE /session/{id} removes the session."""
    session_id, _ = _create_session_with_mock_agent(client)

    del_resp = client.delete(f"/session/{session_id}")
    assert del_resp.status_code == 200

    # Session should be gone
    resp = client.get("/sessions")
    session_ids = [s["session_id"] for s in resp.json()["sessions"]]
    assert session_id not in session_ids


def test_get_topology_no_session(client):
    """GET /topology/nonexistent returns 404."""
    resp = client.get("/topology/nonexistent")
    assert resp.status_code == 404


def test_get_topology_empty(client):
    """GET /topology/{session_id} returns null topology when none generated."""
    session_id, _ = _create_session_with_mock_agent(client)
    resp = client.get(f"/topology/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["topology"] is None


def test_rate_run_no_result(client):
    """POST /rate on a session with no execution result returns 400."""
    session_id, _ = _create_session_with_mock_agent(client)
    resp = client.post("/rate", json={
        "session_id": session_id,
        "rating": 5,
        "comment": "Great!",
    })
    assert resp.status_code == 400


def test_rate_run_session_not_found(client):
    """POST /rate on a nonexistent session returns 404."""
    resp = client.post("/rate", json={
        "session_id": "nonexistent",
        "rating": 4,
    })
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# WebSocket tests
# ---------------------------------------------------------------------------

def test_websocket_invalid_session(client):
    """WebSocket to /ws/nonexistent is rejected (connection closed by server)."""
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/nonexistent") as ws:
            # Server closes before we can send — should raise on connect or first receive
            ws.receive_json()


def test_websocket_connect(client):
    """Valid session connects successfully — cancel immediately accepted."""
    session_id, _ = _create_session_with_mock_agent(client)

    with client.websocket_connect(f"/ws/{session_id}") as ws:
        ws.send_json({"type": "cancel"})
        data = ws.receive_json()
        assert data["type"] == "response"
        assert "Cancelled" in data["content"]


def test_websocket_cancel(client):
    """Cancel message clears pending topology and sends acknowledgement."""
    session_id, _ = _create_session_with_mock_agent(client)

    # Manually inject a pending topology into the session
    session = routes_module.session_manager.get_session(session_id)
    session.pending_topology = {"topology_id": "topo-abc12345"}
    session.pending_estimate = {"total_cost_usd": 0.10}

    with client.websocket_connect(f"/ws/{session_id}") as ws:
        ws.send_json({"type": "cancel"})
        data = ws.receive_json()

    assert data["type"] == "response"
    assert session.pending_topology is None
    assert session.pending_estimate is None


def test_websocket_send_message(client):
    """User message is forwarded to master agent and response returned."""
    session_id, mock_agent = _create_session_with_mock_agent(
        client, response_text="Here is your cold email draft."
    )

    with client.websocket_connect(f"/ws/{session_id}") as ws:
        ws.send_json({"type": "message", "content": "Write me a cold email."})
        data = ws.receive_json()

    assert data["type"] == "response"
    assert "cold email" in data["content"].lower()
    assert mock_agent.called


def test_websocket_unknown_message_type(client):
    """Unknown message type returns an error response."""
    session_id, _ = _create_session_with_mock_agent(client)

    with client.websocket_connect(f"/ws/{session_id}") as ws:
        ws.send_json({"type": "bogus_type"})
        data = ws.receive_json()

    assert data["type"] == "error"
    assert "bogus_type" in data["message"]


def test_websocket_approve_no_topology(client):
    """Approve with no pending topology returns an error."""
    session_id, _ = _create_session_with_mock_agent(client)

    with client.websocket_connect(f"/ws/{session_id}") as ws:
        ws.send_json({"type": "approve"})
        data = ws.receive_json()

    assert data["type"] == "error"
    assert "No topology" in data["message"]


def test_websocket_text_approve_executes_pending_topology(client):
    """Text message 'approve' executes pending topology without special message type."""
    session_id, _ = _create_session_with_mock_agent(client)
    session = routes_module.session_manager.get_session(session_id)
    session.pending_topology = {
        "topology_id": "topo-abc12345",
        "user_prompt": "Run outreach pipeline",
        "nodes": [],
    }

    resolved = SimpleNamespace(
        nodes=[SimpleNamespace(node_id="node_a", concrete_model_id="google/gemini-2.0-flash-001")],
        execution_order=[["node_a"]],
    )
    exec_result = SimpleNamespace(
        topology_id="topo-abc12345",
        final_output="Execution complete.",
        success=True,
        error=None,
        total_latency_seconds=1.2,
        models_used=["anthropic/claude-sonnet-4-6"],
        total_input_tokens=10,
        total_output_tokens=5,
        node_results=[],
    )

    with (
        patch("daap.api.ws_handler.TopologySpec.model_validate", return_value=MagicMock()),
        patch("daap.api.ws_handler.resolve_topology", return_value=resolved),
        patch("daap.api.ws_handler.execute_topology", new=AsyncMock(return_value=exec_result)) as mock_exec,
    ):
        with client.websocket_connect(f"/ws/{session_id}") as ws:
            ws.send_json({"type": "message", "content": "approve"})
            first = ws.receive_json()
            second = ws.receive_json()

    assert first["type"] == "executing"
    assert first["topology_id"] == "topo-abc12345"
    assert second["type"] == "result"
    assert "Execution complete" in second["output"]
    assert mock_exec.await_count == 1


def test_websocket_text_approve_with_mismatched_topology_id(client):
    """Text approval with explicit wrong topology id is rejected."""
    session_id, _ = _create_session_with_mock_agent(client)
    session = routes_module.session_manager.get_session(session_id)
    session.pending_topology = {
        "topology_id": "topo-abc12345",
        "user_prompt": "Run outreach pipeline",
        "nodes": [],
    }

    with client.websocket_connect(f"/ws/{session_id}") as ws:
        ws.send_json({"type": "message", "content": "approve topo-deadbeef"})
        data = ws.receive_json()

    assert data["type"] == "error"
    assert "does not match" in data["message"]


def test_websocket_execution_streams_progress_events(client):
    """Execution emits per-node progress updates before final result."""
    session_id, _ = _create_session_with_mock_agent(client)
    session = routes_module.session_manager.get_session(session_id)
    session.pending_topology = {
        "topology_id": "topo-abc12345",
        "user_prompt": "Run outreach pipeline",
        "nodes": [],
    }

    resolved = SimpleNamespace(
        nodes=[
            SimpleNamespace(node_id="node_a", concrete_model_id="google/gemini-2.0-flash-001"),
            SimpleNamespace(node_id="node_b", concrete_model_id="google/gemini-2.0-flash-001"),
        ],
        execution_order=[["node_a"], ["node_b"]],
    )

    async def _fake_execute(
        resolved,
        user_prompt,
        tracker=None,
        on_node_start=None,
        on_node_complete=None,
    ):
        if on_node_start:
            on_node_start("node_a", "anthropic/claude-sonnet-4-6", 1, 2)
        if on_node_complete:
            on_node_complete(SimpleNamespace(node_id="node_a", latency_seconds=0.4))
        if on_node_start:
            on_node_start("node_b", "anthropic/claude-sonnet-4-6", 2, 2)
        if on_node_complete:
            on_node_complete(SimpleNamespace(node_id="node_b", latency_seconds=0.5))
        return SimpleNamespace(
            topology_id="topo-abc12345",
            final_output="All done.",
            success=True,
            error=None,
            total_latency_seconds=1.3,
            models_used=["anthropic/claude-sonnet-4-6"],
            total_input_tokens=20,
            total_output_tokens=10,
            node_results=[],
        )

    with (
        patch("daap.api.ws_handler.TopologySpec.model_validate", return_value=MagicMock()),
        patch("daap.api.ws_handler.resolve_topology", return_value=resolved),
        patch("daap.api.ws_handler.execute_topology", new=AsyncMock(side_effect=_fake_execute)),
    ):
        with client.websocket_connect(f"/ws/{session_id}") as ws:
            ws.send_json({"type": "message", "content": "approve"})

            events = []
            for _ in range(10):
                event = ws.receive_json()
                events.append(event)
                if event.get("type") == "result":
                    break

    assert any(e.get("type") == "executing" for e in events)
    assert any(
        e.get("type") == "progress" and e.get("event") == "node_start"
        for e in events
    )
    assert any(
        e.get("type") == "progress" and e.get("event") == "node_complete" and e.get("percent_complete") == 100
        for e in events
    )
    assert events[-1]["type"] == "result"


# ---------------------------------------------------------------------------
# Session-scoped toolkit tests
# ---------------------------------------------------------------------------

def test_session_scoped_toolkit_has_core_tools():
    """create_session_scoped_toolkit registers topology, question, and status tools."""
    session = Session(session_id="test-01", created_at=0.0)
    toolkit = create_session_scoped_toolkit(session)

    registered = set(toolkit.tools.keys())
    assert "generate_topology" in registered
    assert "ask_user" in registered
    assert "get_execution_status" in registered


def test_session_scoped_toolkit_attaches_resolver():
    """create_session_scoped_toolkit attaches _resolve_answers to the session."""
    session = Session(session_id="test-02", created_at=0.0)
    create_session_scoped_toolkit(session)

    assert callable(session._resolve_answers)


@pytest.mark.asyncio
async def test_session_scoped_ask_user_uses_session_state():
    """Session-scoped ask_user stores questions on the session, not module-level."""
    import asyncio
    from daap.master import tools as master_tools

    session = Session(session_id="test-03", created_at=0.0)
    toolkit = create_session_scoped_toolkit(session)

    # Retrieve the registered ask_user closure
    ask_user_fn = toolkit.tools["ask_user"].original_func

    questions_json = json.dumps([{
        "question": "What's your product?",
        "options": [{"label": "SaaS", "description": "Software"}],
        "multi_select": False,
    }])

    task = asyncio.create_task(ask_user_fn(questions_json))
    await asyncio.sleep(0)

    # Questions stored on session, not module-level state
    assert session.pending_questions is not None
    assert master_tools.get_pending_questions() is None  # module-level untouched

    session._resolve_answers(["SaaS"])
    await task


@pytest.mark.asyncio
async def test_session_scoped_generate_topology_injects_selected_operator_config():
    """Session-scoped generate_topology injects selected subagent operator/model config."""
    session = Session(session_id="test-04", created_at=0.0)
    session.subagent_operator_config = {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_map": {
            "smart": "anthropic/claude-sonnet-4-6",
        },
    }

    toolkit = create_session_scoped_toolkit(session)
    generate_fn = toolkit.tools["generate_topology"].original_func

    topology_json = json.dumps({
        "nodes": [
            {
                "node_id": "node_a",
                "operator_override": {
                    "provider": "openrouter",
                    "model_map": {"smart": "old-model"},
                },
            }
        ]
    })

    with patch("daap.api.sessions.generate_topology_tool", new=AsyncMock(return_value=MagicMock())) as mock_generate:
        await generate_fn(topology_json)

    sent_json = mock_generate.await_args.args[0]
    sent_topology = json.loads(sent_json)

    assert sent_topology["operator_config"]["model_map"]["smart"] == "anthropic/claude-sonnet-4-6"
    assert sent_topology["nodes"][0]["operator_override"]["model_map"]["smart"] == "anthropic/claude-sonnet-4-6"
