"""Integration tests for topology REST endpoints."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from daap.api.routes import app
import daap.api.routes as routes_module
from daap.api.sessions import SessionManager
from daap.api.topology_routes import (
    set_session_manager as set_topology_session_manager,
    set_store as set_topology_store,
)
from daap.feedback.store import FeedbackStore
from daap.topology.store import TopologyStore


SAMPLE_SPEC = {
    "topology_id": "topo-abc12345",
    "version": 1,
    "created_at": "2026-04-14T00:00:00Z",
    "user_prompt": "find b2b leads in fintech",
    "nodes": [{"node_id": "researcher", "role": "Lead Researcher"}],
    "edges": [],
    "constraints": {},
}


@pytest.fixture(autouse=True)
def reset_globals(tmp_path):
    routes_module.session_manager = SessionManager()
    routes_module.feedback_store = FeedbackStore(db_path=str(tmp_path / "feedback.db"))
    routes_module.topology_store = TopologyStore(db_path=str(tmp_path / "topology.db"))
    set_topology_store(routes_module.topology_store)
    set_topology_session_manager(routes_module.session_manager)
    yield


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def seeded_store():
    store = routes_module.topology_store
    store.save_topology(SAMPLE_SPEC, user_id="user-1")
    return store


def test_list_topologies_empty(client):
    resp = client.get("/topologies?user_id=user-1")
    assert resp.status_code == 200
    assert resp.json()["topologies"] == []


def test_list_topologies_returns_saved(client, seeded_store):
    resp = client.get("/topologies?user_id=user-1")
    assert resp.status_code == 200
    items = resp.json()["topologies"]
    assert len(items) == 1
    assert items[0]["topology_id"] == "topo-abc12345"


def test_get_topology_latest(client, seeded_store):
    resp = client.get("/topologies/topo-abc12345")
    assert resp.status_code == 200
    assert resp.json()["topology_id"] == "topo-abc12345"
    assert resp.json()["version"] == 1


def test_get_topology_not_found(client):
    resp = client.get("/topologies/topo-doesnotexist")
    assert resp.status_code == 404


def test_list_versions(client, seeded_store):
    seeded_store.save_topology(SAMPLE_SPEC, user_id="user-1", overwrite=False)
    resp = client.get("/topologies/topo-abc12345/versions")
    assert resp.status_code == 200
    versions = resp.json()["versions"]
    assert len(versions) == 2
    assert versions[0]["version"] == 2


def test_get_specific_version(client, seeded_store):
    resp = client.get("/topologies/topo-abc12345/v/1")
    assert resp.status_code == 200
    assert resp.json()["version"] == 1


def test_get_runs_empty(client, seeded_store):
    resp = client.get("/topologies/topo-abc12345/runs")
    assert resp.status_code == 200
    assert resp.json()["runs"] == []


def test_patch_topology_overwrite(client, seeded_store):
    patched_spec = {**SAMPLE_SPEC, "user_prompt": "updated prompt"}
    resp = client.patch(
        "/topologies/topo-abc12345",
        json={"spec": patched_spec, "save_mode": "overwrite"},
    )
    assert resp.status_code == 200
    assert resp.json()["version"] == 1
    assert resp.json()["spec"]["user_prompt"] == "updated prompt"


def test_patch_topology_new_version(client, seeded_store):
    patched_spec = {**SAMPLE_SPEC, "user_prompt": "updated prompt"}
    resp = client.patch(
        "/topologies/topo-abc12345",
        json={"spec": patched_spec, "save_mode": "new_version"},
    )
    assert resp.status_code == 200
    assert resp.json()["version"] == 2


def test_patch_unknown_topology_returns_404(client):
    resp = client.patch(
        "/topologies/topo-nope",
        json={"spec": SAMPLE_SPEC, "save_mode": "overwrite"},
    )
    assert resp.status_code == 404


def test_rename_topology(client, seeded_store):
    resp = client.patch(
        "/topologies/topo-abc12345/rename",
        json={"name": "my-renamed-topology"},
    )
    assert resp.status_code == 200

    fetched = client.get("/topologies/topo-abc12345").json()
    assert fetched["name"] == "my-renamed-topology"


def test_set_max_runs(client, seeded_store):
    resp = client.patch(
        "/topologies/topo-abc12345/max-runs",
        json={"max_runs": 5},
    )
    assert resp.status_code == 200

    fetched = client.get("/topologies/topo-abc12345").json()
    assert fetched["max_runs"] == 5


def test_set_max_runs_invalid(client, seeded_store):
    resp = client.patch(
        "/topologies/topo-abc12345/max-runs",
        json={"max_runs": 0},
    )
    assert resp.status_code == 400


def test_soft_delete(client, seeded_store):
    resp = client.delete("/topologies/topo-abc12345?ttl_days=30")
    assert resp.status_code == 200

    listed = client.get("/topologies?user_id=user-1").json()
    assert listed["topologies"] == []


def test_restore_topology(client, seeded_store):
    client.delete("/topologies/topo-abc12345")
    resp = client.post("/topologies/topo-abc12345/restore")
    assert resp.status_code == 200

    listed = client.get("/topologies?user_id=user-1").json()
    assert len(listed["topologies"]) == 1


def test_delete_unknown_topology_returns_404(client):
    resp = client.delete("/topologies/topo-nope")
    assert resp.status_code == 404


def test_rerun_topology_success(client, seeded_store):
    session = routes_module.session_manager.create_session()
    session.user_id = "user-1"

    resolved = SimpleNamespace(nodes=[], execution_order=[[]])
    run_result = SimpleNamespace(
        topology_id="topo-abc12345",
        final_output="rerun output",
        success=True,
        error=None,
        total_latency_seconds=1.5,
        total_input_tokens=11,
        total_output_tokens=22,
        models_used=["google/gemini-2.0-flash-001"],
    )

    with (
        patch("daap.api.topology_routes.TopologySpec.model_validate", return_value=MagicMock()),
        patch("daap.api.topology_routes.resolve_topology", return_value=resolved),
        patch("daap.api.topology_routes.execute_topology", new=AsyncMock(return_value=run_result)),
    ):
        resp = client.post(
            "/topologies/topo-abc12345/rerun",
            json={"session_id": session.session_id, "user_prompt": "run again"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["result"]["final_output"] == "rerun output"

    runs = routes_module.topology_store.get_runs("topo-abc12345")
    assert len(runs) == 1


def test_rerun_deleted_topology_returns_410(client, seeded_store):
    session = routes_module.session_manager.create_session()
    routes_module.topology_store.delete_topology("topo-abc12345")

    resp = client.post(
        "/topologies/topo-abc12345/rerun",
        json={"session_id": session.session_id},
    )
    assert resp.status_code == 410
