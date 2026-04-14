"""Unit tests for TopologyStore."""
import time
import pytest
from daap.topology.models import StoredTopology, TopologyRun


def test_stored_topology_fields():
    t = StoredTopology(
        topology_id="topo-abc12345",
        version=1,
        user_id="user-1",
        name="find-b2b-leads",
        spec={"topology_id": "topo-abc12345", "nodes": []},
        created_at=1000.0,
        updated_at=2000.0,
        deleted_at=None,
        max_runs=10,
    )
    assert t.topology_id == "topo-abc12345"
    assert t.version == 1
    assert t.deleted_at is None
    assert t.max_runs == 10


def test_topology_run_fields():
    r = TopologyRun(
        run_id="run-001",
        topology_id="topo-abc12345",
        topology_version=1,
        user_id="user-1",
        ran_at=3000.0,
        user_prompt="find leads",
        result={"success": True, "final_output": "output"},
        success=True,
        latency_seconds=12.5,
        input_tokens=100,
        output_tokens=200,
    )
    assert r.run_id == "run-001"
    assert r.success is True
    assert r.result["final_output"] == "output"
