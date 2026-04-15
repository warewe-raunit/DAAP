"""Unit tests for TopologyStore."""
import sqlite3
import time

import pytest

from daap.topology.models import StoredTopology, TopologyRun
from daap.topology.store import TopologyStore


SAMPLE_SPEC = {
    "topology_id": "topo-abc12345",
    "version": 1,
    "user_prompt": "find b2b leads in fintech",
    "nodes": [{"node_id": "researcher", "role": "Lead Researcher"}],
    "edges": [],
}

SAMPLE_RESULT = {
    "topology_id": "topo-abc12345",
    "final_output": "here are your leads",
    "success": True,
    "error": None,
    "latency_seconds": 12.5,
    "total_input_tokens": 100,
    "total_output_tokens": 200,
}


@pytest.fixture
def store(tmp_path):
    return TopologyStore(db_path=str(tmp_path / "test.db"))


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


def test_save_and_get_topology(store):
    saved = store.save_topology(SAMPLE_SPEC, user_id="user-1")
    assert saved.topology_id == "topo-abc12345"
    assert saved.version == 1
    assert saved.user_id == "user-1"
    assert saved.spec["user_prompt"] == "find b2b leads in fintech"

    fetched = store.get_topology("topo-abc12345")
    assert fetched is not None
    assert fetched.topology_id == "topo-abc12345"
    assert fetched.spec == saved.spec


def test_get_nonexistent_returns_none(store):
    assert store.get_topology("topo-doesnotexist") is None


def test_auto_name_generated(store):
    saved = store.save_topology(SAMPLE_SPEC, user_id="user-1")
    assert saved.name == "find-b2b-leads-in-fintech"


def test_custom_name_used(store):
    saved = store.save_topology(
        SAMPLE_SPEC,
        user_id="user-1",
        name="my-custom-name",
    )
    assert saved.name == "my-custom-name"


def test_list_topologies_for_user(store):
    spec2 = {
        **SAMPLE_SPEC,
        "topology_id": "topo-zzz99999",
        "user_prompt": "write emails",
    }
    spec_other_user = {
        **SAMPLE_SPEC,
        "topology_id": "topo-uuu77777",
        "user_prompt": "different user",
    }

    store.save_topology(SAMPLE_SPEC, user_id="user-1")
    store.save_topology(spec2, user_id="user-1")
    store.save_topology(spec_other_user, user_id="user-2")

    results = store.list_topologies(user_id="user-1")
    ids = [t.topology_id for t in results]
    assert "topo-abc12345" in ids
    assert "topo-zzz99999" in ids
    assert len(results) == 2


def test_list_returns_latest_version_per_topology(store):
    store.save_topology(SAMPLE_SPEC, user_id="user-1", overwrite=False)
    store.save_topology(SAMPLE_SPEC, user_id="user-1", overwrite=False)
    results = store.list_topologies(user_id="user-1")
    assert len(results) == 1
    assert results[0].version == 2


def test_overwrite_keeps_version_number(store):
    store.save_topology(SAMPLE_SPEC, user_id="user-1")
    saved = store.save_topology(SAMPLE_SPEC, user_id="user-1", overwrite=True)
    assert saved.version == 1
    assert len(store.list_versions("topo-abc12345")) == 1


def test_overwrite_updates_updated_at(store):
    first = store.save_topology(SAMPLE_SPEC, user_id="user-1")
    time.sleep(0.01)
    second = store.save_topology(SAMPLE_SPEC, user_id="user-1", overwrite=True)
    assert second.updated_at > first.updated_at


def test_new_version_increments(store):
    store.save_topology(SAMPLE_SPEC, user_id="user-1", overwrite=False)
    store.save_topology(SAMPLE_SPEC, user_id="user-1", overwrite=False)
    versions = store.list_versions("topo-abc12345")
    assert [v.version for v in versions] == [2, 1]


def test_get_specific_version(store):
    v1_spec = {**SAMPLE_SPEC, "user_prompt": "v1 prompt"}
    v2_spec = {**SAMPLE_SPEC, "user_prompt": "v2 prompt"}

    store.save_topology(v1_spec, user_id="user-1", overwrite=False)
    store.save_topology(v2_spec, user_id="user-1", overwrite=False)

    v1 = store.get_topology("topo-abc12345", version=1)
    v2 = store.get_topology("topo-abc12345", version=2)
    assert v1 is not None
    assert v2 is not None
    assert v1.spec["user_prompt"] == "v1 prompt"
    assert v2.spec["user_prompt"] == "v2 prompt"


def test_get_latest_returns_highest_version(store):
    store.save_topology(SAMPLE_SPEC, user_id="user-1", overwrite=False)
    store.save_topology(
        {**SAMPLE_SPEC, "user_prompt": "v2"},
        user_id="user-1",
        overwrite=False,
    )
    latest = store.get_topology("topo-abc12345")
    assert latest is not None
    assert latest.version == 2


def test_rename_applies_to_all_versions(store):
    store.save_topology(SAMPLE_SPEC, user_id="user-1", overwrite=False)
    store.save_topology(SAMPLE_SPEC, user_id="user-1", overwrite=False)
    store.rename_topology("topo-abc12345", "new-name")

    for version in store.list_versions("topo-abc12345"):
        assert version.name == "new-name"


def test_save_and_get_run(store):
    store.save_topology(SAMPLE_SPEC, user_id="user-1")
    run = store.save_run(
        topology_id="topo-abc12345",
        topology_version=1,
        user_id="user-1",
        result=SAMPLE_RESULT,
        user_prompt="find leads",
    )
    assert run.run_id is not None
    assert run.success is True
    assert run.user_prompt == "find leads"

    runs = store.get_runs("topo-abc12345")
    assert len(runs) == 1
    assert runs[0].run_id == run.run_id


def test_runs_ordered_newest_first(store):
    store.save_topology(SAMPLE_SPEC, user_id="user-1")
    run_one = store.save_run("topo-abc12345", 1, "user-1", SAMPLE_RESULT)
    time.sleep(0.01)
    run_two = store.save_run("topo-abc12345", 1, "user-1", SAMPLE_RESULT)

    runs = store.get_runs("topo-abc12345")
    assert runs[0].run_id == run_two.run_id
    assert runs[1].run_id == run_one.run_id


def test_run_cap_enforced(store):
    store.save_topology(SAMPLE_SPEC, user_id="user-1")
    store.set_max_runs("topo-abc12345", 3)

    for _ in range(5):
        store.save_run("topo-abc12345", 1, "user-1", SAMPLE_RESULT)

    runs = store.get_runs("topo-abc12345")
    assert len(runs) == 3


def test_run_cap_deletes_oldest(store):
    store.save_topology(SAMPLE_SPEC, user_id="user-1")
    store.set_max_runs("topo-abc12345", 2)

    run_one = store.save_run("topo-abc12345", 1, "user-1", SAMPLE_RESULT)
    time.sleep(0.01)
    run_two = store.save_run("topo-abc12345", 1, "user-1", SAMPLE_RESULT)
    time.sleep(0.01)
    run_three = store.save_run("topo-abc12345", 1, "user-1", SAMPLE_RESULT)

    runs = store.get_runs("topo-abc12345")
    run_ids = [r.run_id for r in runs]
    assert run_one.run_id not in run_ids
    assert run_two.run_id in run_ids
    assert run_three.run_id in run_ids


def test_get_runs_limit(store):
    store.save_topology(SAMPLE_SPEC, user_id="user-1")
    for _ in range(5):
        store.save_run("topo-abc12345", 1, "user-1", SAMPLE_RESULT)

    runs = store.get_runs("topo-abc12345", limit=2)
    assert len(runs) == 2


def test_soft_delete_hides_from_list(store):
    store.save_topology(SAMPLE_SPEC, user_id="user-1")
    store.delete_topology("topo-abc12345")
    results = store.list_topologies(user_id="user-1")
    assert len(results) == 0


def test_soft_delete_visible_with_include_deleted(store):
    store.save_topology(SAMPLE_SPEC, user_id="user-1")
    store.delete_topology("topo-abc12345")
    results = store.list_topologies(user_id="user-1", include_deleted=True)
    assert len(results) == 1
    assert results[0].deleted_at is not None


def test_restore_makes_visible_again(store):
    store.save_topology(SAMPLE_SPEC, user_id="user-1")
    store.delete_topology("topo-abc12345")
    store.restore_topology("topo-abc12345")
    results = store.list_topologies(user_id="user-1")
    assert len(results) == 1
    assert results[0].deleted_at is None


def test_purge_removes_past_ttl(store):
    store.save_topology(SAMPLE_SPEC, user_id="user-1")

    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE topologies SET deleted_at = ?, delete_ttl_days = 0 WHERE topology_id = ?",
            (time.time() - 1, "topo-abc12345"),
        )
        conn.commit()

    count = store.purge_expired()
    assert count == 1
    assert store.get_topology("topo-abc12345") is None


def test_purge_leaves_active_topologies(store):
    store.save_topology(SAMPLE_SPEC, user_id="user-1")
    count = store.purge_expired()
    assert count == 0
    assert store.get_topology("topo-abc12345") is not None


def test_purge_leaves_recently_deleted(store):
    store.save_topology(SAMPLE_SPEC, user_id="user-1")
    store.delete_topology("topo-abc12345", ttl_days=30)
    count = store.purge_expired()
    assert count == 0
