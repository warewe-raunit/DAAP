"""
DAAP Feedback Store Tests — pytest suite for feedback/store.py and feedback/collector.py
"""

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from daap.feedback.store import FeedbackStore
from daap.feedback.collector import collect_run_feedback
from daap.executor.engine import ExecutionResult, NodeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_store(tmp_path: Path) -> FeedbackStore:
    return FeedbackStore(db_path=str(tmp_path / "test.db"))


def make_execution_result(success: bool = True, error: str | None = None) -> ExecutionResult:
    return ExecutionResult(
        topology_id="topo-test0001",
        final_output="10 leads found and emails drafted.",
        node_results=[
            NodeResult(node_id="researcher", output_text="Raw leads...", latency_seconds=1.2),
            NodeResult(node_id="drafter", output_text="Email drafts...", latency_seconds=0.8),
        ],
        total_latency_seconds=2.0,
        success=success,
        error=error,
    )


SAMPLE_TOPOLOGY = {
    "topology_id": "topo-test0001",
    "nodes": [{"node_id": "researcher"}],
}

SAMPLE_RESULT = {
    "topology_id": "topo-test0001",
    "final_output": "10 leads found.",
    "success": True,
    "error": None,
    "latency_seconds": 2.0,
}


# ---------------------------------------------------------------------------
# FeedbackStore tests
# ---------------------------------------------------------------------------

def test_db_created_on_init(tmp_path):
    """FeedbackStore creates the SQLite .db file on initialisation."""
    db_path = str(tmp_path / "feedback.db")
    assert not Path(db_path).exists()
    FeedbackStore(db_path=db_path)
    assert Path(db_path).exists()


def test_store_run(tmp_path):
    """store_run inserts a record retrievable via get_runs_for_session."""
    store = make_store(tmp_path)
    store.store_run(
        session_id="sess-001",
        topology_json=SAMPLE_TOPOLOGY,
        execution_result=SAMPLE_RESULT,
    )
    runs = store.get_runs_for_session("sess-001")
    assert len(runs) == 1
    run = runs[0]
    assert run["session_id"] == "sess-001"
    assert run["success"] == 1
    assert run["total_latency_seconds"] == 2.0
    assert json.loads(run["topology_json"])["topology_id"] == "topo-test0001"


def test_store_rating_updates_existing_run(tmp_path):
    """store_rating attaches a rating to the most recent run for a session."""
    store = make_store(tmp_path)
    store.store_run("sess-002", SAMPLE_TOPOLOGY, SAMPLE_RESULT)

    store.store_rating(
        session_id="sess-002",
        rating=5,
        comment="Great results!",
    )

    runs = store.get_runs_for_session("sess-002")
    assert len(runs) == 1
    assert runs[0]["rating"] == 5
    assert runs[0]["comment"] == "Great results!"


def test_store_rating_inserts_if_no_run(tmp_path):
    """store_rating inserts a new row when no run exists for the session."""
    store = make_store(tmp_path)
    store.store_rating(
        session_id="sess-003",
        rating=4,
        comment="Good",
        topology_json=SAMPLE_TOPOLOGY,
        execution_result=SAMPLE_RESULT,
    )
    runs = store.get_runs_for_session("sess-003")
    assert len(runs) == 1
    assert runs[0]["rating"] == 4


def test_get_runs_for_session_filters_correctly(tmp_path):
    """get_runs_for_session returns only runs for the requested session."""
    store = make_store(tmp_path)
    store.store_run("sess-A", SAMPLE_TOPOLOGY, SAMPLE_RESULT)
    store.store_run("sess-A", SAMPLE_TOPOLOGY, SAMPLE_RESULT)
    store.store_run("sess-B", SAMPLE_TOPOLOGY, SAMPLE_RESULT)

    runs_a = store.get_runs_for_session("sess-A")
    runs_b = store.get_runs_for_session("sess-B")

    assert len(runs_a) == 2
    assert len(runs_b) == 1
    assert all(r["session_id"] == "sess-A" for r in runs_a)


def test_get_all_rated_runs(tmp_path):
    """get_all_rated_runs returns only runs with a rating."""
    store = make_store(tmp_path)
    store.store_run("sess-X", SAMPLE_TOPOLOGY, SAMPLE_RESULT)
    store.store_run("sess-Y", SAMPLE_TOPOLOGY, SAMPLE_RESULT)
    store.store_run("sess-Z", SAMPLE_TOPOLOGY, SAMPLE_RESULT)

    store.store_rating("sess-X", rating=5)
    store.store_rating("sess-Y", rating=3)

    rated = store.get_all_rated_runs()
    session_ids = {r["session_id"] for r in rated}
    assert "sess-X" in session_ids
    assert "sess-Y" in session_ids
    assert "sess-Z" not in session_ids


def test_db_persists_across_instances(tmp_path):
    """Data written by one FeedbackStore instance is readable by another."""
    db_path = str(tmp_path / "shared.db")
    store1 = FeedbackStore(db_path=db_path)
    store1.store_run("sess-persist", SAMPLE_TOPOLOGY, SAMPLE_RESULT)

    store2 = FeedbackStore(db_path=db_path)
    runs = store2.get_runs_for_session("sess-persist")
    assert len(runs) == 1
    assert runs[0]["session_id"] == "sess-persist"


def test_purge_expired_removes_old_rows(tmp_path):
    store = make_store(tmp_path)
    store.store_run("sess-old", SAMPLE_TOPOLOGY, SAMPLE_RESULT)
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE runs SET timestamp = ? WHERE session_id = ?",
            (time.time() - (3 * 86400), "sess-old"),
        )
        conn.commit()

    purged = store.purge_expired(retention_days=1)
    assert purged == 1
    assert store.get_runs_for_session("sess-old") == []


# ---------------------------------------------------------------------------
# Collector tests
# ---------------------------------------------------------------------------

def test_collect_run_feedback(tmp_path):
    """collect_run_feedback writes ExecutionResult to the store."""
    store = make_store(tmp_path)
    result = make_execution_result(success=True)

    collect_run_feedback(
        feedback_store=store,
        session_id="sess-col",
        topology_dict=SAMPLE_TOPOLOGY,
        execution_result=result,
    )

    runs = store.get_runs_for_session("sess-col")
    assert len(runs) == 1
    stored = json.loads(runs[0]["execution_result"])
    assert stored["topology_id"] == "topo-test0001"
    assert stored["success"] is True
    assert len(stored["node_results"]) == 2


def test_collect_run_feedback_truncates_long_output(tmp_path):
    """collect_run_feedback truncates final_output to 1000 chars for storage."""
    store = make_store(tmp_path)
    long_result = ExecutionResult(
        topology_id="topo-long",
        final_output="x" * 5000,
        node_results=[],
        total_latency_seconds=1.0,
        success=True,
    )

    collect_run_feedback(store, "sess-long", SAMPLE_TOPOLOGY, long_result)

    runs = store.get_runs_for_session("sess-long")
    stored = json.loads(runs[0]["execution_result"])
    assert len(stored["final_output"]) <= 1000
