"""Tests for deterministic session persistence compaction."""

import json

from daap.api.sessions import Session, SessionStore


def _make_message(role: str, i: int) -> dict:
    return {"role": role, "content": f"{role} message {i}"}


def test_session_store_compacts_long_history(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "sessions.db"))
    session = Session(session_id="sess-1", user_id="u1", created_at=0.0)
    session.conversation = [
        _make_message("user" if i % 2 == 0 else "assistant", i)
        for i in range(170)
    ]

    store.save(session)
    row = store.load_one("sess-1")
    assert row is not None
    persisted = json.loads(row["conversation"])

    assert len(persisted) == 101
    assert persisted[0]["metadata"]["daap_compacted"] is True
    assert persisted[0]["metadata"]["strategy"] == "deterministic-summary-v1"


def test_session_store_keeps_short_history_verbatim(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "sessions.db"))
    session = Session(session_id="sess-2", user_id="u2", created_at=0.0)
    session.conversation = [_make_message("user", i) for i in range(10)]

    store.save(session)
    row = store.load_one("sess-2")
    assert row is not None
    persisted = json.loads(row["conversation"])
    assert persisted == session.conversation
