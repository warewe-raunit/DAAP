"""
DAAP Memory Tests — config, scopes, extractors, reader, writer, palace.

All tests mock mem0.Memory — no real API calls, no Qdrant, no OpenAI key needed.
"""

import asyncio
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_mem0_client(results=None):
    """Build a mock Mem0 Memory client."""
    client = MagicMock()
    results = results or []
    client.get_all.return_value = {"results": results}
    client.search.return_value = {"results": results}
    client.add.return_value = {"results": []}
    client.delete_all.return_value = None
    return client


# ===========================================================================
# Config tests
# ===========================================================================

class TestConfig:
    def test_build_config_production_has_qdrant(self):
        from daap.memory.config import build_config
        cfg = build_config("production")
        assert cfg["vector_store"]["provider"] == "qdrant"
        assert cfg["vector_store"]["config"]["collection_name"] == "daap_memories"

    def test_build_config_testing_no_vector_store(self):
        from daap.memory.config import build_config
        cfg = build_config("testing")
        assert "vector_store" not in cfg

    def test_build_config_llm_uses_openrouter(self):
        from daap.memory.config import build_config
        cfg = build_config("production")
        assert cfg["llm"]["provider"] == "openai"
        assert "openrouter.ai" in cfg["llm"]["config"]["openai_base_url"]
        assert "gemini" in cfg["llm"]["config"]["model"]

    def test_build_config_embedder_is_text_embedding(self):
        from daap.memory.config import build_config
        cfg = build_config()
        assert cfg["embedder"]["config"]["model"] == "text-embedding-3-small"

    def test_check_memory_available_missing_openai_key(self, monkeypatch):
        import daap.memory.config as cfg_mod
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        cfg_mod.reset_memory_client()
        from daap.memory.config import check_memory_available
        ok, reason = check_memory_available()
        assert not ok
        assert "OPENAI_API_KEY" in reason

    def test_check_memory_available_missing_openrouter_key(self, monkeypatch):
        import daap.memory.config as cfg_mod
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        cfg_mod.reset_memory_client()
        from daap.memory.config import check_memory_available
        ok, reason = check_memory_available()
        assert not ok
        assert "OPENROUTER_API_KEY" in reason

    def test_reset_memory_client_clears_singleton(self):
        import daap.memory.config as cfg_mod
        cfg_mod._memory_client = MagicMock()
        cfg_mod.reset_memory_client()
        assert cfg_mod._memory_client is None


# ===========================================================================
# Scopes tests
# ===========================================================================

class TestScopes:
    def test_profile_scope(self):
        from daap.memory.scopes import profile_scope
        s = profile_scope("alice")
        assert s == {"user_id": "alice", "agent_id": "profile"}

    def test_master_scope(self):
        from daap.memory.scopes import master_scope
        s = master_scope("alice")
        assert s == {"user_id": "alice", "agent_id": "master"}

    def test_agent_diary_scope_normalizes_role(self):
        from daap.memory.scopes import agent_diary_scope
        s = agent_diary_scope("alice", "Lead Researcher")
        assert s == {"user_id": "alice", "agent_id": "researcher_diary"}

    def test_agent_diary_scope_writer(self):
        from daap.memory.scopes import agent_diary_scope
        s = agent_diary_scope("alice", "Email Drafter")
        assert s["agent_id"] == "writer_diary"

    def test_all_user_scope(self):
        from daap.memory.scopes import all_user_scope
        s = all_user_scope("bob")
        assert s == {"user_id": "bob"}

    def test_normalize_role_evaluator(self):
        from daap.memory.scopes import _normalize_role
        assert _normalize_role("Lead Scorer") == "evaluator"

    def test_normalize_role_formatter(self):
        from daap.memory.scopes import _normalize_role
        assert _normalize_role("Output Formatter") == "formatter"

    def test_normalize_role_unknown_falls_back(self):
        from daap.memory.scopes import _normalize_role
        assert _normalize_role("Outbound Specialist") == "outbound_specialist"


# ===========================================================================
# Extractor tests
# ===========================================================================

class TestExtractors:
    def test_extract_profile_from_conversation_no_clarifications(self):
        from daap.memory.extractors import extract_profile_from_conversation
        msgs = extract_profile_from_conversation("I sell SaaS to construction firms")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert "SaaS" in msgs[0]["content"]

    def test_extract_profile_from_conversation_with_clarifications(self):
        from daap.memory.extractors import extract_profile_from_conversation
        clars = [("What is your ICP?", "SMB construction"), ("What's your ACV?", "$12k")]
        msgs = extract_profile_from_conversation("help me", clars)
        assert len(msgs) == 5  # user + 2x(assistant+user)
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["content"] == "SMB construction"

    def test_extract_run_summary_success(self):
        from daap.memory.extractors import extract_run_summary
        topo = {"nodes": [{"role": "researcher"}, {"role": "writer"}]}
        result = {"success": True, "total_cost_usd": 0.0023, "total_latency_seconds": 5.1}
        summary = extract_run_summary(topo, result, user_rating=4)
        assert "2-node" in summary
        assert "succeeded" in summary
        assert "4/5" in summary

    def test_extract_run_summary_failure(self):
        from daap.memory.extractors import extract_run_summary
        topo = {"nodes": [{"role": "researcher"}]}
        result = {"success": False, "error": "timeout", "cost_usd": 0, "latency_seconds": 0}
        summary = extract_run_summary(topo, result)
        assert "failed" in summary
        assert "timeout" in summary

    def test_extract_agent_observation_success(self):
        from daap.memory.extractors import extract_agent_observation
        obs = extract_agent_observation("researcher", "Found 10 leads", 3.2, "gemini-flash", True)
        assert "researcher" in obs
        assert "completed" in obs
        assert "Found 10 leads" in obs

    def test_extract_correction_low_rating(self):
        from daap.memory.extractors import extract_correction_from_rating
        correction = extract_correction_from_rating(1, comment="too vague", topology_summary="3 nodes")
        assert correction is not None
        assert "1/5" in correction
        assert "too vague" in correction
        assert "Avoid" in correction

    def test_extract_correction_high_rating_returns_none(self):
        from daap.memory.extractors import extract_correction_from_rating
        correction = extract_correction_from_rating(4)
        assert correction is None

    def test_extract_correction_rating_2_triggers(self):
        from daap.memory.extractors import extract_correction_from_rating
        correction = extract_correction_from_rating(2)
        assert correction is not None


# ===========================================================================
# Reader tests (with mocked client)
# ===========================================================================

class TestReader:
    @pytest.fixture(autouse=True)
    def mock_client(self, monkeypatch):
        self.client = _mock_mem0_client(
            results=[{"memory": "User sells SaaS"}, {"memory": "ICP: construction firms"}]
        )
        monkeypatch.setattr("daap.memory.reader.get_memory_client", lambda: self.client)

    def test_load_user_profile_returns_memory_strings(self):
        from daap.memory.reader import load_user_profile
        result = load_user_profile("alice")
        assert result == ["User sells SaaS", "ICP: construction firms"]
        self.client.get_all.assert_called_once()

    def test_load_user_profile_returns_empty_on_exception(self):
        from daap.memory.reader import load_user_profile
        self.client.get_all.side_effect = RuntimeError("boom")
        result = load_user_profile("alice")
        assert result == []

    def test_load_master_history_uses_search(self):
        from daap.memory.reader import load_master_history
        load_master_history("alice", "lead gen emails")
        self.client.search.assert_called_once()
        call_kwargs = self.client.search.call_args[1]
        assert call_kwargs["query"] == "lead gen emails"

    def test_load_master_history_empty_query_uses_fallback(self):
        from daap.memory.reader import load_master_history
        load_master_history("alice", "")
        call_kwargs = self.client.search.call_args[1]
        assert call_kwargs["query"] == "past run topology result"

    def test_load_agent_diary_with_query_uses_search(self):
        from daap.memory.reader import load_agent_diary
        load_agent_diary("alice", "researcher", query="find leads")
        self.client.search.assert_called_once()

    def test_load_agent_diary_no_query_uses_get_all(self):
        from daap.memory.reader import load_agent_diary
        load_agent_diary("alice", "researcher")
        self.client.get_all.assert_called_once()

    def test_format_profile_for_prompt_empty(self):
        from daap.memory.reader import format_profile_for_prompt
        result = format_profile_for_prompt([])
        assert result == ""

    def test_format_profile_for_prompt_includes_header(self):
        from daap.memory.reader import format_profile_for_prompt
        result = format_profile_for_prompt(["fact one", "fact two"])
        assert "## What I Know About This User" in result
        assert "- fact one" in result

    def test_format_diary_for_prompt_includes_role(self):
        from daap.memory.reader import format_diary_for_prompt
        result = format_diary_for_prompt(["lesson1"], "researcher")
        assert "Researcher" in result
        assert "lesson1" in result


# ===========================================================================
# Writer tests (async)
# ===========================================================================

class TestWriter:
    @pytest.fixture(autouse=True)
    def mock_client(self, monkeypatch):
        self.client = _mock_mem0_client()
        monkeypatch.setattr("daap.memory.writer.get_memory_client", lambda: self.client)

    def test_write_profile_async_calls_client_add(self):
        from daap.memory.writer import write_profile_async
        asyncio.run(
            write_profile_async("alice", "I sell SaaS to construction")
        )
        self.client.add.assert_called_once()

    def test_write_run_summary_async_calls_client_add(self):
        from daap.memory.writer import write_run_summary_async
        topo = {"nodes": [{"role": "researcher"}]}
        result = {"success": True, "total_cost_usd": 0.001, "total_latency_seconds": 2.0}
        asyncio.run(
            write_run_summary_async("alice", topo, result)
        )
        self.client.add.assert_called_once()

    def test_write_correction_skips_high_rating(self):
        from daap.memory.writer import write_correction_async
        asyncio.run(
            write_correction_async("alice", 5)
        )
        self.client.add.assert_not_called()

    def test_write_correction_fires_for_low_rating(self):
        from daap.memory.writer import write_correction_async
        asyncio.run(
            write_correction_async("alice", 1, comment="bad output")
        )
        self.client.add.assert_called_once()


# ===========================================================================
# Palace (DaapMemory facade) tests
# ===========================================================================

class TestPalace:
    def test_daap_memory_unavailable_when_no_keys(self, monkeypatch):
        monkeypatch.setattr("daap.memory.reader.memory_is_available", lambda: False)
        from daap.memory.palace import DaapMemory
        mem = DaapMemory()
        assert not mem.available

    def test_get_user_profile_returns_empty_when_unavailable(self, monkeypatch):
        monkeypatch.setattr("daap.memory.reader.memory_is_available", lambda: False)
        from daap.memory.palace import DaapMemory
        mem = DaapMemory()
        assert mem.get_user_profile("alice") == []

    def test_get_past_runs_returns_empty_when_unavailable(self, monkeypatch):
        monkeypatch.setattr("daap.memory.reader.memory_is_available", lambda: False)
        from daap.memory.palace import DaapMemory
        mem = DaapMemory()
        assert mem.get_past_runs("alice", "leads") == []

    def test_format_for_master_prompt_returns_empty_when_unavailable(self, monkeypatch):
        monkeypatch.setattr("daap.memory.reader.memory_is_available", lambda: False)
        from daap.memory.palace import DaapMemory
        mem = DaapMemory()
        result = mem.format_for_master_prompt("alice", "some prompt")
        assert result == ""

    def test_format_for_node_prompt_returns_empty_when_unavailable(self, monkeypatch):
        monkeypatch.setattr("daap.memory.reader.memory_is_available", lambda: False)
        from daap.memory.palace import DaapMemory
        mem = DaapMemory()
        result = mem.format_for_node_prompt("alice", "researcher", "find leads")
        assert result == ""

    def test_remember_run_fires_when_available(self, monkeypatch):
        fired = []
        monkeypatch.setattr("daap.memory.reader.memory_is_available", lambda: True)
        monkeypatch.setattr("daap.memory.writer.fire_and_forget", lambda coro: fired.append(coro))
        from daap.memory.palace import DaapMemory
        mem = DaapMemory()
        mem.remember_run("alice", {"nodes": []}, {"success": True})
        assert len(fired) == 1

    def test_remember_run_skips_when_unavailable(self, monkeypatch):
        fired = []
        monkeypatch.setattr("daap.memory.reader.memory_is_available", lambda: False)
        monkeypatch.setattr("daap.memory.writer.fire_and_forget", lambda coro: fired.append(coro))
        from daap.memory.palace import DaapMemory
        mem = DaapMemory()
        mem.remember_run("alice", {"nodes": []}, {"success": True})
        assert len(fired) == 0

    def test_remember_correction_fires_regardless_of_rating(self, monkeypatch):
        fired = []
        monkeypatch.setattr("daap.memory.reader.memory_is_available", lambda: True)
        monkeypatch.setattr("daap.memory.writer.fire_and_forget", lambda coro: fired.append(coro))
        from daap.memory.palace import DaapMemory
        mem = DaapMemory()
        mem.remember_correction("alice", 5)
        # fire_and_forget is called; write_correction_async skips internally for high rating
        assert len(fired) == 1

    def test_get_memory_singleton(self, monkeypatch):
        monkeypatch.setattr("daap.memory.reader.memory_is_available", lambda: False)
        import daap.memory.palace as palace_mod
        palace_mod._default_memory = None
        from daap.memory.palace import get_memory
        m1 = get_memory()
        m2 = get_memory()
        assert m1 is m2
        palace_mod._default_memory = None  # cleanup
