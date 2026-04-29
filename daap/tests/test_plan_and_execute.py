"""
Plan-and-Execute: 17 assertions.

Run: python -m pytest daap/tests/test_plan_and_execute.py -v
"""

import json
import types
import pytest
import unittest.mock as mock

from daap.master.planner import (
    PlanResult,
    INTENT_TOPOLOGY,
    INTENT_ANSWER,
    INTENT_CLARIFY,
    MAX_PLANNER_ITERS,
    plan_turn,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _make_fake_model(response_text: str):
    """Fake model for plan_turn — provides config attrs; constrained clone responds."""
    m = types.SimpleNamespace(
        content=response_text,
        model_name="qwen/qwen3.6-plus",
        api_key="test-key",
        client_kwargs={"base_url": "https://openrouter.ai/api/v1"},
    )
    return m


class _FakeFormatter:
    def format_messages(self, msgs):
        return msgs


_fmt = _FakeFormatter()


def _planner_mock(response_text: str):
    """Context manager: patches TrackedOpenAIChatModel used inside plan_turn."""
    async def _fake_call(msgs):
        return types.SimpleNamespace(content=response_text)

    cm = mock.patch("daap.executor.tracked_model.TrackedOpenAIChatModel")
    return cm, _fake_call


# All planner tests that need network-free results use this helper
async def _plan_with(text: str, user_msg: str = "test") -> PlanResult:
    """Run plan_turn with TrackedOpenAIChatModel patched to return text.

    Uses AsyncMock so `await inst(msgs)` returns the desired SimpleNamespace.
    instance.__call__ assignment on a MagicMock does NOT intercept `await inst(...)`
    because Python's call protocol uses type(x).__call__, not the instance attr.
    """
    with mock.patch("daap.executor.tracked_model.TrackedOpenAIChatModel") as MockCls:
        MockCls.return_value = mock.AsyncMock(
            return_value=types.SimpleNamespace(content=text)
        )
        return await plan_turn(user_msg, _make_fake_model(text), _fmt)


# ---------------------------------------------------------------------------
# 1-3: PlanResult intent storage
# ---------------------------------------------------------------------------

def test_01_planresult_stores_topology():
    assert PlanResult(intent=INTENT_TOPOLOGY).intent == INTENT_TOPOLOGY

def test_02_planresult_stores_answer():
    assert PlanResult(intent=INTENT_ANSWER).intent == INTENT_ANSWER

def test_03_planresult_stores_clarify():
    assert PlanResult(intent=INTENT_CLARIFY).intent == INTENT_CLARIFY


# ---------------------------------------------------------------------------
# 4-5: PlanResult.fallback()
# ---------------------------------------------------------------------------

def test_04_fallback_intent_is_topology():
    assert PlanResult.fallback().intent == INTENT_TOPOLOGY

def test_05_fallback_has_nonempty_steps():
    assert len(PlanResult.fallback().steps) > 0


# ---------------------------------------------------------------------------
# 6-7: PlanResult.hint()
# ---------------------------------------------------------------------------

def test_06_hint_contains_intent():
    r = PlanResult(intent=INTENT_TOPOLOGY, steps=["delegate_to_architect"], max_iters=6)
    assert "TOPOLOGY" in r.hint()

def test_07_hint_contains_budget():
    r = PlanResult(intent=INTENT_TOPOLOGY, steps=["delegate_to_architect"], max_iters=6)
    assert "Iteration budget: 6" in r.hint()


# ---------------------------------------------------------------------------
# 8-16: plan_turn async behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_08_fallback_on_invalid_json():
    result = await _plan_with("not json!!!")
    assert result.intent == INTENT_TOPOLOGY

@pytest.mark.asyncio
async def test_09_fallback_on_model_error():
    with mock.patch("daap.executor.tracked_model.TrackedOpenAIChatModel") as MockCls:
        MockCls.return_value = mock.AsyncMock(side_effect=RuntimeError("timeout"))
        result = await plan_turn("test", _make_fake_model(""), _fmt)
    assert result.intent == INTENT_TOPOLOGY

@pytest.mark.asyncio
async def test_10_parses_answer_intent():
    payload = json.dumps({"intent": "ANSWER", "steps": [], "max_iters": 1})
    result = await _plan_with(payload, "what tools?")
    assert result.intent == INTENT_ANSWER

@pytest.mark.asyncio
async def test_11_parses_clarify_intent():
    payload = json.dumps({"intent": "CLARIFY", "steps": [], "max_iters": 1})
    result = await _plan_with(payload, "research that")
    assert result.intent == INTENT_CLARIFY

@pytest.mark.asyncio
async def test_12_parses_topology_intent_with_steps():
    payload = json.dumps({
        "intent": "TOPOLOGY",
        "steps": ["delegate_to_architect", "ask_user", "execute_pending_topology"],
        "max_iters": 6,
    })
    result = await _plan_with(payload, "find 20 reddit posts")
    assert result.intent == INTENT_TOPOLOGY
    assert "delegate_to_architect" in result.steps

@pytest.mark.asyncio
async def test_13_caps_max_iters_at_ceiling():
    payload = json.dumps({"intent": "TOPOLOGY", "steps": [], "max_iters": 99})
    result = await _plan_with(payload)
    assert result.max_iters <= MAX_PLANNER_ITERS

@pytest.mark.asyncio
async def test_14_min_max_iters_is_one():
    payload = json.dumps({"intent": "ANSWER", "steps": [], "max_iters": 0})
    result = await _plan_with(payload)
    assert result.max_iters >= 1

@pytest.mark.asyncio
async def test_15_strips_markdown_fences():
    inner = json.dumps({"intent": "ANSWER", "steps": [], "max_iters": 1})
    fenced = f"```json\n{inner}\n```"
    result = await _plan_with(fenced)
    assert result.intent == INTENT_ANSWER

@pytest.mark.asyncio
async def test_16_normalizes_unknown_intent_to_topology():
    payload = json.dumps({"intent": "EXECUTE", "steps": [], "max_iters": 3})
    result = await _plan_with(payload)
    assert result.intent == INTENT_TOPOLOGY


# ---------------------------------------------------------------------------
# 17: MAX_PLANNER_ITERS matches TOPOLOGY path bound
# ---------------------------------------------------------------------------

def test_17_max_planner_iters_matches_topology_path():
    # TOPOLOGY path: delegate_to_architect(1) + ask_user(1) + execute(1) + buffer(3) = 6
    # MAX_PLANNER_ITERS is the hard ceiling; fallback uses 6 for the TOPOLOGY path
    assert PlanResult.fallback().max_iters == 6
    assert MAX_PLANNER_ITERS >= PlanResult.fallback().max_iters


# ---------------------------------------------------------------------------
# R4: constrained decoding — _repair_topology_json
# ---------------------------------------------------------------------------

from daap.master.tools import _repair_topology_json
import unittest.mock as mock


@pytest.mark.asyncio
async def test_r4_repair_returns_none_on_complete_failure():
    """All repair attempts fail → None returned (not raised)."""
    async def _failing(msgs):
        raise RuntimeError("network error")

    with mock.patch("daap.executor.tracked_model.TrackedOpenAIChatModel") as MockCls:
        inst = mock.MagicMock()
        inst.__call__ = _failing
        MockCls.return_value = inst
        result = await _repair_topology_json('{"bad json:}')
    assert result is None


@pytest.mark.asyncio
async def test_r4_repair_parses_returned_json():
    """Repair returns valid JSON → parsed dict returned."""
    valid = {"topology_id": "topo-abcd1234", "version": 1}

    with mock.patch("daap.executor.tracked_model.TrackedOpenAIChatModel") as MockCls:
        MockCls.return_value = mock.AsyncMock(
            return_value=types.SimpleNamespace(content=json.dumps(valid))
        )
        result = await _repair_topology_json('{"bad json:}')
    assert result == valid


# ---------------------------------------------------------------------------
# R5: temperature + seed — master generate_kwargs
# ---------------------------------------------------------------------------

def test_r5_master_temperature_nonzero():
    """Master model must have temperature > 0 to explore on retry."""
    from daap.master.agent import _build_model_and_formatter
    model, _, _ = _build_model_and_formatter(operator_config=None)
    gk = getattr(model, "generate_kwargs", {}) or {}
    assert gk.get("temperature", 0) > 0, "master temperature must be > 0"


def test_r5_master_no_seed():
    """Seed must be absent — deterministic seed repeats same broken JSON."""
    from daap.master.agent import _build_model_and_formatter
    model, _, _ = _build_model_and_formatter(operator_config=None)
    gk = getattr(model, "generate_kwargs", {}) or {}
    assert "seed" not in gk, "master generate_kwargs must not contain seed"
