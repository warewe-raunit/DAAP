# Design: Cost Explosion Fix + Estimation Bug Fix

**Date:** 2026-04-17
**Status:** Approved
**Scope:** `daap/executor/`, `daap/spec/`, `daap/master/`, `daap/api/`

---

## Problem Summary

### P1 — Uncontrolled Cost Explosion (CRITICAL)

No hard spending cap exists during execution. A single topology can make unlimited
LLM calls (retries × parallel instances × ReAct iterations × consolidation calls)
with no abort signal tied to `constraints.max_cost_usd`.

Root causes:
1. `engine.py` never converts token counts to USD or compares against `max_cost_usd`
2. `patterns.py:_llm_consolidate` creates a raw `OpenAIChatModel` — tokens not tracked
3. Retry loop catches all exceptions identically; 429 (quota exceeded) retried same as timeout
4. `max_react_iterations` default = 10 (NodeSpec) with no token-level budget

### P2 — Estimation Shows Same Value for Every Topology

Root causes:
1. `_last_topology_result` in `tools.py` is a module-level global — concurrent sessions
   overwrite each other; both see the last writer's estimate
2. Claude model IDs (`claude-haiku-4-5-20251001`, `claude-sonnet-4-6`, `claude-opus-4-6`)
   absent from `MODEL_PRICING` — all fall to `DEFAULT_PRICING` ($1/$5 per 1M),
   collapsing all operator-config topologies to the same estimate
3. `:.2f` display format rounds sub-cent costs to `$0.00`, masking real variation

---

## Architecture

### 1. Cost-Aware TokenTracker (`tools/token_tracker.py`)

Add `total_cost_usd(pricing_fn)` method. Receives a callable
`pricing_fn(model_id) -> dict[str, float]` and sums cost across all recorded calls.
No import of resolver inside the tracker — caller provides the pricing function
(dependency inversion).

```
TokenTracker.total_cost_usd(pricing_fn) -> float
    = sum over _calls:
        call.input_tokens * pricing_fn(call.model_id)["input_per_1m"] / 1_000_000
      + call.output_tokens * pricing_fn(call.model_id)["output_per_1m"] / 1_000_000
```

### 2. Global Hard Cap in `engine.py`

After each step completes, compute `current_spend = exec_tracker.total_cost_usd(get_model_pricing)`.
If `current_spend >= constraints.max_cost_usd`, return a failed `ExecutionResult`
with `error="Cost limit exceeded: spent $X.XXXX of $Y.YY budget"`.

Check happens BEFORE starting the next step (post-step, pre-next-step boundary).
This means a step that pushes over budget runs to completion but the pipeline aborts
before the next step starts — clean boundary, no mid-step cancellation complexity.

### 3. Circuit Breaker in `engine.py`

Track `consecutive_errors: int = 0` in the step-retry loop.

Error classification (industry standard for OpenAI-compatible clients):
- Import `openai.RateLimitError` — catches 429 directly (typed)
- Import `openai.APIStatusError` — catches 500/503 via `.status_code`
- Fallback: `"429" in str(exc) or "rate limit" in str(exc).lower()`

Thresholds:
- **2 consecutive retryable errors** (429 or 5xx) → fail fast, raise immediately
- On success → reset `consecutive_errors = 0`

This is "consecutive-failure counter" pattern — appropriate for a single-run pipeline.
The full GoF half-open pattern is overkill for a pipeline that terminates after execution.

Error message must distinguish: `"Circuit breaker open: 2 consecutive rate-limit errors. Quota likely exceeded."` vs generic node failure — so the API layer can surface a meaningful message to the user.

### 4. Consolidation Token Tracking

Chain: add `tracker: TokenTracker | None` field to `BuiltNode`.
`node_builder.py` sets it from the `tracker` param already passed to `build_node`.
`patterns.py:_run_one` passes `built_node.tracker` down to `consolidate_outputs`,
which passes it to `_llm_consolidate`, which wraps its model in `TrackedOpenAIChatModel`
instead of raw `OpenAIChatModel`.

No changes to `run_execution_step` signature — tracker travels with the node.

### 5. Estimation: Fix Module Global Race

Replace `_last_topology_result: dict` module global in `tools.py` with a
`contextvars.ContextVar`. This is the industry-standard fix for shared mutable
state in async Python — each `asyncio.Task` gets its own copy of the variable,
so concurrent sessions never see each other's values.

```python
from contextvars import ContextVar

_topology_result_var: ContextVar[dict] = ContextVar(
    'daap_topology_result',
    default={"topology": None, "estimate": None},
)
```

`get_last_topology_result()` and `clear_last_topology_result()` keep identical
signatures but use `_topology_result_var.get()` / `_topology_result_var.set()`.
`ws_handler.py` requires zero changes — it calls the same helper functions.

Note: `contextvars` context propagates to child tasks created with
`asyncio.create_task()`. Since each WebSocket session runs its agent in a child
task (`asyncio.create_task(session.master_agent(...))`), the variable is shared
correctly within a session and isolated across sessions.

### 6. Estimation: Model Pricing Coverage

No additional models added (Anthropic/Claude models excluded — cost).
`DEFAULT_PRICING` ($1/$5 per 1M) remains the fallback for unknown models.
Operators using non-registry models see conservative overestimates, which is
acceptable — better to overestimate than underestimate cost.

### 7. Estimation: Adaptive Display Format

In `estimator.py:_build_user_summary` and `tools.py` display strings:

```python
def _format_cost(usd: float) -> str:
    if usd < 0.001:
        return f"${usd:.5f}"
    if usd < 0.10:
        return f"${usd:.4f}"
    return f"${usd:.2f}"
```

Apply to `user_facing_summary`, `generate_topology` tool response text, and
the `ws_handler` plan message.

---

## Data Flow (post-fix)

```
execute_topology()
  ├── build all nodes (tracker attached to BuiltNode)
  └── for each step:
        ├── [NEW] check cost cap → abort if exceeded
        ├── retry loop:
        │     ├── run_execution_step()
        │     │     └── _run_one(built_node)
        │     │           ├── run_parallel_instances()  ← tracked via TrackedModel
        │     │           └── consolidate_outputs(tracker=built_node.tracker)
        │     │                 └── _llm_consolidate()  ← NOW tracked
        │     ├── on success: reset consecutive_errors
        │     └── on 429/5xx: increment consecutive_errors
        │           if >= 2: circuit open → abort
        └── [NEW] post-step cost check
```

---

## Schema Changes

`ConstraintSpec` gets one new field:

```python
max_tokens_per_node: int = 50_000   # per-node token budget (input + output)
```

Enforcement: after each step, if any node's token delta exceeds this, the step
result is accepted but a warning is logged and the budget is flagged.
Full mid-step enforcement requires AgentScope internals — out of scope for this fix.
The cap prevents multi-step bleed but not within-step overrun.

---

## Files Changed

| File | Change |
|------|--------|
| `daap/tools/token_tracker.py` | Add `total_cost_usd(pricing_fn)` |
| `daap/executor/engine.py` | Cost cap check per step; circuit breaker in retry loop |
| `daap/executor/node_builder.py` | Add `tracker` to `BuiltNode`; set in `build_node` |
| `daap/executor/patterns.py` | Pass `tracker` through consolidation chain; use `TrackedOpenAIChatModel` in `_llm_consolidate` |
| `daap/spec/schema.py` | Add `max_tokens_per_node` to `ConstraintSpec` |
| `daap/spec/resolver.py` | Add Claude + common model pricing to `MODEL_PRICING` |
| `daap/spec/estimator.py` | Add `_format_cost()` helper; apply to summary |
| `daap/master/tools.py` | Remove `_last_topology_result` global; keep metadata in `ToolResponse` |
| `daap/api/ws_handler.py` | Read estimate from tool response metadata, not global |

---

## What This Does NOT Change

- AgentScope ReActAgent internals (no mid-iteration token checking)
- Retry count defaults (`max_retries_per_node = 2` stays)
- Topology validation logic
- Memory system
- Session lifecycle

---

## Success Criteria

1. A topology with `max_cost_usd: 0.01` running an expensive 4-node react topology
   aborts after the first step that pushes past $0.01, with a clear cost-exceeded error
2. Two back-to-back 429 errors in the same step retry loop abort immediately with
   "circuit breaker open" message (not after exhausting all retries)
3. Two concurrent sessions each generate different topologies and see their own estimates
4. A topology using `claude-sonnet-4-6` shows a non-zero, non-default-pricing estimate
5. A $0.005 topology displays as `$0.0050`, not `$0.00`
