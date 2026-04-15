# DAAP Phase 2 — P0: Fix Model Tiers + Model Migration

## Context

You are modifying the existing DAAP codebase. Phase 1 is complete with 141 tests passing.

**Read these files first before writing any code:**
- `DAAP_PROJECT_CONTEXT.md` — full architecture overview
- `daap/spec/resolver.py` — MODEL_REGISTRY lives here
- `daap/executor/tracked_model.py` — TrackedOpenAIChatModel
- `daap/executor/node_builder.py` — build_node() creates agents
- `daap/spec/estimator.py` — cost estimation uses pricing constants
- `daap/spec/schema.py` — TopologySpec, NodeSpec, operator_config
- `daap/master/prompts.py` — master agent system prompt

---

## Why This Is P0 (Do This First, Before Anything Else)

### CRITICAL: Gemini 2.0 Flash shuts down June 1, 2026

Google deprecated `google/gemini-2.0-flash-001`. Shutdown date: June 1, 2026. You have ~6 weeks. Every model tier in DAAP currently maps to this model. When endpoint dies, DAAP = dead.

### Model tiers are fake

```python
# Current state in resolver.py — ALL THREE TIERS = SAME MODEL
MODEL_REGISTRY = {
    "fast":     "google/gemini-2.0-flash-001",
    "smart":    "google/gemini-2.0-flash-001",
    "powerful": "google/gemini-2.0-flash-001",
}
```

No actual tier differentiation → no signal for Phase 2 optimization → half of Phase 2 meaningless.

### After P0

- Each tier = genuinely different model with different capability + cost
- Estimator uses real per-model pricing
- Master agent prompt knows real tier costs so it makes smarter topology decisions
- System survives Gemini 2.0 Flash shutdown
- Foundation for Phase 2 dynamic model selection exists

---

## Cheapest Viable Model Lineup (via OpenRouter)

All prices per 1M tokens. All models accessible via OpenRouter's OpenAI-compatible API (your existing setup).

### Recommended Tier Mapping (Ultra-Budget)

| Tier | OpenRouter Model ID | Input $/1M | Output $/1M | Context | Why |
|------|-------------------|------------|-------------|---------|-----|
| `fast` | `google/gemini-2.5-flash-lite` | $0.10 | $0.40 | 1M | Cheapest non-deprecated model with tool calling. Replaced your current Gemini 2.0 Flash. Good for search/extract/format. |
| `smart` | `deepseek/deepseek-v3.2` | $0.26 | $0.38 | 164K | GPT-5 class reasoning at 1/50th cost. Strong tool use + agentic performance. Evaluator/writer/scorer nodes. |
| `powerful` | `google/gemini-2.5-flash` | $0.30 | $2.50 | 1M | Built-in thinking mode. Strong reasoning. Master agent tier. Topology generation needs this. |

**Total cost for a typical 4-node sales outreach run: ~$0.005–$0.02**
(vs Claude tiers which would be $0.10–$0.50 for same run)

### Cost Comparison

| Tier | This lineup | Claude equivalent | Savings |
|------|------------|-------------------|---------|
| fast | $0.10/$0.40 | Haiku $1.00/$5.00 | 10-12x cheaper |
| smart | $0.26/$0.38 | Sonnet $3.00/$15.00 | 11-39x cheaper |
| powerful | $0.30/$2.50 | Opus $5.00/$25.00 | 10-17x cheaper |

### Why these specific models

**Fast = Gemini 2.5 Flash Lite ($0.10/$0.40)**
- Direct successor to your deprecated Gemini 2.0 Flash
- Same provider (Google) → minimal behavior changes
- 1M context window — no truncation issues
- Tool calling supported
- Ultra-low latency (optimized for speed)
- Thinking mode available if needed (disabled by default for speed)

**Smart = DeepSeek V3.2 ($0.26/$0.38)**
- GPT-5 class performance at ~1/50th GPT-5 price
- Strong agentic tool-use (trained with agentic task synthesis pipeline)
- Output pricing ($0.38) actually cheaper than input ($0.26) — unusual, great for writer/evaluator nodes that produce long output
- 90% cache discount on repeated prompts ($0.028 input on cache hit)
- 164K context — sufficient for all DAAP node types
- ⚠️ Privacy caveat: data processed on DeepSeek servers (China). Not an issue for Phase 1/dev. Revisit for enterprise (Phase 3).

**Powerful = Gemini 2.5 Flash ($0.30/$2.50)**
- Best cost/intelligence ratio for master agent work
- Built-in "thinking" mode — configurable reasoning depth
- 1M context window
- Strong structured output + tool use
- Only $0.30 input — topology generation prompts are input-heavy

### Alternative: If you want to avoid DeepSeek (privacy concerns)

| Tier | Model | Input $/1M | Output $/1M |
|------|-------|------------|-------------|
| `fast` | `google/gemini-2.5-flash-lite` | $0.10 | $0.40 |
| `smart` | `google/gemini-2.5-flash` | $0.30 | $2.50 |
| `powerful` | `google/gemini-3-flash-preview` | $0.50 | $3.00 |

All-Google stack. Slightly more expensive but single provider, no China data concerns.

### Alternative: Absolute cheapest possible

| Tier | Model | Input $/1M | Output $/1M |
|------|-------|------------|-------------|
| `fast` | `google/gemini-2.0-flash-lite` | $0.075 | $0.30 |
| `smart` | `google/gemini-2.5-flash-lite` | $0.10 | $0.40 |
| `powerful` | `deepseek/deepseek-v3.2` | $0.26 | $0.38 |

⚠️ Gemini 2.0 Flash Lite also deprecated (same June 1 shutdown). Don't use this.

---

## What You Are Modifying (7 files)

1. `daap/spec/resolver.py` — MODEL_REGISTRY + new MODEL_PRICING dict
2. `daap/spec/estimator.py` — Use MODEL_PRICING for real cost estimation
3. `daap/executor/node_builder.py` — Verify TrackedOpenAIChatModel works with new models
4. `daap/executor/tracked_model.py` — Verify token tracking works across providers
5. `daap/master/prompts.py` — Update master agent system prompt with real tier info + costs
6. `daap/master/agent.py` — Update master agent model if it's hardcoded
7. `daap/tests/` — Update any tests that hardcode `google/gemini-2.0-flash-001`

---

## 1. Update MODEL_REGISTRY + Add MODEL_PRICING — `daap/spec/resolver.py`

### Find this (current code):

```python
MODEL_REGISTRY = {
    "fast":     "google/gemini-2.0-flash-001",
    "smart":    "google/gemini-2.0-flash-001",
    "powerful": "google/gemini-2.0-flash-001",
}
```

### Replace with:

```python
# Model tier → concrete OpenRouter model ID
# Updated: April 2026
# Gemini 2.0 Flash deprecated (shutdown June 1 2026) — migrated to newer models
MODEL_REGISTRY = {
    "fast":     "google/gemini-2.5-flash-lite",     # search, extract, format — cheapest viable
    "smart":    "deepseek/deepseek-v3.2",            # evaluate, score, write — GPT-5 class @ 1/50th cost
    "powerful": "google/gemini-2.5-flash",           # master agent, complex planning — thinking mode
}

# Pricing per 1M tokens (USD) — used by estimator.py
# Source: OpenRouter posted rates, April 2026
# Keep in sync with MODEL_REGISTRY. Add entries for any model that might appear
# in operator_config overrides or saved topologies.
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Active models
    "google/gemini-2.5-flash-lite": {
        "input_per_1m": 0.10,
        "output_per_1m": 0.40,
    },
    "deepseek/deepseek-v3.2": {
        "input_per_1m": 0.26,
        "output_per_1m": 0.38,
    },
    "google/gemini-2.5-flash": {
        "input_per_1m": 0.30,
        "output_per_1m": 2.50,
    },
    # Upgrade options (for future use / operator_config)
    "google/gemini-3-flash-preview": {
        "input_per_1m": 0.50,
        "output_per_1m": 3.00,
    },
    "deepseek/deepseek-v3.2-speciale": {
        "input_per_1m": 0.40,
        "output_per_1m": 1.20,
    },
    # Legacy — backward compat with saved topologies
    "google/gemini-2.0-flash-001": {
        "input_per_1m": 0.10,
        "output_per_1m": 0.40,
    },
}

# Default pricing for unknown models (conservative estimate)
DEFAULT_PRICING = {
    "input_per_1m": 1.00,
    "output_per_1m": 5.00,
}
```

### Add helper function (same file):

```python
def get_model_pricing(model_id: str) -> dict[str, float]:
    """Get pricing for a model ID. Returns DEFAULT_PRICING if model unknown."""
    return MODEL_PRICING.get(model_id, DEFAULT_PRICING)
```

---

## 2. Update Estimator — `daap/spec/estimator.py`

The estimator currently uses placeholder pricing or hardcoded constants. Update it to use MODEL_PRICING from resolver.

### What to change:

Find wherever cost is calculated per node. It should look something like this (may vary):

```python
# Old pattern — flat pricing or placeholder
cost = (input_tokens + output_tokens) * SOME_CONSTANT
```

### Replace cost calculation with:

```python
from daap.spec.resolver import get_model_pricing

def estimate_node_cost(
    model_id: str,
    estimated_input_tokens: int,
    estimated_output_tokens: int,
    parallel_instances: int = 1,
) -> float:
    """Estimate cost for a single node execution in USD."""
    pricing = get_model_pricing(model_id)
    input_cost = (estimated_input_tokens / 1_000_000) * pricing["input_per_1m"]
    output_cost = (estimated_output_tokens / 1_000_000) * pricing["output_per_1m"]
    return (input_cost + output_cost) * parallel_instances
```

### Update token estimation heuristics per node role:

```python
# Token estimation heuristics by node role
# These are rough. Phase 2 replaces with learned values from feedback data.
TOKEN_ESTIMATES = {
    "researcher": {"input": 2000, "output": 3000},   # search prompts + results
    "evaluator":  {"input": 3000, "output": 1500},   # receives data + scores it
    "writer":     {"input": 2500, "output": 2000},   # receives brief + writes output
    "formatter":  {"input": 1500, "output": 1000},   # light transform
    "default":    {"input": 2000, "output": 2000},
}
```

### Update the topology-level estimate function:

```python
def estimate_topology_cost(resolved_topology) -> dict:
    """
    Pre-execution cost + latency estimate for a resolved topology.
    
    Returns:
        {
            "estimated_cost_usd": float,
            "estimated_latency_seconds": float,
            "per_node": {node_id: {"cost_usd": float, "latency_seconds": float}},
            "warnings": list[str],
        }
    """
    total_cost = 0.0
    per_node = {}
    warnings = []
    
    for node in resolved_topology.nodes:
        role_key = _classify_role(node.role)
        tokens = TOKEN_ESTIMATES.get(role_key, TOKEN_ESTIMATES["default"])
        
        node_cost = estimate_node_cost(
            model_id=node.resolved_model_id,
            estimated_input_tokens=tokens["input"],
            estimated_output_tokens=tokens["output"],
            parallel_instances=node.instance_config.parallel_instances,
        )
        
        per_node[node.node_id] = {
            "cost_usd": round(node_cost, 6),
            "model": node.resolved_model_id,
            "instances": node.instance_config.parallel_instances,
        }
        total_cost += node_cost
    
    # Warn if cost seems high for the cheap models
    if total_cost > 0.50:
        warnings.append(
            f"Estimated cost ${total_cost:.2f} is high. "
            f"Consider reducing parallel instances or switching nodes to 'fast' tier."
        )
    
    # Latency estimation (rough)
    # Sequential steps: sum. Parallel nodes in same step: max.
    estimated_latency = _estimate_latency(resolved_topology)
    
    return {
        "estimated_cost_usd": round(total_cost, 6),
        "estimated_latency_seconds": round(estimated_latency, 1),
        "per_node": per_node,
        "warnings": warnings,
    }


def _classify_role(role: str) -> str:
    """Map free-text role description to a known role key for token estimation."""
    role_lower = role.lower()
    if any(w in role_lower for w in ["research", "search", "find", "discover"]):
        return "researcher"
    if any(w in role_lower for w in ["evaluat", "scor", "rank", "qualif"]):
        return "evaluator"
    if any(w in role_lower for w in ["writ", "draft", "compos", "email"]):
        return "writer"
    if any(w in role_lower for w in ["format", "clean", "transform"]):
        return "formatter"
    return "default"


def _estimate_latency(resolved_topology) -> float:
    """
    Estimate total latency by walking execution_order.
    Sequential steps = sum of step latencies.
    Parallel nodes in same step = max (they run concurrently).
    """
    AVG_API_CALL_SECONDS = {
        "google/gemini-2.5-flash-lite": 1.5,    # fastest
        "deepseek/deepseek-v3.2": 3.0,           # moderate
        "google/gemini-2.5-flash": 2.5,           # moderate
    }
    DEFAULT_LATENCY = 3.0
    
    total = 0.0
    for step in resolved_topology.execution_order:
        step_latencies = []
        for node_id in step:
            node = _find_node(resolved_topology, node_id)
            if node is None:
                step_latencies.append(DEFAULT_LATENCY)
                continue
            
            base = AVG_API_CALL_SECONDS.get(
                node.resolved_model_id, DEFAULT_LATENCY
            )
            # react mode = multiple tool calls = longer
            if node.agent_mode == "react":
                base *= node.max_react_iterations * 0.6  # not all iterations used
            
            step_latencies.append(base)
        
        total += max(step_latencies) if step_latencies else 0.0
    
    return total


def _find_node(resolved_topology, node_id: str):
    """Find node by ID in resolved topology."""
    for n in resolved_topology.nodes:
        if n.node_id == node_id:
            return n
    return None
```

---

## 3. Verify Executor Compatibility — `daap/executor/node_builder.py`

### Key concern: multi-provider models via OpenRouter

Your existing setup uses OpenRouter as a unified gateway. All models are accessed via `https://openrouter.ai/api/v1` with `OPENROUTER_API_KEY`. This means:

- **No code change needed for switching models.** OpenRouter handles routing.
- Changing from `google/gemini-2.0-flash-001` to `google/gemini-2.5-flash-lite` or `deepseek/deepseek-v3.2` = just changing the model string.

### What to verify:

1. `build_node()` passes the `resolved_model_id` string to `TrackedOpenAIChatModel`. Confirm this string flows from resolver → node_builder without being overridden.

2. Check if `parallel_tool_calls=False` is still needed:
   - Your PROJECT_CONTEXT says this was Gemini-specific ("Gemini strict: tool call + result count must match exactly")
   - DeepSeek V3.2 handles parallel tool calls fine
   - Gemini 2.5 Flash Lite may also handle them
   - **Recommendation:** Keep `parallel_tool_calls=False` for now. Safer. Revisit in P1 for per-model tool call config.

3. Check if `max_react_iterations` default works for DeepSeek. DeepSeek V3.2 has 8K output limit in chat mode (164K context). If a react loop generates many tool calls, 8K output might truncate. Monitor this in testing.

### Add to node_builder.py:

```python
# Per-model configuration overrides
# Some models have quirks that need specific settings
MODEL_CONFIG_OVERRIDES = {
    "deepseek/deepseek-v3.2": {
        # DeepSeek has 8K max output in chat mode.
        # If react loops hit this limit, node output truncates silently.
        # Monitor in testing. Switch to deepseek-reasoner endpoint (64K output) if needed.
        "max_output_tokens": 8000,
    },
    "google/gemini-2.5-flash-lite": {
        # Thinking mode disabled by default for speed. 
        # Enable via OpenRouter reasoning parameter if node needs it.
    },
    "google/gemini-2.5-flash": {
        # Thinking mode available. For master agent tier, consider enabling.
    },
}
```

---

## 4. Verify Token Tracking — `daap/executor/tracked_model.py`

TrackedOpenAIChatModel subclasses AgentScope's OpenAIChatModel. It captures `ChatUsage` from API responses.

### What to verify:

OpenRouter returns usage data in standard OpenAI format for all providers:

```json
{
  "usage": {
    "prompt_tokens": 1234,
    "completion_tokens": 567,
    "total_tokens": 1801
  }
}
```

This should work unchanged across Gemini/DeepSeek/Claude via OpenRouter. But verify:

1. DeepSeek sometimes returns reasoning tokens separately. Check if `completion_tokens` includes reasoning tokens or if there's a `reasoning_tokens` field. If so, add them to output tracking.

2. Run a quick manual test after migration: send one request per model, check that `TokenTracker.to_dict()` shows correct counts.

---

## 5. Update Master Agent Prompt — `daap/master/prompts.py`

### What to change:

The master agent system prompt tells the LLM what model tiers are available and their characteristics. Currently it probably says something generic or references Gemini for all tiers.

### Find the section about model tiers and replace with:

```python
MODEL_TIER_PROMPT_SECTION = """
## Model Tiers — Use the Right Tier for Each Node

You MUST select the appropriate model tier for each node. Using "powerful" for simple search tasks wastes money. Using "fast" for complex reasoning produces bad output.

| Tier | What It Is | Cost (per 1M tokens) | Use For |
|------|-----------|---------------------|---------|
| "fast" | Gemini 2.5 Flash Lite — ultra-fast, cheapest | $0.10 in / $0.40 out | Web search, data extraction, formatting, simple transforms. Nodes that call tools and collate results. |
| "smart" | DeepSeek V3.2 — GPT-5 class reasoning, very cheap | $0.26 in / $0.38 out | Evaluation, scoring, ranking, email writing, analysis, summarization. Nodes that need good judgment. |
| "powerful" | Gemini 2.5 Flash — thinking mode, strong reasoning | $0.30 in / $2.50 out | Complex planning, multi-step reasoning, synthesis of many inputs. Only when "smart" is insufficient. |

### Decision Rules:
- Default to "fast" for any node that primarily calls tools (search, fetch, read)
- Use "smart" for nodes that must reason about data, write quality output, or make judgments
- Use "powerful" ONLY for nodes that synthesize complex multi-source data or need planning
- When in doubt, use "smart" — it's only $0.26/$0.38, almost as cheap as "fast"

### Cost Awareness:
- A 4-node topology with all "fast" nodes costs ~$0.002
- A 4-node topology with all "smart" nodes costs ~$0.005
- A 4-node topology with all "powerful" nodes costs ~$0.02
- Each parallel instance multiplies node cost (4 instances = 4x cost)
- Keep parallel_instances low (2-4) unless task genuinely benefits from more searchers
"""
```

### Also update the master agent's own model:

In `daap/master/agent.py` (or wherever the master agent's model is configured), change from:

```python
model = "google/gemini-2.0-flash-001"
```

To:

```python
model = "google/gemini-2.5-flash"  # powerful tier for master agent
```

The master agent generates topology specs — it needs the best reasoning available. Gemini 2.5 Flash with thinking mode at $0.30/$2.50 is the right balance.

---

## 6. Update operator_config Handling — `daap/spec/resolver.py`

The `operator_config` field in TopologySpec allows overriding model mappings. Verify the resolution chain still works:

```
node.operator_override → topology.operator_config → MODEL_REGISTRY
```

No code change needed here unless the resolution chain is broken. Just verify:

1. Saved topologies with `google/gemini-2.0-flash-001` in `operator_config` still resolve (they hit the legacy entry in MODEL_PRICING)
2. New topologies without `operator_config` use the updated MODEL_REGISTRY

---

## 7. Update Tests

### Find all tests that hardcode `google/gemini-2.0-flash-001`

```bash
grep -r "gemini-2.0-flash-001" daap/tests/
grep -r "gemini-2.0-flash" daap/tests/
```

### Replace with appropriate model IDs:

- Tests for "fast" tier → `google/gemini-2.5-flash-lite`
- Tests for "smart" tier → `deepseek/deepseek-v3.2`
- Tests for "powerful" tier → `google/gemini-2.5-flash`
- Tests that just need ANY valid model → `google/gemini-2.5-flash-lite` (cheapest)

### New tests to add:

```python
# test_model_tiers.py

def test_model_registry_has_three_tiers():
    """MODEL_REGISTRY must have fast, smart, powerful."""
    from daap.spec.resolver import MODEL_REGISTRY
    assert "fast" in MODEL_REGISTRY
    assert "smart" in MODEL_REGISTRY
    assert "powerful" in MODEL_REGISTRY


def test_all_tiers_are_different_models():
    """Each tier must map to a genuinely different model."""
    from daap.spec.resolver import MODEL_REGISTRY
    models = list(MODEL_REGISTRY.values())
    assert len(set(models)) == 3, (
        f"Tiers must map to 3 different models, got: {models}"
    )


def test_all_registry_models_have_pricing():
    """Every model in MODEL_REGISTRY must have an entry in MODEL_PRICING."""
    from daap.spec.resolver import MODEL_REGISTRY, MODEL_PRICING
    for tier, model_id in MODEL_REGISTRY.items():
        assert model_id in MODEL_PRICING, (
            f"Tier '{tier}' uses model '{model_id}' but MODEL_PRICING has no entry for it"
        )


def test_pricing_structure():
    """Each pricing entry must have input_per_1m and output_per_1m as positive floats."""
    from daap.spec.resolver import MODEL_PRICING
    for model_id, pricing in MODEL_PRICING.items():
        assert "input_per_1m" in pricing, f"{model_id} missing input_per_1m"
        assert "output_per_1m" in pricing, f"{model_id} missing output_per_1m"
        assert pricing["input_per_1m"] > 0, f"{model_id} input_per_1m must be positive"
        assert pricing["output_per_1m"] > 0, f"{model_id} output_per_1m must be positive"


def test_fast_tier_is_cheapest():
    """Fast tier should be cheapest (or equal) on input pricing."""
    from daap.spec.resolver import MODEL_REGISTRY, MODEL_PRICING
    fast_price = MODEL_PRICING[MODEL_REGISTRY["fast"]]["input_per_1m"]
    smart_price = MODEL_PRICING[MODEL_REGISTRY["smart"]]["input_per_1m"]
    powerful_price = MODEL_PRICING[MODEL_REGISTRY["powerful"]]["input_per_1m"]
    assert fast_price <= smart_price, "Fast tier should be <= smart on input cost"
    assert fast_price <= powerful_price, "Fast tier should be <= powerful on input cost"


def test_get_model_pricing_known():
    """get_model_pricing returns correct pricing for known models."""
    from daap.spec.resolver import get_model_pricing
    p = get_model_pricing("google/gemini-2.5-flash-lite")
    assert p["input_per_1m"] == 0.10
    assert p["output_per_1m"] == 0.40


def test_get_model_pricing_unknown():
    """get_model_pricing returns DEFAULT_PRICING for unknown models."""
    from daap.spec.resolver import get_model_pricing, DEFAULT_PRICING
    p = get_model_pricing("nonexistent/model-xyz")
    assert p == DEFAULT_PRICING


def test_legacy_model_has_pricing():
    """Deprecated gemini-2.0-flash-001 still has pricing (backward compat)."""
    from daap.spec.resolver import MODEL_PRICING
    assert "google/gemini-2.0-flash-001" in MODEL_PRICING


def test_estimate_node_cost_basic():
    """estimate_node_cost returns correct value for known model."""
    from daap.spec.estimator import estimate_node_cost
    # 1000 input + 1000 output on gemini-2.5-flash-lite
    # = (1000/1M * 0.10) + (1000/1M * 0.40) = 0.0001 + 0.0004 = 0.0005
    cost = estimate_node_cost(
        model_id="google/gemini-2.5-flash-lite",
        estimated_input_tokens=1000,
        estimated_output_tokens=1000,
        parallel_instances=1,
    )
    assert abs(cost - 0.0005) < 0.00001


def test_estimate_node_cost_parallel():
    """Parallel instances multiply cost."""
    from daap.spec.estimator import estimate_node_cost
    cost_1 = estimate_node_cost("google/gemini-2.5-flash-lite", 1000, 1000, 1)
    cost_4 = estimate_node_cost("google/gemini-2.5-flash-lite", 1000, 1000, 4)
    assert abs(cost_4 - cost_1 * 4) < 0.00001


def test_no_deprecated_models_in_registry():
    """MODEL_REGISTRY must not reference deprecated models."""
    from daap.spec.resolver import MODEL_REGISTRY
    deprecated = ["google/gemini-2.0-flash-001", "google/gemini-2.0-flash-lite"]
    for tier, model_id in MODEL_REGISTRY.items():
        assert model_id not in deprecated, (
            f"Tier '{tier}' uses deprecated model '{model_id}'"
        )
```

---

## 8. Manual Smoke Test (After All Changes)

Run these after code changes, before declaring P0 done:

### 8a. Unit tests pass
```bash
pytest daap/tests/ -v
```

All 141+ existing tests pass (with updated model IDs). New model tier tests pass.

### 8b. CLI smoke test
```bash
python scripts/chat.py
```

Type: "Find 3 B2B leads for a project management SaaS targeting construction companies"

Verify:
- Master agent generates topology (not crash)
- Topology shows different model_tier values for different nodes
- Cost estimate is in the $0.005–$0.05 range (not $0 or $100)
- If you approve + execute: nodes run without API errors

### 8c. Model-specific check

After execution, check `session.token_tracker.to_dict()` (or look at CLI output):
- `models_used` should list 2-3 different model IDs (not one repeated)
- Token counts should be reasonable (not 0, not millions)

---

## Acceptance Criteria

P0 is done when:

- [ ] MODEL_REGISTRY has 3 different, non-deprecated models
- [ ] MODEL_PRICING has entries for all registry models + legacy model
- [ ] Estimator uses real per-model pricing
- [ ] Master agent system prompt describes real tier capabilities + costs
- [ ] Master agent model = `google/gemini-2.5-flash` (not deprecated model)
- [ ] All existing tests pass with updated model IDs
- [ ] 12 new model tier tests pass
- [ ] CLI smoke test: topology generates, different tiers assigned, execution works
- [ ] No reference to `google/gemini-2.0-flash-001` in MODEL_REGISTRY (only in legacy pricing)

---

## What This Does NOT Cover (Handled in Later Phases)

- Per-model tool call configuration (parallel_tool_calls per provider) → P1
- Dynamic model selection learning from feedback → P2
- A/B testing different model tiers → P2
- Model fallback chains (if DeepSeek is down, fall back to Gemini) → P3
- Per-node model override in UI → P3

---

## Decision for Builder

**Pick one of the three tier mappings before starting:**

1. **Recommended (mixed provider, cheapest):** Gemini Flash Lite + DeepSeek V3.2 + Gemini 2.5 Flash
2. **All-Google (simpler, slightly pricier):** Gemini Flash Lite + Gemini 2.5 Flash + Gemini 3 Flash Preview  
3. **Your own choice** — but ensure tiers are genuinely different models with increasing capability

The build instructions above use option 1. If you pick option 2, replace all `deepseek/deepseek-v3.2` references with `google/gemini-2.5-flash` and adjust pricing accordingly.
