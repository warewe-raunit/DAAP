"""
Tests for model tier registry and pricing — P0 migration validation.

These tests enforce:
- 3 genuinely different model tiers
- No deprecated models in registry
- Every registry model has pricing
- Estimator uses real per-model pricing
"""

import pytest


def test_model_registry_has_three_tiers():
    from daap.spec.resolver import MODEL_REGISTRY
    assert "fast" in MODEL_REGISTRY
    assert "smart" in MODEL_REGISTRY
    assert "powerful" in MODEL_REGISTRY


def test_all_tiers_are_different_models():
    from daap.spec.resolver import MODEL_REGISTRY
    models = list(MODEL_REGISTRY.values())
    assert len(set(models)) == 3, (
        f"Tiers must map to 3 different models, got: {models}"
    )


def test_all_registry_models_have_pricing():
    from daap.spec.resolver import MODEL_REGISTRY, MODEL_PRICING
    for tier, model_id in MODEL_REGISTRY.items():
        assert model_id in MODEL_PRICING, (
            f"Tier '{tier}' uses model '{model_id}' but MODEL_PRICING has no entry for it"
        )


def test_pricing_structure():
    from daap.spec.resolver import MODEL_PRICING
    for model_id, pricing in MODEL_PRICING.items():
        assert "input_per_1m" in pricing, f"{model_id} missing input_per_1m"
        assert "output_per_1m" in pricing, f"{model_id} missing output_per_1m"
        assert pricing["input_per_1m"] >= 0, f"{model_id} input_per_1m must be non-negative"
        assert pricing["output_per_1m"] >= 0, f"{model_id} output_per_1m must be non-negative"


def test_fast_tier_is_cheapest_on_input():
    from daap.spec.resolver import MODEL_REGISTRY, MODEL_PRICING
    fast_price = MODEL_PRICING[MODEL_REGISTRY["fast"]]["input_per_1m"]
    smart_price = MODEL_PRICING[MODEL_REGISTRY["smart"]]["input_per_1m"]
    powerful_price = MODEL_PRICING[MODEL_REGISTRY["powerful"]]["input_per_1m"]
    assert fast_price <= smart_price, "Fast tier should be <= smart on input cost"
    assert fast_price <= powerful_price, "Fast tier should be <= powerful on input cost"


def test_get_model_pricing_known():
    from daap.spec.resolver import get_model_pricing
    p = get_model_pricing("google/gemini-2.5-flash-lite")
    assert p["input_per_1m"] == 0.10
    assert p["output_per_1m"] == 0.40


def test_get_model_pricing_deepseek():
    from daap.spec.resolver import get_model_pricing
    p = get_model_pricing("deepseek/deepseek-v3.2")
    assert p["input_per_1m"] == 0.26
    assert p["output_per_1m"] == 0.38


def test_get_model_pricing_gemini_25_flash():
    from daap.spec.resolver import get_model_pricing
    p = get_model_pricing("google/gemini-2.5-flash")
    assert p["input_per_1m"] == 0.30
    assert p["output_per_1m"] == 2.50


def test_get_model_pricing_unknown_returns_default():
    from daap.spec.resolver import get_model_pricing, DEFAULT_PRICING
    p = get_model_pricing("nonexistent/model-xyz")
    assert p == DEFAULT_PRICING


def test_legacy_model_has_pricing():
    from daap.spec.resolver import MODEL_PRICING
    assert "google/gemini-2.0-flash-001" in MODEL_PRICING


def test_no_deprecated_models_in_registry():
    from daap.spec.resolver import MODEL_REGISTRY
    deprecated = ["google/gemini-2.0-flash-001", "google/gemini-2.0-flash-lite"]
    for tier, model_id in MODEL_REGISTRY.items():
        assert model_id not in deprecated, (
            f"Tier '{tier}' uses deprecated model '{model_id}' (shuts down June 1 2026)"
        )


def test_estimate_node_cost_basic():
    from daap.spec.estimator import _get_pricing
    pricing = _get_pricing("google/gemini-2.5-flash-lite")
    input_tokens = 1000
    output_tokens = 1000
    cost = (
        input_tokens * pricing["input_per_1m"] / 1_000_000
        + output_tokens * pricing["output_per_1m"] / 1_000_000
    )
    # (1000/1M * 0.10) + (1000/1M * 0.40) = 0.0001 + 0.0004 = 0.0005
    assert abs(cost - 0.0005) < 0.00001


def test_model_latency_heuristics_exist():
    from daap.spec.estimator import MODEL_LATENCY_SECONDS
    from daap.spec.resolver import MODEL_REGISTRY
    for tier, model_id in MODEL_REGISTRY.items():
        assert model_id in MODEL_LATENCY_SECONDS, (
            f"No latency heuristic for '{model_id}' (tier: {tier})"
        )


def test_master_agent_uses_powerful_model():
    import importlib.util, sys
    from daap.spec.resolver import MODEL_REGISTRY
    # Read the constant directly without importing agentscope-dependent modules
    spec = importlib.util.find_spec("daap.master.agent")
    source = open(spec.origin).read()
    powerful_model = MODEL_REGISTRY["powerful"]
    assert powerful_model in source, (
        f"daap/master/agent.py should reference powerful tier model '{powerful_model}'"
    )
