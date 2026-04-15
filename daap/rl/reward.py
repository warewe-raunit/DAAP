"""
DAAP RL Reward — multi-objective scalar reward computation.

Reward formula:
  rated:   score = rating_norm * 0.6 + cost_efficiency * 0.2 + latency_efficiency * 0.2
  unrated: score = success * 0.5 + cost_efficiency * 0.25 + latency_efficiency * 0.25

All sub-scores in [0.0, 1.0]. Final reward in [0.0, 1.0].
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Cost normalization bounds (USD)
COST_BASELINE_USD: float = 0.001   # cheapest realistic run → efficiency = 1.0
COST_CEILING_USD: float = 0.50     # unacceptably expensive → efficiency = 0.0

# Latency normalization bounds (seconds)
LATENCY_BASELINE_SECONDS: float = 5.0    # theoretical minimum for 3-node fast topology
LATENCY_CEILING_SECONDS: float = 120.0   # ConstraintSpec default max_total_latency


def compute_reward(
    result: dict,
    topology: dict | None = None,
    rating: int | None = None,
) -> float:
    """
    Compute a scalar reward in [0.0, 1.0] for a topology execution outcome.

    Args:
        result:   Execution result dict. Keys used:
                    success (bool), latency_seconds (float),
                    total_input_tokens (int), total_output_tokens (int)
        topology: Raw TopologySpec dict (reserved for Phase 4 cost lookups).
        rating:   Optional user rating 1-5. If None, uses implicit reward.

    Returns:
        Scalar reward in [0.0, 1.0]. Returns 0.5 on any exception.
    """
    try:
        result_dict = result if isinstance(result, dict) else {}

        success = bool(result_dict.get("success", False))
        latency = _to_float(
            result_dict.get(
                "latency_seconds",
                result_dict.get("total_latency_seconds", 0.0),
            )
        )
        input_tokens = _to_int(result_dict.get("total_input_tokens", 0))
        output_tokens = _to_int(result_dict.get("total_output_tokens", 0))

        cost_usd = _to_float(result_dict.get("cost_usd", result_dict.get("total_cost_usd")))
        if cost_usd <= 0:
            cost_usd = _estimate_cost_from_tokens(input_tokens, output_tokens)

        cost_eff = _cost_efficiency(cost_usd)
        lat_eff = _latency_efficiency(latency)

        if rating is not None:
            rating_norm = _normalize_rating(rating)
            reward = rating_norm * 0.6 + cost_eff * 0.2 + lat_eff * 0.2
        else:
            success_score = 1.0 if success else 0.0
            reward = success_score * 0.5 + cost_eff * 0.25 + lat_eff * 0.25

        return float(max(0.0, min(1.0, reward)))

    except Exception:
        logger.exception("reward computation failed; returning 0.5")
        return 0.5


def _normalize_rating(rating: int) -> float:
    """Map 1-5 rating to [0.0, 1.0]. 1 → 0.0, 3 → 0.5, 5 → 1.0."""
    clamped = max(1, min(5, int(rating)))
    return (clamped - 1) / 4.0


def _cost_efficiency(cost_usd: float) -> float:
    """
    Map actual cost to [0.0, 1.0] efficiency.
    At or below COST_BASELINE → 1.0. At or above COST_CEILING → 0.0.
    Linear between.
    """
    if cost_usd <= COST_BASELINE_USD:
        return 1.0
    if cost_usd >= COST_CEILING_USD:
        return 0.0
    return 1.0 - (cost_usd - COST_BASELINE_USD) / (COST_CEILING_USD - COST_BASELINE_USD)


def _latency_efficiency(latency_seconds: float) -> float:
    """
    Map actual latency to [0.0, 1.0] efficiency.
    At or below LATENCY_BASELINE → 1.0. At or above LATENCY_CEILING → 0.0.
    Linear between.
    """
    if latency_seconds <= LATENCY_BASELINE_SECONDS:
        return 1.0
    if latency_seconds >= LATENCY_CEILING_SECONDS:
        return 0.0
    return 1.0 - (latency_seconds - LATENCY_BASELINE_SECONDS) / (
        LATENCY_CEILING_SECONDS - LATENCY_BASELINE_SECONDS
    )


def _estimate_cost_from_tokens(input_tokens: int, output_tokens: int) -> float:
    """
    Estimate cost from token counts assuming smart-tier pricing
    (DeepSeek V3.2: $0.26/1M input, $0.38/1M output).

    Conservative overestimate — biases the reward toward flagging expensive runs.
    """
    safe_input = max(0, int(input_tokens))
    safe_output = max(0, int(output_tokens))
    return (safe_input * 0.26 + safe_output * 0.38) / 1_000_000


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _to_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except Exception:
        return 0
