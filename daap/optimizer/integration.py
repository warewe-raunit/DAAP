"""
Integration layer: connects optimizer to DAAP's existing flow.

Touch point 1: BEFORE execution — get tier recommendations per role
Touch point 2: AFTER user rates a run — compute reward, update bandits, persist
"""

import logging

from daap.optimizer.bandit import TopologyOptimizer
from daap.optimizer.context import CONTEXT_DIM, compute_reward, extract_context
from daap.optimizer.store import BanditStore

logger = logging.getLogger(__name__)

_store = BanditStore()
_optimizer_cache: dict[str, TopologyOptimizer] = {}


def get_tier_recommendations(
    user_id: str,
    user_prompt: str,
    proposed_roles: list[str],
    node_count: int,
    has_parallel: bool,
) -> dict[str, str]:
    """
    Get model tier recommendations for a proposed topology.

    Called BEFORE execution. Returns {role: recommended_tier}.
    """
    optimizer = _load_optimizer(user_id)
    run_count = _store.get_user_run_count(user_id)

    ctx = extract_context(
        user_prompt=user_prompt,
        node_count=node_count,
        has_parallel=has_parallel,
        user_total_runs=run_count,
    )

    return optimizer.recommend(context=ctx.to_vector(), roles=proposed_roles)


def record_run_outcome(
    user_id: str,
    user_prompt: str,
    node_configs: dict[str, str],
    user_rating: int,
    actual_cost_usd: float,
    budget_usd: float,
    latency_seconds: float,
    timeout_seconds: float,
    topology_id: str = None,
    node_count: int = 1,
    has_parallel: bool = False,
):
    """
    Update optimizer after a rated run.

    Called AFTER user submits a rating (POST /rate).
    """
    optimizer = _load_optimizer(user_id)
    run_count = _store.get_user_run_count(user_id)

    ctx = extract_context(
        user_prompt=user_prompt,
        node_count=node_count,
        has_parallel=has_parallel,
        user_total_runs=run_count,
    )
    ctx_vec = ctx.to_vector()

    reward = compute_reward(
        user_rating=user_rating,
        actual_cost_usd=actual_cost_usd,
        budget_usd=budget_usd,
        latency_seconds=latency_seconds,
        timeout_seconds=timeout_seconds,
    )

    optimizer.update(context=ctx_vec, node_configs=node_configs, reward=reward)

    for role, tier in node_configs.items():
        _store.log_observation(
            user_id=user_id,
            context_vec=ctx_vec,
            role=role,
            arm=tier,
            reward=reward,
            topology_id=topology_id,
        )

    _store.save_optimizer(user_id, optimizer)


def format_recommendations_for_prompt(recs: dict[str, str]) -> str:
    """
    Format recommendations as text to inject into master agent system prompt.

    The master agent is NOT forced to follow these. They're guidance.
    """
    if not recs:
        return ""

    lines = [
        "\n## Learned Tier Recommendations (from past runs)",
        "Based on this user's history, the optimizer suggests:",
    ]
    for role, tier in recs.items():
        lines.append(f"  - {role} nodes → \"{tier}\" tier")
    lines.append(
        "These are suggestions, not requirements. "
        "Override if the task clearly needs a different tier."
    )
    return "\n".join(lines)


def _load_optimizer(user_id: str) -> TopologyOptimizer:
    """Load optimizer from cache or DB."""
    if user_id in _optimizer_cache:
        return _optimizer_cache[user_id]

    optimizer = TopologyOptimizer(dim=CONTEXT_DIM)

    saved = _store.load_optimizer(user_id, dim=CONTEXT_DIM)
    for role, arms in saved.items():
        bandit = optimizer._get_or_create_bandit(role)
        for arm_name, (B, f, n_pulls) in arms.items():
            if arm_name in bandit.arm_states:
                state = bandit.arm_states[arm_name]
                state.B = B
                state.f = f
                state.n_pulls = n_pulls

    _optimizer_cache[user_id] = optimizer
    return optimizer
