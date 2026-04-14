"""
DAAP Estimator — pre-execution cost and latency estimation for resolved topologies.

Uses heuristics to estimate token counts per node role, then prices them
against the operator's model pricing. Phase 2 replaces heuristics with
learned values from real execution data.

Zero external dependencies — no LLM calls, no network.
"""

from dataclasses import dataclass, field

from daap.spec.resolver import ResolvedTopology, ResolvedNode


# ---------------------------------------------------------------------------
# Pricing — per 1M tokens (TODO: verify current pricing from provider docs)
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic (via OpenRouter)
    "claude-haiku-4-5-20251001":            {"input_per_1m": 1.00,  "output_per_1m": 5.00},
    "anthropic/claude-haiku-4-5-20251001":  {"input_per_1m": 1.00,  "output_per_1m": 5.00},
    "claude-sonnet-4-6":                    {"input_per_1m": 3.00,  "output_per_1m": 15.00},
    "anthropic/claude-sonnet-4-6":          {"input_per_1m": 3.00,  "output_per_1m": 15.00},
    "claude-opus-4-6":                      {"input_per_1m": 15.00, "output_per_1m": 75.00},
    "anthropic/claude-opus-4-6":            {"input_per_1m": 15.00, "output_per_1m": 75.00},
    # Google Gemini (via OpenRouter)
    "google/gemini-2.0-flash-001":          {"input_per_1m": 0.10,  "output_per_1m": 0.40},
    "google/gemini-flash-1.5":              {"input_per_1m": 0.075, "output_per_1m": 0.30},
    # OpenAI (via OpenRouter)
    "openai/gpt-4o":                        {"input_per_1m": 2.50,  "output_per_1m": 10.00},
    "openai/gpt-4o-mini":                   {"input_per_1m": 0.15,  "output_per_1m": 0.60},
    # Free router
    "openrouter/free":                      {"input_per_1m": 0.00,  "output_per_1m": 0.00},
    # Qwen
    "qwen/qwen3.5-plus-02-15":             {"input_per_1m": 0.26,  "output_per_1m": 1.56},
}

# Fallback pricing when model not in registry
_FALLBACK_PRICING = {"input_per_1m": 0.10, "output_per_1m": 0.40}


# ---------------------------------------------------------------------------
# Heuristics — rough estimates; Phase 2 replaces with learned values
# ---------------------------------------------------------------------------

ROLE_HEURISTICS: dict[str, dict[str, int]] = {
    "researcher": {"input_tokens": 2000, "output_tokens": 3000, "tool_calls": 5},
    "evaluator":  {"input_tokens": 3000, "output_tokens": 2000, "tool_calls": 3},
    "writer":     {"input_tokens": 3000, "output_tokens": 2000, "tool_calls": 0},
    "drafter":    {"input_tokens": 3000, "output_tokens": 2000, "tool_calls": 0},
    "parser":     {"input_tokens": 1000, "output_tokens": 1000, "tool_calls": 0},
    "synthesizer":{"input_tokens": 3000, "output_tokens": 2000, "tool_calls": 1},
    "default":    {"input_tokens": 2000, "output_tokens": 2000, "tool_calls": 2},
}

AVG_LATENCY_PER_API_CALL_SECONDS: float = 3.0
AVG_LATENCY_PER_TOOL_CALL_SECONDS: float = 2.0

RISK_ORDER = {"free": 0, "low": 1, "high": 2, "destructive": 3}


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class NodeEstimate:
    """Cost and latency estimate for a single node."""
    node_id: str
    model_id: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_tool_calls: int
    estimated_cost_usd: float       # total node cost (per-instance * instance_count)
    estimated_latency_seconds: float # single instance latency (parallel doesn't multiply)
    instance_count: int


@dataclass
class CostSuggestion:
    """A cost reduction suggestion with risk classification."""
    description: str
    risk: str                       # "free" | "low" | "high" | "destructive"
    reason: str
    estimated_savings_usd: float
    node_id: str


@dataclass
class LatencySuggestion:
    """A latency reduction suggestion."""
    description: str
    risk: str
    reason: str
    estimated_savings_seconds: float
    node_id: str | None             # None = topology-level suggestion


@dataclass
class TopologyEstimate:
    """Cost and latency estimate for the full topology."""
    total_cost_usd: float
    total_latency_seconds: float
    node_estimates: list[NodeEstimate]
    within_budget: bool
    within_timeout: bool
    cost_suggestions: list[CostSuggestion]      # sorted: free → low → high → destructive
    latency_suggestions: list[LatencySuggestion]
    user_facing_summary: str
    min_viable_cost_usd: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_heuristics(role: str) -> dict[str, int]:
    role_lower = role.lower()
    for keyword, h in ROLE_HEURISTICS.items():
        if keyword in role_lower:
            return h
    return ROLE_HEURISTICS["default"]


def _get_pricing(model_id: str) -> dict[str, float]:
    return MODEL_PRICING.get(model_id, _FALLBACK_PRICING)


def _estimate_node(node: ResolvedNode) -> NodeEstimate:
    h = _get_heuristics(node.role)
    pricing = _get_pricing(node.concrete_model_id)

    # Add system prompt token estimate
    prompt_tokens = len(node.system_prompt) // 4
    input_tokens = h["input_tokens"] + prompt_tokens
    output_tokens = h["output_tokens"]
    tool_calls = h["tool_calls"]

    cost_per_instance = (
        input_tokens * pricing["input_per_1m"] / 1_000_000
        + output_tokens * pricing["output_per_1m"] / 1_000_000
    )

    # ReAct overhead: tool loop adds extra LLM calls
    if node.agent_mode == "react":
        cost_per_instance *= (1 + tool_calls * 0.3)

    total_cost = cost_per_instance * node.parallel_instances

    # Latency: parallel instances run concurrently = single instance time
    latency = AVG_LATENCY_PER_API_CALL_SECONDS
    if node.agent_mode == "react":
        latency += tool_calls * AVG_LATENCY_PER_TOOL_CALL_SECONDS

    return NodeEstimate(
        node_id=node.node_id,
        model_id=node.concrete_model_id,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_tool_calls=tool_calls,
        estimated_cost_usd=round(total_cost, 6),
        estimated_latency_seconds=round(latency, 2),
        instance_count=node.parallel_instances,
    )


def _build_user_summary(resolved: ResolvedTopology, node_estimates: list[NodeEstimate],
                        total_cost: float, total_latency: float) -> str:
    node_map = {n.node_id: n for n in resolved.nodes}
    steps = []
    for group in resolved.execution_order:
        parts = []
        for nid in group:
            node = node_map[nid]
            count = node.parallel_instances
            role = node.role
            if count > 1:
                parts.append(f"{count} {role}s (parallel)")
            else:
                parts.append(f"1 {role}")
        steps.append(" + ".join(parts))
    plan = " → ".join(steps)
    return (
        f"Plan: {plan}. "
        f"Estimated cost: ${total_cost:.2f} | "
        f"Estimated time: ~{total_latency:.0f} seconds."
    )


def _min_viable_cost(resolved: ResolvedTopology) -> float:
    """Floor estimate: every node on fast/single/1-instance."""
    fast_pricing = _get_pricing("claude-haiku-4-5-20251001")
    total = 0.0
    for node in resolved.nodes:
        h = _get_heuristics(node.role)
        prompt_tokens = len(node.system_prompt) // 4
        input_tokens = h["input_tokens"] + prompt_tokens
        output_tokens = h["output_tokens"]
        cost = (
            input_tokens * fast_pricing["input_per_1m"] / 1_000_000
            + output_tokens * fast_pricing["output_per_1m"] / 1_000_000
        )
        total += cost
    return round(total, 6)


def _generate_cost_suggestions(
    resolved: ResolvedTopology,
    node_estimates: list[NodeEstimate],
) -> list[CostSuggestion]:
    suggestions: list[CostSuggestion] = []
    est_map = {e.node_id: e for e in node_estimates}
    node_map = {n.node_id: n for n in resolved.nodes}

    for node in resolved.nodes:
        est = est_map[node.node_id]
        h = _get_heuristics(node.role)

        # FREE: react mode with very low tool call heuristic → switch to single
        if node.agent_mode == "react" and h["tool_calls"] <= 1:
            pricing = _get_pricing(node.concrete_model_id)
            react_overhead = est.estimated_cost_usd * (node.parallel_instances * 0.3 * h["tool_calls"])
            savings = round(react_overhead / max(node.parallel_instances, 1), 6)
            suggestions.append(CostSuggestion(
                description=f"Switch node '{node.node_id}' from react to single mode",
                risk="free",
                reason=f"Role '{node.role}' uses ≤1 tool call on average. Single mode skips the tool loop overhead.",
                estimated_savings_usd=savings,
                node_id=node.node_id,
            ))

        # FREE: max_react_iterations much higher than expected tool calls
        if node.agent_mode == "react" and node.max_react_iterations > h["tool_calls"] * 3:
            suggestions.append(CostSuggestion(
                description=f"Reduce max_react_iterations on '{node.node_id}' from {node.max_react_iterations} to {max(h['tool_calls'] * 2, 3)}",
                risk="free",
                reason=f"Expected ~{h['tool_calls']} tool calls. Current cap of {node.max_react_iterations} is overly generous.",
                estimated_savings_usd=0.0,
                node_id=node.node_id,
            ))

        # LOW: reduce parallel instances from ≥3
        if node.parallel_instances >= 3:
            per_instance_cost = est.estimated_cost_usd / node.parallel_instances
            suggestions.append(CostSuggestion(
                description=f"Reduce '{node.node_id}' from {node.parallel_instances} to {node.parallel_instances - 1} parallel instances",
                risk="low",
                reason=f"Diminishing returns. {node.parallel_instances - 1} instances still provide good coverage.",
                estimated_savings_usd=round(per_instance_cost, 6),
                node_id=node.node_id,
            ))

        # LOW: powerful tier for non-planning role → suggest smart
        if node.concrete_model_id == MODEL_PRICING.get("claude-opus-4-6") or "opus" in node.concrete_model_id:
            planning_keywords = ["master", "planner", "orchestrat", "architect"]
            is_planning = any(kw in node.role.lower() for kw in planning_keywords)
            if not is_planning:
                opus_pricing = _get_pricing(node.concrete_model_id)
                sonnet_pricing = _get_pricing("claude-sonnet-4-6")
                savings = est.estimated_cost_usd * (
                    1 - sonnet_pricing["input_per_1m"] / opus_pricing["input_per_1m"]
                )
                suggestions.append(CostSuggestion(
                    description=f"Downgrade '{node.node_id}' from powerful to smart model tier",
                    risk="low",
                    reason=f"Role '{node.role}' likely doesn't need maximum reasoning depth.",
                    estimated_savings_usd=round(max(savings, 0), 6),
                    node_id=node.node_id,
                ))

        # HIGH: smart tier for evaluator/reasoning → suggest fast
        if "sonnet" in node.concrete_model_id or "smart" in node.concrete_model_id:
            evaluation_keywords = ["evaluat", "scor", "rank", "classif", "filter"]
            is_evaluator = any(kw in node.role.lower() for kw in evaluation_keywords)
            if is_evaluator:
                suggestions.append(CostSuggestion(
                    description=f"Downgrade '{node.node_id}' from smart to fast model tier",
                    risk="high",
                    reason=f"WARNING: Evaluation quality will degrade. Fast model has weaker reasoning.",
                    estimated_savings_usd=round(est.estimated_cost_usd * 0.67, 6),
                    node_id=node.node_id,
                ))

        # HIGH: reduce parallel from 2 to 1
        if node.parallel_instances == 2:
            per_instance_cost = est.estimated_cost_usd / 2
            suggestions.append(CostSuggestion(
                description=f"Reduce '{node.node_id}' from 2 to 1 parallel instance",
                risk="high",
                reason=f"WARNING: Single instance provides no redundancy or coverage breadth.",
                estimated_savings_usd=round(per_instance_cost, 6),
                node_id=node.node_id,
            ))

        # DESTRUCTIVE: remove node entirely
        suggestions.append(CostSuggestion(
            description=f"Remove node '{node.node_id}' from the topology",
            risk="destructive",
            reason=f"DESTRUCTIVE: Removing '{node.role}' eliminates this stage from the pipeline entirely.",
            estimated_savings_usd=round(est.estimated_cost_usd, 6),
            node_id=node.node_id,
        ))

    suggestions.sort(key=lambda s: RISK_ORDER[s.risk])
    return suggestions


def _generate_latency_suggestions(
    resolved: ResolvedTopology,
    node_estimates: list[NodeEstimate],
) -> list[LatencySuggestion]:
    suggestions: list[LatencySuggestion] = []
    node_map = {n.node_id: n for n in resolved.nodes}

    for node in resolved.nodes:
        h = _get_heuristics(node.role)

        # FREE: switch to faster model (if I/O bound, not reasoning bound)
        if node.agent_mode == "react" and h["tool_calls"] >= 3:
            suggestions.append(LatencySuggestion(
                description=f"Switch '{node.node_id}' to fast model tier",
                risk="free",
                reason=f"Latency for '{node.role}' is dominated by tool calls, not model inference speed.",
                estimated_savings_seconds=1.0,
                node_id=node.node_id,
            ))

        # LOW: reduce max_react_iterations
        if node.agent_mode == "react" and node.max_react_iterations > 5:
            savings = (node.max_react_iterations - 5) * AVG_LATENCY_PER_TOOL_CALL_SECONDS * 0.3
            suggestions.append(LatencySuggestion(
                description=f"Reduce max_react_iterations on '{node.node_id}' to 5",
                risk="low",
                reason=f"Fewer allowed iterations caps worst-case latency for this node.",
                estimated_savings_seconds=round(savings, 1),
                node_id=node.node_id,
            ))

        # HIGH: skip pipeline stages
        suggestions.append(LatencySuggestion(
            description=f"Remove node '{node.node_id}' from topology",
            risk="high",
            reason=f"WARNING: Skipping '{node.role}' stage reduces quality significantly.",
            estimated_savings_seconds=AVG_LATENCY_PER_API_CALL_SECONDS + h["tool_calls"] * AVG_LATENCY_PER_TOOL_CALL_SECONDS,
            node_id=node.node_id,
        ))

    suggestions.sort(key=lambda s: RISK_ORDER[s.risk])
    return suggestions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_topology(resolved: ResolvedTopology) -> TopologyEstimate:
    """
    Estimate cost and latency for a resolved topology.

    Cost = sum of per-node costs (parallel instances multiply cost).
    Latency = critical path: each step's latency = slowest node in that step.
    """
    node_map = {n.node_id: n for n in resolved.nodes}
    node_estimates = [_estimate_node(n) for n in resolved.nodes]
    est_map = {e.node_id: e for e in node_estimates}

    total_cost = round(sum(e.estimated_cost_usd for e in node_estimates), 6)

    # Critical path latency
    total_latency = 0.0
    for group in resolved.execution_order:
        step_latency = max(est_map[nid].estimated_latency_seconds for nid in group)
        total_latency += step_latency
    total_latency = round(total_latency, 2)

    constraints = resolved.constraints
    within_budget = total_cost <= constraints.max_cost_usd
    within_timeout = total_latency <= constraints.max_latency_seconds

    cost_suggestions = _generate_cost_suggestions(resolved, node_estimates)
    latency_suggestions = _generate_latency_suggestions(resolved, node_estimates)

    summary = _build_user_summary(resolved, node_estimates, total_cost, total_latency)
    min_cost = _min_viable_cost(resolved)

    return TopologyEstimate(
        total_cost_usd=total_cost,
        total_latency_seconds=total_latency,
        node_estimates=node_estimates,
        within_budget=within_budget,
        within_timeout=within_timeout,
        cost_suggestions=cost_suggestions,
        latency_suggestions=latency_suggestions,
        user_facing_summary=summary,
        min_viable_cost_usd=min_cost,
    )
