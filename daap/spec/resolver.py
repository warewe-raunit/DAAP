"""
DAAP Resolver — maps abstract spec references to concrete runtime identifiers.

Translates ModelTier enums → concrete model IDs, abstract tool names → framework
tool identifiers, and computes execution_order (parallel groups via topological sort).

Supports any LLM operator: checks node.operator_override, then topology.operator_config,
then falls back to the default MODEL_REGISTRY.
"""

from collections import deque
from dataclasses import dataclass, field

from daap.spec.schema import TopologySpec, NodeSpec


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

# Model tier → concrete OpenRouter model ID
# Updated: April 2026
# Gemini 2.0 Flash deprecated (shutdown June 1 2026) — migrated to newer models
MODEL_REGISTRY: dict[str, str] = {
    "fast":     "google/gemini-2.5-flash-lite",   # search, extract, format — cheapest viable
    "smart":    "deepseek/deepseek-v3.2",          # evaluate, score, write — GPT-5 class @ 1/50th cost
    "powerful": "google/gemini-2.5-flash",         # master agent, complex planning — thinking mode
}

# Pricing per 1M tokens (USD) — used by estimator.py
# Source: OpenRouter posted rates, April 2026
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Active tier models
    "google/gemini-2.5-flash-lite": {"input_per_1m": 0.10,  "output_per_1m": 0.40},
    "deepseek/deepseek-v3.2":       {"input_per_1m": 0.26,  "output_per_1m": 0.38},
    "google/gemini-2.5-flash":      {"input_per_1m": 0.30,  "output_per_1m": 2.50},
    # Upgrade options (operator_config overrides / future use)
    "google/gemini-3-flash-preview": {"input_per_1m": 0.50, "output_per_1m": 3.00},
    # Legacy — backward compat with saved topologies
    "google/gemini-2.0-flash-001":  {"input_per_1m": 0.10,  "output_per_1m": 0.40},
}

# Conservative fallback for unknown models
DEFAULT_PRICING: dict[str, float] = {"input_per_1m": 1.00, "output_per_1m": 5.00}


def get_model_pricing(model_id: str) -> dict[str, float]:
    """Return pricing for a model ID. Falls back to DEFAULT_PRICING if unknown."""
    return MODEL_PRICING.get(model_id, DEFAULT_PRICING)

# Phase 1: tool names resolve to string identifiers.
# Phase 3: becomes a registry of actual tool instances.
TOOL_REGISTRY: dict[str, str] = {
    "WebSearch":               "agentscope.tools.WebSearch",
    "WebFetch":                "agentscope.tools.WebFetch",
    "DeepCrawl":               "agentscope.tools.DeepCrawl",
    "RedditSearch":            "agentscope.tools.RedditSearch",
    "RedditFetch":             "agentscope.tools.RedditFetch",
    "BatchRedditFetch":        "agentscope.tools.BatchRedditFetch",
    "KeywordsEverywhere":      "agentscope.tools.KeywordsEverywhere",
    "KeywordsEverywhereTraffic": "agentscope.tools.KeywordsEverywhereTraffic",
    "ReadFile":                "agentscope.tools.ReadFile",
    "WriteFile":               "agentscope.tools.WriteFile",
    "CodeExecution":           "agentscope.tools.CodeExecution",
}

CONSOLIDATION_REGISTRY: dict[str, str] = {
    "merge": "daap.executor.consolidation.merge_outputs",
    "deduplicate": "daap.executor.consolidation.deduplicate_outputs",
    "rank": "daap.executor.consolidation.rank_outputs",
    "vote": "daap.executor.consolidation.vote_outputs",
}


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class ResolvedNode:
    """A NodeSpec with all abstract names replaced by concrete identifiers."""
    node_id: str
    role: str
    concrete_model_id: str           # e.g. "claude-haiku-4-5-20251001"
    system_prompt: str
    concrete_tool_ids: list[str]     # e.g. ["agentscope.tools.WebSearch", "mcp://linkedin"]
    inputs: list                     # IOSchema objects (unchanged)
    outputs: list                    # IOSchema objects (unchanged)
    parallel_instances: int
    consolidation_func: str | None   # e.g. "daap.executor.consolidation.merge_outputs"
    consolidation_strategy: str | None = None  # e.g. "merge", "deduplicate", "rank", "vote"
    handoff_mode: str = "never"      # enum value as string
    agent_mode: str = "react"        # "react" or "single"
    max_react_iterations: int = 10
    # Operator info — carried through for node_builder to select the right model class
    operator_provider: str = "openrouter"        # "openrouter" | "opencode" | ...
    operator_base_url: str | None = None          # None = provider SDK default
    operator_api_key_env: str = "OPENROUTER_API_KEY"  # env var holding the API key


@dataclass
class ResolvedTopology:
    """Fully resolved topology ready for the execution engine."""
    topology_id: str
    version: int
    created_at: str
    user_prompt: str
    nodes: list[ResolvedNode]
    edges: list                      # EdgeSpec objects (unchanged)
    constraints: object              # ConstraintSpec (unchanged)
    metadata: dict
    execution_order: list[list[str]] # parallel groups, e.g. [["a"], ["b", "c"], ["d"]]


@dataclass
class ResolutionError:
    """A single resolution failure."""
    node_id: str
    field: str           # which field failed
    abstract_name: str   # the name that couldn't be resolved
    message: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_model(node: NodeSpec, topology: TopologySpec) -> str | None:
    """Resolve model tier → concrete model ID. Operator override chain."""
    tier = node.model_tier.value
    # 1. node-level operator override
    if node.operator_override and node.operator_override.model_map:
        concrete = node.operator_override.model_map.get(tier)
        if concrete:
            return concrete
    # 2. topology-level operator config
    if topology.operator_config and topology.operator_config.model_map:
        concrete = topology.operator_config.model_map.get(tier)
        if concrete:
            return concrete
    # 3. default registry
    return MODEL_REGISTRY.get(tier)


def _resolve_tool(tool_name: str) -> str | None:
    """
    Resolve abstract tool name → concrete identifier.
    MCP tools (mcp://<service>) pass through as-is.
    """
    if tool_name.startswith("mcp://"):
        return tool_name  # pass-through; executor establishes MCP connection
    return TOOL_REGISTRY.get(tool_name)


def _compute_execution_order(topology: TopologySpec) -> list[list[str]]:
    """
    Kahn's algorithm — returns parallel groups (BFS topological sort).

    Each inner list = nodes that can run simultaneously at that step.
    Assumes the topology is a DAG (validator enforces this).
    """
    node_ids = [n.node_id for n in topology.nodes]
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    adjacency: dict[str, list[str]] = {nid: [] for nid in node_ids}

    for edge in topology.edges:
        adjacency[edge.source_node_id].append(edge.target_node_id)
        in_degree[edge.target_node_id] += 1

    queue: deque[str] = deque(nid for nid in node_ids if in_degree[nid] == 0)
    order: list[list[str]] = []

    while queue:
        # All nodes currently in queue can run in parallel
        group = list(queue)
        queue.clear()
        order.append(group)
        for nid in group:
            for neighbor in adjacency[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

    return order


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_topology(
    topology: TopologySpec,
) -> ResolvedTopology | list[ResolutionError]:
    """
    Resolve all abstract names in the topology to concrete identifiers.

    Also computes execution_order via parallel-group topological sort.

    Returns ResolvedTopology on success, or list[ResolutionError] on failure.
    """
    errors: list[ResolutionError] = []
    resolved_nodes: list[ResolvedNode] = []

    for node in topology.nodes:
        # --- Model resolution ---
        concrete_model = _resolve_model(node, topology)
        if concrete_model is None:
            errors.append(ResolutionError(
                node_id=node.node_id,
                field="model_tier",
                abstract_name=node.model_tier.value,
                message=(
                    f"Node '{node.node_id}': model tier '{node.model_tier.value}' not found "
                    f"in registry or operator config. Available tiers: {list(MODEL_REGISTRY)}"
                ),
            ))
            concrete_model = ""  # placeholder so we can continue

        # --- Tool resolution ---
        concrete_tools: list[str] = []
        for binding in node.tools:
            resolved = _resolve_tool(binding.name)
            if resolved is None:
                errors.append(ResolutionError(
                    node_id=node.node_id,
                    field="tools",
                    abstract_name=binding.name,
                    message=(
                        f"Node '{node.node_id}': tool '{binding.name}' not found in registry. "
                        f"Available tools: {list(TOOL_REGISTRY)}. "
                        f"MCP tools must use format mcp://service_name or mcp://service_name/tool_name."
                    ),
                ))
            else:
                concrete_tools.append(resolved)

        # --- Consolidation resolution ---
        consolidation_func: str | None = None
        if node.instance_config.consolidation is not None:
            key = node.instance_config.consolidation.value
            consolidation_func = CONSOLIDATION_REGISTRY.get(key)
            if consolidation_func is None:
                errors.append(ResolutionError(
                    node_id=node.node_id,
                    field="consolidation",
                    abstract_name=key,
                    message=(
                        f"Node '{node.node_id}': consolidation strategy '{key}' not found. "
                        f"Available: {list(CONSOLIDATION_REGISTRY)}"
                    ),
                ))

        # Determine operator for this node (override → topology → defaults)
        if node.operator_override:
            op = node.operator_override
        elif topology.operator_config:
            op = topology.operator_config
        else:
            op = None

        consolidation_strategy: str | None = (
            node.instance_config.consolidation.value
            if node.instance_config.consolidation is not None
            else None
        )

        resolved_nodes.append(ResolvedNode(
            node_id=node.node_id,
            role=node.role,
            concrete_model_id=concrete_model,
            system_prompt=node.system_prompt,
            concrete_tool_ids=concrete_tools,
            inputs=node.inputs,
            outputs=node.outputs,
            parallel_instances=node.instance_config.parallel_instances,
            consolidation_func=consolidation_func,
            consolidation_strategy=consolidation_strategy,
            handoff_mode=node.handoff_mode.value,
            agent_mode=node.agent_mode.value,
            max_react_iterations=node.max_react_iterations,
            operator_provider=op.provider if op else "openrouter",
            operator_base_url=op.base_url if op else None,
            operator_api_key_env=op.api_key_env if op else "OPENROUTER_API_KEY",
        ))

    if errors:
        return errors

    execution_order = _compute_execution_order(topology)

    return ResolvedTopology(
        topology_id=topology.topology_id,
        version=topology.version,
        created_at=topology.created_at,
        user_prompt=topology.user_prompt,
        nodes=resolved_nodes,
        edges=topology.edges,
        constraints=topology.constraints,
        metadata=topology.metadata,
        execution_order=execution_order,
    )
