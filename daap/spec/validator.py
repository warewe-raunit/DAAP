"""
DAAP Validator — 5-category validation pipeline for topology specs.

Validates: structural integrity, I/O compatibility, resource limits,
tool availability, and pattern correctness.

Returns ALL errors across ALL categories — master agent gets the full
picture for a single retry rather than fixing one error at a time.

Zero external dependencies — no LLM calls, no network.
"""

import re
from collections import deque
from dataclasses import dataclass

from daap.spec.schema import AgentMode, HandoffMode, TopologySpec


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class ValidationError:
    """Single validation failure with enough context to fix it."""
    category: str       # "structural" | "io" | "resource" | "tool" | "pattern"
    node_id: str | None # which node caused it (None = topology-level)
    message: str
    suggestion: str     # injected into master agent retry prompt — be specific


@dataclass
class ValidationResult:
    """Result of the full validation pipeline."""
    is_valid: bool
    errors: list[ValidationError]

    @property
    def error_summary(self) -> str:
        if not self.errors:
            return "Topology is valid."
        lines = []
        for e in self.errors:
            node_str = f" (node: {e.node_id})" if e.node_id else ""
            lines.append(f"[{e.category}]{node_str}: {e.message} → Fix: {e.suggestion}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Category 1: Structural
# ---------------------------------------------------------------------------

def _validate_structural(
    topology: TopologySpec,
    errors: list[ValidationError],
) -> None:
    node_ids = [n.node_id for n in topology.nodes]
    node_id_set = set(node_ids)

    # 1. Node count within limits
    if len(topology.nodes) > topology.constraints.max_nodes:
        errors.append(ValidationError(
            category="structural",
            node_id=None,
            message=f"Topology has {len(topology.nodes)} nodes, exceeds limit of {topology.constraints.max_nodes}",
            suggestion=f"Reduce node count from {len(topology.nodes)} to {topology.constraints.max_nodes} or increase max_nodes constraint.",
        ))

    # 2. Duplicate node IDs (defensive — Pydantic already checks this)
    seen: dict[str, int] = {}
    for nid in node_ids:
        seen[nid] = seen.get(nid, 0) + 1
    for nid, count in seen.items():
        if count > 1:
            errors.append(ValidationError(
                category="structural",
                node_id=nid,
                message=f"Node ID '{nid}' appears {count} times",
                suggestion=f"Each node must have a unique ID. Rename one of the '{nid}' nodes.",
            ))

    # 3. Edge references valid (defensive)
    for edge in topology.edges:
        if edge.source_node_id not in node_id_set:
            errors.append(ValidationError(
                category="structural",
                node_id=None,
                message=f"Edge references source node '{edge.source_node_id}' which does not exist",
                suggestion=f"Available nodes: {node_ids}. Fix source_node_id or add the missing node.",
            ))
        if edge.target_node_id not in node_id_set:
            errors.append(ValidationError(
                category="structural",
                node_id=None,
                message=f"Edge references target node '{edge.target_node_id}' which does not exist",
                suggestion=f"Available nodes: {node_ids}. Fix target_node_id or add the missing node.",
            ))

    # 4. Orphan nodes — nodes with zero incoming AND zero outgoing edges
    has_incoming: set[str] = set()
    has_outgoing: set[str] = set()
    for edge in topology.edges:
        has_outgoing.add(edge.source_node_id)
        has_incoming.add(edge.target_node_id)

    for node in topology.nodes:
        if node.node_id not in has_incoming and node.node_id not in has_outgoing:
            if len(topology.nodes) > 1:  # single-node topologies are fine with no edges
                errors.append(ValidationError(
                    category="structural",
                    node_id=node.node_id,
                    message=f"Node '{node.node_id}' has no incoming or outgoing edges (orphan)",
                    suggestion=f"Connect '{node.node_id}' to the pipeline or remove it.",
                ))

    # 5. DAG check — no cycles (Kahn's algorithm)
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    adjacency: dict[str, list[str]] = {nid: [] for nid in node_ids}
    for edge in topology.edges:
        if edge.source_node_id in adjacency and edge.target_node_id in in_degree:
            adjacency[edge.source_node_id].append(edge.target_node_id)
            in_degree[edge.target_node_id] += 1

    queue: deque[str] = deque(nid for nid in node_ids if in_degree[nid] == 0)
    visited: list[str] = []
    while queue:
        nid = queue.popleft()
        visited.append(nid)
        for neighbor in adjacency[nid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(visited) != len(node_ids):
        cycle_nodes = [nid for nid in node_ids if nid not in set(visited)]
        errors.append(ValidationError(
            category="structural",
            node_id=None,
            message=f"Cycle detected involving nodes: {cycle_nodes}. DAAP requires a DAG.",
            suggestion=f"Break the cycle by removing one of the edges between: {cycle_nodes}.",
        ))

    # 6. Total instance count within limits
    total_instances = sum(n.instance_config.parallel_instances for n in topology.nodes)
    if total_instances > topology.constraints.max_total_instances:
        heavy = sorted(
            topology.nodes,
            key=lambda n: n.instance_config.parallel_instances,
            reverse=True,
        )[:3]
        heavy_names = [f"'{n.node_id}'({n.instance_config.parallel_instances})" for n in heavy]
        errors.append(ValidationError(
            category="structural",
            node_id=None,
            message=f"Total instances ({total_instances}) exceeds limit ({topology.constraints.max_total_instances})",
            suggestion=f"Reduce parallel_instances on nodes with highest counts: {heavy_names}.",
        ))


# ---------------------------------------------------------------------------
# Category 2: I/O Compatibility
# ---------------------------------------------------------------------------

def _validate_io(
    topology: TopologySpec,
    errors: list[ValidationError],
) -> None:
    node_map = {n.node_id: n for n in topology.nodes}

    # Build set of nodes that receive at least one incoming edge per data_key
    satisfied_inputs: dict[str, set[str]] = {n.node_id: set() for n in topology.nodes}

    for edge in topology.edges:
        source = node_map.get(edge.source_node_id)
        target = node_map.get(edge.target_node_id)
        if source is None or target is None:
            continue  # already caught by structural

        source_output_keys = {o.data_key: o for o in source.outputs}
        target_input_keys = {i.data_key: i for i in target.inputs}

        # 1. Source must produce edge.data_key
        if edge.data_key not in source_output_keys:
            errors.append(ValidationError(
                category="io",
                node_id=edge.source_node_id,
                message=(
                    f"Edge from '{edge.source_node_id}' to '{edge.target_node_id}' "
                    f"references data_key '{edge.data_key}' but '{edge.source_node_id}' doesn't produce it"
                ),
                suggestion=f"Available outputs from '{edge.source_node_id}': {list(source_output_keys)}.",
            ))

        # 2. Target must consume edge.data_key
        if edge.data_key not in target_input_keys:
            errors.append(ValidationError(
                category="io",
                node_id=edge.target_node_id,
                message=(
                    f"Edge from '{edge.source_node_id}' to '{edge.target_node_id}' "
                    f"references data_key '{edge.data_key}' but '{edge.target_node_id}' doesn't consume it"
                ),
                suggestion=f"Available inputs on '{edge.target_node_id}': {list(target_input_keys)}.",
            ))

        # 3. Type compatibility (Phase 1: string equality)
        if edge.data_key in source_output_keys and edge.data_key in target_input_keys:
            src_type = source_output_keys[edge.data_key].data_type
            tgt_type = target_input_keys[edge.data_key].data_type
            if src_type != tgt_type:
                errors.append(ValidationError(
                    category="io",
                    node_id=edge.target_node_id,
                    message=(
                        f"Type mismatch on edge '{edge.source_node_id}' → '{edge.target_node_id}': "
                        f"source produces '{src_type}', target expects '{tgt_type}'"
                    ),
                    suggestion=(
                        f"Align data_type on the IOSchema. Either change '{edge.source_node_id}' "
                        f"output to '{tgt_type}' or change '{edge.target_node_id}' input to '{src_type}'."
                    ),
                ))

        # Track satisfied inputs
        if edge.data_key in target_input_keys:
            satisfied_inputs[edge.target_node_id].add(edge.data_key)

    # Determine which nodes have incoming edges (first nodes in DAG have none)
    has_incoming: set[str] = set()
    for edge in topology.edges:
        if edge.target_node_id in node_map:
            has_incoming.add(edge.target_node_id)

    # 4. All declared inputs must be satisfied
    for node in topology.nodes:
        if node.node_id not in has_incoming:
            continue  # first node — receives user prompt directly, no edge inputs needed
        for inp in node.inputs:
            if inp.data_key not in satisfied_inputs[node.node_id]:
                errors.append(ValidationError(
                    category="io",
                    node_id=node.node_id,
                    message=f"Node '{node.node_id}' declares input '{inp.data_key}' but no incoming edge provides it",
                    suggestion=(
                        f"Add an edge from a node that produces '{inp.data_key}' "
                        f"to '{node.node_id}', or remove the input declaration."
                    ),
                ))

    # 5. Consolidation required when parallel > 1 (defensive)
    for node in topology.nodes:
        if node.instance_config.parallel_instances > 1 and node.instance_config.consolidation is None:
            errors.append(ValidationError(
                category="io",
                node_id=node.node_id,
                message=f"Node '{node.node_id}' has {node.instance_config.parallel_instances} parallel instances but no consolidation strategy",
                suggestion="Add consolidation: one of 'merge', 'deduplicate', 'rank', 'vote'.",
            ))


# ---------------------------------------------------------------------------
# Category 3: Resource Validation
# ---------------------------------------------------------------------------

def _validate_resources(
    topology: TopologySpec,
    available_models: set[str],
    errors: list[ValidationError],
) -> None:
    # 1. Instance counts in range (1–10 per node)
    for node in topology.nodes:
        pi = node.instance_config.parallel_instances
        if not (1 <= pi <= 10):
            errors.append(ValidationError(
                category="resource",
                node_id=node.node_id,
                message=f"Node '{node.node_id}' has {pi} parallel instances",
                suggestion="parallel_instances must be between 1 and 10.",
            ))

    # 2. Model tier available
    for node in topology.nodes:
        if node.model_tier.value not in available_models:
            errors.append(ValidationError(
                category="resource",
                node_id=node.node_id,
                message=f"Node '{node.node_id}' requests model tier '{node.model_tier.value}' which is not available",
                suggestion=f"Available model tiers: {sorted(available_models)}.",
            ))

    # 3. max_react_iterations in sane range
    for node in topology.nodes:
        mri = node.max_react_iterations
        if not (1 <= mri <= 25):
            errors.append(ValidationError(
                category="resource",
                node_id=node.node_id,
                message=f"Node '{node.node_id}' has max_react_iterations={mri}",
                suggestion="max_react_iterations must be between 1 and 25.",
            ))

    # 4 & 5. Cost and latency — call resolver + estimator
    try:
        from daap.spec.resolver import resolve_topology
        from daap.spec.estimator import estimate_topology

        resolved = resolve_topology(topology)
        if isinstance(resolved, list):
            for res_err in resolved:
                errors.append(ValidationError(
                    category="resource",
                    node_id=res_err.node_id,
                    message=f"Resolution failed for '{res_err.field}': {res_err.message}",
                    suggestion=f"Fix abstract name '{res_err.abstract_name}'.",
                ))
        else:
            estimate = estimate_topology(resolved)
            if not estimate.within_budget:
                safe = [s for s in estimate.cost_suggestions if s.risk in ("free", "low")]
                sugg = (
                    "\n".join(
                        f"- [{s.risk.upper()}] {s.description} (saves ~${s.estimated_savings_usd:.2f}): {s.reason}"
                        for s in safe
                    )
                    if safe
                    else "No safe cost reductions available. Topology may be at minimum viable cost."
                )
                errors.append(ValidationError(
                    category="resource",
                    node_id=None,
                    message=(
                        f"Estimated cost ${estimate.total_cost_usd:.2f} exceeds budget "
                        f"${topology.constraints.max_cost_usd:.2f}. "
                        f"Minimum viable cost: ${estimate.min_viable_cost_usd:.2f}"
                    ),
                    suggestion=sugg,
                ))
            if not estimate.within_timeout:
                safe = [s for s in estimate.latency_suggestions if s.risk in ("free", "low")]
                sugg = (
                    "\n".join(
                        f"- [{s.risk.upper()}] {s.description} (saves ~{s.estimated_savings_seconds:.0f}s): {s.reason}"
                        for s in safe
                    )
                    if safe
                    else "No safe latency reductions available."
                )
                errors.append(ValidationError(
                    category="resource",
                    node_id=None,
                    message=(
                        f"Estimated latency {estimate.total_latency_seconds:.0f}s "
                        f"exceeds timeout {topology.constraints.max_latency_seconds:.0f}s"
                    ),
                    suggestion=sugg,
                ))
    except Exception as exc:
        errors.append(ValidationError(
            category="resource",
            node_id=None,
            message=f"Cost/latency estimation failed: {exc}",
            suggestion="Check resolver and estimator modules for errors.",
        ))


# ---------------------------------------------------------------------------
# Category 4: Tool Validation
# ---------------------------------------------------------------------------

_MCP_PATTERN = re.compile(r"^mcp://[a-zA-Z0-9_\-]+(?:/[a-zA-Z0-9_.\-]+)?$")


def _validate_tools(
    topology: TopologySpec,
    available_tools: set[str],
    errors: list[ValidationError],
) -> None:
    for node in topology.nodes:
        seen_tools: dict[str, int] = {}
        for binding in node.tools:
            name = binding.name
            seen_tools[name] = seen_tools.get(name, 0) + 1

            # 1. MCP format check
            if name.startswith("mcp://"):
                if not _MCP_PATTERN.match(name):
                    errors.append(ValidationError(
                        category="tool",
                        node_id=node.node_id,
                        message=f"Malformed MCP tool reference '{name}'",
                        suggestion=(
                            "Expected format: mcp://service_name or "
                            "mcp://service_name/tool_name "
                            "(alphanumeric, hyphens, underscores, dots in tool_name)."
                        ),
                    ))
                    continue

            # 2. Tool available
            if name not in available_tools:
                errors.append(ValidationError(
                    category="tool",
                    node_id=node.node_id,
                    message=f"Node '{node.node_id}' references tool '{name}' which is not available",
                    suggestion=f"Available tools: {sorted(available_tools)}.",
                ))

        # 3. Duplicate tools
        for name, count in seen_tools.items():
            if count > 1:
                errors.append(ValidationError(
                    category="tool",
                    node_id=node.node_id,
                    message=f"Node '{node.node_id}' lists tool '{name}' {count} times",
                    suggestion=f"Remove duplicate '{name}' entries from node '{node.node_id}' tools list.",
                ))

        # 4. ReAct nodes must have tools
        if node.agent_mode == AgentMode.REACT and not node.tools:
            errors.append(ValidationError(
                category="tool",
                node_id=node.node_id,
                message=f"Node '{node.node_id}' is set to ReAct mode but has no tools",
                suggestion="Either add tools or switch to agent_mode='single'.",
            ))


# ---------------------------------------------------------------------------
# Category 5: Pattern Validation
# ---------------------------------------------------------------------------

def _validate_patterns(
    topology: TopologySpec,
    errors: list[ValidationError],
) -> None:
    node_map = {n.node_id: n for n in topology.nodes}

    # Build outgoing edges per node
    outgoing: dict[str, list[str]] = {n.node_id: [] for n in topology.nodes}
    for edge in topology.edges:
        if edge.source_node_id in outgoing:
            outgoing[edge.source_node_id].append(edge.target_node_id)

    for node in topology.nodes:
        # 1. Parallel fan-out with no consolidation
        if node.instance_config.parallel_instances > 1:
            if node.instance_config.consolidation is None:
                targets = outgoing[node.node_id]
                if targets:
                    errors.append(ValidationError(
                        category="pattern",
                        node_id=node.node_id,
                        message=(
                            f"Node '{node.node_id}' fans out to {node.instance_config.parallel_instances} "
                            f"instances but has no consolidation strategy"
                        ),
                        suggestion=(
                            f"Downstream node(s) {targets} would receive "
                            f"{node.instance_config.parallel_instances} separate outputs. "
                            f"Add consolidation: 'merge', 'deduplicate', 'rank', or 'vote'."
                        ),
                    ))

        # 2. Coordinator (ALWAYS) with outgoing edges
        if node.handoff_mode == HandoffMode.ALWAYS:
            targets = outgoing[node.node_id]
            if targets:
                errors.append(ValidationError(
                    category="pattern",
                    node_id=node.node_id,
                    message=(
                        f"Node '{node.node_id}' has handoff_mode=ALWAYS but has "
                        f"{len(targets)} outgoing edge(s)"
                    ),
                    suggestion=(
                        f"In ALWAYS mode the node transfers full control and exits — "
                        f"downstream edges {targets} may never execute. "
                        f"Remove those edges or change handoff_mode to 'never' or 'optional'."
                    ),
                ))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DEFAULT_AVAILABLE_TOOLS: set[str] = {
    "WebSearch", "WebFetch", "ReadFile", "WriteFile", "CodeExecution",
}
DEFAULT_AVAILABLE_MODELS: set[str] = {"fast", "smart", "powerful"}


def validate_topology(
    topology: TopologySpec,
    available_tools: set[str] = DEFAULT_AVAILABLE_TOOLS,
    available_models: set[str] = DEFAULT_AVAILABLE_MODELS,
) -> ValidationResult:
    """
    Run all 5 validation categories in order.

    Runs ALL categories even if earlier ones fail — master agent gets the
    complete error picture, reducing retry count.

    Args:
        topology:         parsed TopologySpec
        available_tools:  tool names currently available (including mcp:// tools)
        available_models: model tier names currently available

    Returns:
        ValidationResult with is_valid=True if all checks pass.
    """
    errors: list[ValidationError] = []

    _validate_structural(topology, errors)
    _validate_io(topology, errors)
    _validate_resources(topology, available_models, errors)
    _validate_tools(topology, available_tools, errors)
    _validate_patterns(topology, errors)

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
