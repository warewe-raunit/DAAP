"""
DAAP Execution Engine — top-level orchestrator for topology execution.

Takes a ResolvedTopology, builds agents via node_builder, walks the
execution_order DAG step-by-step, and returns a structured ExecutionResult.

Design constraint: this file does NOT import from agentscope directly.
It works with BuiltNode objects. Only node_builder.py knows about AgentScope.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

from agentscope.message import Msg

from daap.executor.node_builder import build_node, BuiltNode
from daap.executor.patterns import run_execution_step
from daap.spec.resolver import ResolvedTopology, get_model_pricing
from daap.tools.registry import get_tool_registry
from daap.tools.token_tracker import TokenTracker

logger = logging.getLogger(__name__)


def _is_rate_limit_error(exc: Exception) -> bool:
    """
    Detect 429 / rate-limit errors from OpenAI-compatible clients.
    Checks typed exceptions first (openai SDK), falls back to string matching.
    """
    try:
        import openai
        if isinstance(exc, openai.RateLimitError):
            return True
        if isinstance(exc, openai.APIStatusError) and exc.status_code == 429:
            return True
    except ImportError:
        pass
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "ratelimit" in msg


def _is_server_error(exc: Exception) -> bool:
    """Detect 5xx server errors from OpenAI-compatible clients."""
    try:
        import openai
        if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
            return True
    except ImportError:
        pass
    msg = str(exc).lower()
    return any(code in msg for code in ("500", "502", "503", "504"))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class NodeResult:
    """Execution result for a single node."""
    node_id: str
    output_text: str
    latency_seconds: float
    model_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ExecutionResult:
    """Complete execution result for a topology run."""
    topology_id: str
    final_output: str
    node_results: list[NodeResult]
    total_latency_seconds: float
    success: bool
    error: str | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    models_used: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

async def execute_topology(
    resolved: ResolvedTopology,
    user_prompt: str,
    tracker: TokenTracker | None = None,
    on_node_start: object | None = None,     # callable(node_id, model_id, step_num, total_steps)
    on_node_complete: object | None = None,  # callable(NodeResult)
    daap_memory=None,                        # optional DaapMemory for node prompt enrichment
    user_id: str | None = None,              # optional user_id for per-user memory
) -> ExecutionResult:
    """
    Execute a fully resolved topology end-to-end.

    Steps:
    1. Load tool registry
    2. Build all nodes (ResolvedNode → BuiltNode)
    3. Walk execution_order step by step
    4. For each step: run all nodes concurrently, collect outputs
    5. Return final output from the last node + per-node results

    Error handling: any node failure aborts the pipeline (Phase 1).
    Phase 3 adds partial results, fallback nodes, and smarter retry.
    """
    run_start = time.time()
    node_results: list[NodeResult] = []

    try:
        tool_registry = get_tool_registry()

        # Per-execution tracker — creates fresh one if caller didn't supply
        exec_tracker = tracker or TokenTracker()

        # 1. Build all nodes (optionally enriches system prompts via agent diary)
        built_nodes: dict[str, BuiltNode] = {}
        for rnode in resolved.nodes:
            built = await build_node(rnode, tool_registry, daap_memory=daap_memory, tracker=exec_tracker, user_id=user_id)
            built_nodes[rnode.node_id] = built

        # 2. Map node_id → output data_key (first declared output)
        node_output_key: dict[str, str] = {
            n.node_id: n.outputs[0].data_key
            for n in resolved.nodes
            if n.outputs
        }

        # 3. Initialize data store
        initial_msg = Msg(name="user", content=user_prompt, role="user")
        data_store: dict[str, Msg] = {}  # data_key → Msg

        total_steps = len(resolved.execution_order)

        # 4. Walk execution order
        consecutive_errors: int = 0
        circuit_breaker_threshold: int = 2

        for step_num, step in enumerate(resolved.execution_order, 1):
            step_nodes = [built_nodes[nid] for nid in step]
            max_retries = resolved.constraints.max_retries_per_node

            # --- Global hard cost cap (pre-step check) ---
            current_spend = exec_tracker.total_cost_usd(get_model_pricing)
            budget = resolved.constraints.max_cost_usd
            if current_spend >= budget:
                return ExecutionResult(
                    topology_id=resolved.topology_id,
                    final_output="",
                    node_results=node_results,
                    total_latency_seconds=round(time.time() - run_start, 3),
                    success=False,
                    error=(
                        f"Cost limit exceeded before step {step_num}: "
                        f"spent ${current_spend:.4f} of ${budget:.4f} budget. "
                        f"Increase max_cost_usd or simplify the topology."
                    ),
                    total_input_tokens=exec_tracker.total_input,
                    total_output_tokens=exec_tracker.total_output,
                )

            # Fire on_node_start for each node in step
            if on_node_start:
                for nid in step:
                    rnode = next(n for n in resolved.nodes if n.node_id == nid)
                    model_id = rnode.concrete_model_id
                    if "/" not in model_id:
                        model_id = f"anthropic/{model_id}"
                    on_node_start(nid, model_id, step_num, total_steps)

            step_start = time.time()
            # Snapshot tokens BEFORE step runs so we can compute per-step delta
            input_before_step = exec_tracker.total_input
            output_before_step = exec_tracker.total_output

            # Retry wrapper per step with circuit breaker
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    step_raw = await run_execution_step(
                        step_nodes, data_store, resolved.edges, initial_msg
                    )
                    last_exc = None
                    consecutive_errors = 0  # reset on success
                    break
                except Exception as exc:
                    last_exc = exc
                    if _is_rate_limit_error(exc) or _is_server_error(exc):
                        consecutive_errors += 1
                        logger.warning(
                            "Retryable API error (consecutive=%d): %s",
                            consecutive_errors, exc,
                        )
                        if consecutive_errors >= circuit_breaker_threshold:
                            return ExecutionResult(
                                topology_id=resolved.topology_id,
                                final_output="",
                                node_results=node_results,
                                total_latency_seconds=round(time.time() - run_start, 3),
                                success=False,
                                error=(
                                    f"Circuit breaker open: {consecutive_errors} consecutive "
                                    f"rate-limit/server errors. API quota likely exceeded. "
                                    f"Last error: {exc}"
                                ),
                                total_input_tokens=exec_tracker.total_input,
                                total_output_tokens=exec_tracker.total_output,
                            )
                    if attempt < max_retries:
                        await asyncio.sleep(1)

            if last_exc is not None:
                failed_node = step[0]  # report first node in step
                return ExecutionResult(
                    topology_id=resolved.topology_id,
                    final_output="",
                    node_results=node_results,
                    total_latency_seconds=time.time() - run_start,
                    success=False,
                    error=f"Node '{failed_node}' failed after {max_retries} retries: {last_exc}",
                    total_input_tokens=exec_tracker.total_input,
                    total_output_tokens=exec_tracker.total_output,
                )

            step_latency = time.time() - step_start

            # Distribute step token delta equally across nodes that ran.
            # For single-node steps this is exact; for parallel steps it's an approximation.
            nodes_ran = [nid for nid in step if nid in step_raw]
            n_ran = max(len(nodes_ran), 1)
            step_input_delta = (exec_tracker.total_input - input_before_step) // n_ran
            step_output_delta = (exec_tracker.total_output - output_before_step) // n_ran

            # Per-node token budget warning
            node_token_budget = resolved.constraints.max_tokens_per_node
            node_tokens_used = step_input_delta + step_output_delta
            if node_tokens_used > node_token_budget:
                logger.warning(
                    "Step %d exceeded per-node token budget: used %d tokens (budget: %d). "
                    "Consider reducing max_react_iterations or parallel_instances.",
                    step_num, node_tokens_used, node_token_budget,
                )

            # Remap node_id keys → data_key keys in data_store
            for nid in step:
                if nid in step_raw:
                    output_key = node_output_key.get(nid, nid)
                    data_store[output_key] = step_raw[nid]
                    rnode = next(n for n in resolved.nodes if n.node_id == nid)
                    model_id = rnode.concrete_model_id
                    if "/" not in model_id:
                        model_id = f"anthropic/{model_id}"
                    nr = NodeResult(
                        node_id=nid,
                        output_text=_truncate(step_raw[nid]),
                        latency_seconds=round(step_latency, 3),
                        model_id=model_id,
                        input_tokens=step_input_delta,
                        output_tokens=step_output_delta,
                    )
                    node_results.append(nr)
                    if on_node_complete:
                        on_node_complete(nr)

        # 5. Final output = last node's output
        last_nid = resolved.execution_order[-1][-1]
        final_key = node_output_key.get(last_nid, last_nid)
        final_msg = data_store.get(final_key)
        final_text = (
            final_msg.content if final_msg and isinstance(final_msg.content, str)
            else str(final_msg.content) if final_msg
            else "No output produced."
        )

        return ExecutionResult(
            topology_id=resolved.topology_id,
            final_output=final_text,
            node_results=node_results,
            total_latency_seconds=round(time.time() - run_start, 3),
            success=True,
            total_input_tokens=exec_tracker.total_input,
            total_output_tokens=exec_tracker.total_output,
            models_used=exec_tracker.models_used,
        )

    except Exception as exc:
        return ExecutionResult(
            topology_id=resolved.topology_id,
            final_output="",
            node_results=node_results,
            total_latency_seconds=round(time.time() - run_start, 3),
            success=False,
            error=str(exc),
        )


def _truncate(msg: Msg, limit: int = 500) -> str:
    text = msg.content if isinstance(msg.content, str) else str(msg.content)
    return text[:limit] + ("..." if len(text) > limit else "")
