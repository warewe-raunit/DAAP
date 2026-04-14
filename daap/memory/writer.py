"""
DAAP Memory Writer — post-run writes to Mem0.

Called after runs complete and when users provide ratings/feedback.
Constructs natural language summaries that Mem0 extracts facts from.
"""

from daap.executor.engine import ExecutionResult
from daap.memory.client import DaapMemory


def write_run_to_memory(
    memory: DaapMemory,
    user_id: str,
    topology_summary: str,
    execution_result: ExecutionResult,
    user_rating: int | None = None,
    user_comment: str = "",
) -> None:
    """
    Write a completed run to memory as a natural language summary.

    Mem0 extracts individual facts (cost, latency, node count, outcome)
    for future retrieval when designing similar topologies.
    """
    parts = [
        f"Completed pipeline: {topology_summary}.",
        f"Result: {'successful' if execution_result.success else 'failed'}.",
        f"Latency: {execution_result.total_latency_seconds:.0f} seconds.",
        f"Nodes executed: {len(execution_result.node_results)}.",
    ]

    if execution_result.error:
        parts.append(f"Error: {execution_result.error}.")

    if user_rating is not None:
        parts.append(f"User rated this run {user_rating}/5.")

    if user_comment:
        parts.append(f"User feedback: {user_comment}.")

    memory.store_run_result(
        user_id=user_id,
        run_summary=" ".join(parts),
        run_id=execution_result.topology_id,
    )


def write_user_feedback(
    memory: DaapMemory,
    user_id: str,
    feedback_text: str,
) -> None:
    """Store direct user feedback or corrections."""
    memory.store_user_rating(user_id, feedback_text)


def write_agent_learnings_from_run(
    memory: DaapMemory,
    execution_result: ExecutionResult,
    topology_nodes: list[dict],
) -> None:
    """
    Extract and store per-node learnings from a run.

    Phase 1: records what each node did and how long it took.
    Phase 2: LLM-based extraction of what worked vs. what didn't.
    """
    for node_result in execution_result.node_results:
        # Resolve role name from topology nodes list
        node_role = "unknown"
        for node in topology_nodes:
            if node.get("node_id") == node_result.node_id:
                node_role = node.get("role", "unknown").lower()
                break

        # Map to canonical role for agent diary scoping
        if "research" in node_role:
            agent_role = "researcher"
        elif "evaluat" in node_role or "scor" in node_role or "qualif" in node_role:
            agent_role = "evaluator"
        elif "writ" in node_role or "draft" in node_role or "email" in node_role:
            agent_role = "writer"
        elif "personal" in node_role:
            agent_role = "personalizer"
        else:
            agent_role = "general"

        learning = (
            f"Node '{node_result.node_id}' (role: {node_role}) "
            f"completed in {node_result.latency_seconds:.1f}s. "
            f"Output preview: {node_result.output_text[:200]}"
        )

        memory.store_agent_learning(agent_role, learning)
