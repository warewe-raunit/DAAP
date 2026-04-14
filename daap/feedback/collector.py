"""
DAAP Feedback Collector — captures execution metrics after a topology run.

Called by the WebSocket handler after execute_topology() completes.
Stores everything Phase 2 needs for RL optimization.
"""

from daap.executor.engine import ExecutionResult
from daap.feedback.store import FeedbackStore


def collect_run_feedback(
    feedback_store: FeedbackStore,
    session_id: str,
    topology_dict: dict | None,
    execution_result: ExecutionResult,
) -> None:
    """
    Capture metrics from a completed execution run and persist to SQLite.

    Args:
        feedback_store:   the FeedbackStore instance to write to
        session_id:       session that triggered this run
        topology_dict:    raw TopologySpec dict (for replay / analysis)
        execution_result: ExecutionResult from execute_topology()
    """
    result_dict = {
        "topology_id": execution_result.topology_id,
        "final_output": execution_result.final_output[:1000],  # truncate for storage
        "success": execution_result.success,
        "error": execution_result.error,
        "latency_seconds": execution_result.total_latency_seconds,
        "node_results": [
            {
                "node_id": nr.node_id,
                "latency_seconds": nr.latency_seconds,
                "output_preview": nr.output_text[:200],
            }
            for nr in execution_result.node_results
        ],
    }

    feedback_store.store_run(
        session_id=session_id,
        topology_json=topology_dict,
        execution_result=result_dict,
    )
