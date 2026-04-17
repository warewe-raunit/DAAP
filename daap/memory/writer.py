"""
Memory writer — async, non-blocking, silent-on-failure.

Every write is fire-and-forget. If Mem0 is down, DAAP continues normally.
Logs errors but never raises.
"""

import asyncio
import logging

from daap.memory.config import get_memory_client
from daap.memory.scopes import profile_scope, master_scope, agent_diary_scope
from daap.memory.extractors import (
    extract_profile_from_conversation,
    extract_run_summary,
    extract_agent_observation,
    extract_correction_from_rating,
)

logger = logging.getLogger(__name__)


async def write_profile_async(
    user_id: str,
    user_prompt: str,
    clarifications: list[tuple[str, str]] | None = None,
):
    """
    Extract and store user profile facts from initial conversation.

    Called: after user describes task (first message + ask_user responses).
    Non-blocking. Fails silently.
    """
    try:
        messages = extract_profile_from_conversation(user_prompt, clarifications)
        client = get_memory_client()

        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.add(
                messages=messages,
                **profile_scope(user_id),
                metadata={"source": "initial_conversation"},
            )
        )
    except Exception as e:
        logger.warning(f"write_profile_async failed for user {user_id}: {e}")


async def write_run_summary_async(
    user_id: str,
    topology: dict,
    execution_result: dict,
    user_rating: int | None = None,
):
    """
    Store run summary in master agent memory.

    Called: after every run completes (regardless of rating).
    """
    try:
        summary = extract_run_summary(topology, execution_result, user_rating)
        client = get_memory_client()

        topology_id = topology.get("topology_id", "unknown")

        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.add(
                messages=[{"role": "user", "content": summary}],
                **master_scope(user_id),
                metadata={
                    "source": "run_summary",
                    "topology_id": topology_id,
                    "success": execution_result.get("success", False),
                    "cost_usd": execution_result.get("total_cost_usd", 0),
                    "rating": user_rating,
                },
            )
        )
    except Exception as e:
        logger.warning(f"write_run_summary_async failed for user {user_id}: {e}")


async def write_agent_diary_async(
    user_id: str,
    role: str,
    node_output: str,
    latency_seconds: float,
    model_used: str,
    success: bool,
):
    """
    Store per-node observation in agent diary.

    Called: after each node completes.
    Creates role-specific learning that enriches future node prompts.
    """
    try:
        observation = extract_agent_observation(
            role, node_output, latency_seconds, model_used, success
        )
        client = get_memory_client()

        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.add(
                messages=[{"role": "assistant", "content": observation}],
                **agent_diary_scope(user_id, role),
                metadata={
                    "source": "agent_diary",
                    "model_used": model_used,
                    "success": success,
                },
            )
        )
    except Exception as e:
        logger.warning(f"write_agent_diary_async failed: {e}")


async def write_correction_async(
    user_id: str,
    rating: int,
    comment: str | None = None,
    topology_summary: str | None = None,
):
    """
    Store explicit negative feedback as high-priority correction.

    Called: when user submits rating ≤ 2. No-op for ratings > 2.
    """
    correction = extract_correction_from_rating(rating, comment, topology_summary)
    if correction is None:
        return

    try:
        client = get_memory_client()
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.add(
                messages=[{"role": "user", "content": correction}],
                **profile_scope(user_id),
                metadata={
                    "source": "correction",
                    "rating": rating,
                    "priority": "high",
                },
            )
        )
    except Exception as e:
        logger.warning(f"write_correction_async failed for user {user_id}: {e}")


# ============================================================================
# Client-DaapMemory-based write helpers (used by sessions.py)
# ============================================================================

def write_run_to_memory(
    memory,
    user_id: str,
    topology_summary: str,
    execution_result,
) -> None:
    """
    Store run outcome in memory. No-op if memory unavailable or wrong interface.

    Args:
        memory:           DaapMemory instance (daap.memory.client.DaapMemory)
        user_id:          user identifier
        topology_summary: short human-readable summary of the topology task
        execution_result: object with .success, .topology_id, .total_latency_seconds,
                          .total_input_tokens, .total_output_tokens, .error attributes
    """
    try:
        status = "successful" if execution_result.success else "failed"
        text = (
            f"Run {status}: {topology_summary}. "
            f"Topology: {execution_result.topology_id}. "
            f"Latency: {execution_result.total_latency_seconds:.1f}s. "
            f"Tokens: {execution_result.total_input_tokens} in / "
            f"{execution_result.total_output_tokens} out."
        )
        if not execution_result.success and execution_result.error:
            text += f" Error: {execution_result.error}"
        memory.store_run_result(user_id, text)
    except Exception as e:
        logger.debug("write_run_to_memory skipped: %s", e)


def write_agent_learnings_from_run(
    memory,
    execution_result,
    topology_nodes: list[dict],
) -> None:
    """
    Store per-node learnings after a run. No-op if memory unavailable.

    Args:
        memory:          DaapMemory instance (daap.memory.client.DaapMemory)
        execution_result: object with .node_results (list of NodeResult-like objects)
        topology_nodes:  list of node dicts with "node_id" and "role" keys
    """
    from daap.memory.scopes import _normalize_role

    node_id_to_role: dict[str, str] = {
        n.get("node_id", ""): n.get("role", n.get("node_id", "unknown"))
        for n in topology_nodes
        if isinstance(n, dict)
    }

    for nr in execution_result.node_results:
        try:
            role = node_id_to_role.get(nr.node_id, nr.node_id)
            normalized = _normalize_role(role)
            learning = (
                f"Node '{normalized}' completed in {nr.latency_seconds:.1f}s. "
                f"Output: {nr.output_text[:200]}"
            )
            memory.store_agent_learning(normalized, learning)
        except Exception as e:
            logger.debug("write_agent_learnings_from_run skipped for %s: %s", nr.node_id, e)


# ============================================================================
# Fire-and-forget helpers (for sync callers)
# ============================================================================

def fire_and_forget(coro):
    """
    Schedule a coroutine without awaiting it.

    Use when a sync function needs to trigger an async memory write
    without blocking on its result.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(coro)
        else:
            coro.close()
            logger.debug("No running event loop — memory write skipped")
    except Exception as e:
        logger.warning(f"fire_and_forget failed: {e}")
