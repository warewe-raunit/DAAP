"""
Memory reader — blocking, but fast.

Reads are synchronous because the caller needs results before continuing.
Mem0's selective pipeline has ~1.5s p95 latency — acceptable for blocking reads.

All reads are defensive: return empty on failure, never crash.
"""

import logging

from daap.memory.config import get_memory_client, check_memory_available
from daap.memory.observability import record_memory_error, record_memory_event
from daap.memory.scopes import profile_scope, master_scope, agent_diary_scope

logger = logging.getLogger(__name__)


def load_user_profile(user_id: str, limit: int = 10) -> list[str]:
    """
    Load user profile facts (ICP, product, preferences).

    Called: at master agent start, to inform topology generation.
    Returns list of memory strings. Empty list on failure or no data.
    """
    try:
        client = get_memory_client()
        result = client.get_all(
            filters=profile_scope(user_id),
            limit=limit,
        )
        record_memory_event("load_user_profile", True)
        return [m["memory"] for m in result.get("results", [])]
    except Exception as e:
        logger.warning(f"load_user_profile failed for user {user_id}: {e}")
        record_memory_error("load_user_profile", e)
        record_memory_event("load_user_profile", False)
        return []


def search_user_profile(user_id: str, query: str, limit: int = 5) -> list[str]:
    """
    Semantic search in user profile.

    Use when you want specific info rather than all.
    """
    try:
        client = get_memory_client()
        result = client.search(
            query=query,
            filters=profile_scope(user_id),
            limit=limit,
        )
        record_memory_event("search_user_profile", True)
        return [m["memory"] for m in result.get("results", [])]
    except Exception as e:
        logger.warning(f"search_user_profile failed: {e}")
        record_memory_error("search_user_profile", e)
        record_memory_event("search_user_profile", False)
        return []


def load_master_history(user_id: str, query: str, limit: int = 5) -> list[str]:
    """
    Search master agent memory for similar past runs.

    Called: at master agent start with user's new prompt as query.
    Returns semantically similar past run summaries.
    """
    if not query or not query.strip():
        query = "past run topology result"

    try:
        client = get_memory_client()
        result = client.search(
            query=query,
            filters=master_scope(user_id),
            limit=limit,
        )
        record_memory_event("load_master_history", True)
        return [m["memory"] for m in result.get("results", [])]
    except Exception as e:
        logger.warning(f"load_master_history failed: {e}")
        record_memory_error("load_master_history", e)
        record_memory_event("load_master_history", False)
        return []


def load_agent_diary(
    user_id: str,
    role: str,
    query: str | None = None,
    limit: int = 5,
) -> list[str]:
    """
    Load relevant diary entries for a specific node role.

    Called: per-node, before execution, to enrich system prompt.

    Args:
        user_id: whose diary
        role: node role (e.g., "researcher")
        query: optional semantic search query (uses node task as query)
        limit: max entries to return
    """
    try:
        client = get_memory_client()
        scope = agent_diary_scope(user_id, role)

        if query:
            result = client.search(query=query, limit=limit, **scope)
        else:
            result = client.get_all(limit=limit, **scope)

        record_memory_event("load_agent_diary", True)
        return [m["memory"] for m in result.get("results", [])]
    except Exception as e:
        logger.warning(f"load_agent_diary failed: {e}")
        record_memory_error("load_agent_diary", e)
        record_memory_event("load_agent_diary", False)
        return []


def memory_is_available() -> bool:
    """Quick check: can we use memory right now?"""
    ok, _ = check_memory_available()
    return ok


# ============================================================================
# Client-DaapMemory-based helpers (used by node_builder and sessions)
# ============================================================================

def load_agent_context_for_node(memory, role: str, task: str) -> str:
    """
    Load and format agent learnings for a node's system prompt.

    Args:
        memory: DaapMemory instance (daap.memory.client.DaapMemory)
        role:   node role (e.g., "researcher", "writer")
        task:   short task description for semantic search

    Returns:
        Formatted string with learnings, or "" if none found.
    """
    try:
        learnings = memory.get_agent_learnings(role, task)
    except Exception:
        return ""

    learnings = [l for l in learnings if l]
    if not learnings:
        return ""

    lines = ["Learnings from past runs:"]
    for l in learnings:
        lines.append(f"- {l}")
    return "\n".join(lines)


def load_user_context_for_master(memory, user_id: str, query: str) -> dict | None:
    """
    Load user context for the master agent system prompt.

    Args:
        memory:  DaapMemory instance (daap.memory.client.DaapMemory)
        user_id: user identifier
        query:   current user prompt (for semantic search)

    Returns:
        Dict with keys profile/preferences/recent_runs, or None if no memories.
    """
    try:
        ctx = memory.get_user_context(user_id, query)
    except Exception:
        return None

    if not any(ctx.values()):
        return None
    return ctx


def format_user_context_for_prompt(ctx: dict | None) -> str:
    """
    Format user context dict into prompt-ready text.

    Returns "" for None or empty dict.
    """
    if not ctx:
        return ""

    sections = []
    if ctx.get("profile"):
        lines = ["## User Profile"]
        lines.extend(f"- {m}" for m in ctx["profile"] if m)
        sections.append("\n".join(lines))
    if ctx.get("preferences"):
        lines = ["## User Preferences"]
        lines.extend(f"- {m}" for m in ctx["preferences"] if m)
        sections.append("\n".join(lines))
    if ctx.get("recent_runs"):
        lines = ["## Recent Run History"]
        lines.extend(f"- {m}" for m in ctx["recent_runs"] if m)
        sections.append("\n".join(lines))

    return "\n\n".join(sections) if sections else ""


# ============================================================================
# Formatting helpers — convert memory lists to prompt-ready text
# ============================================================================

def format_profile_for_prompt(profile_memories: list[str]) -> str:
    """Format profile memories for injection into master agent system prompt."""
    if not profile_memories:
        return ""

    lines = ["\n## What I Know About This User"]
    for m in profile_memories:
        lines.append(f"- {m}")
    return "\n".join(lines) + "\n"


def format_history_for_prompt(history_memories: list[str]) -> str:
    """Format past run history for master agent prompt."""
    if not history_memories:
        return ""

    lines = ["\n## Relevant Past Runs"]
    for m in history_memories:
        lines.append(f"- {m}")
    return "\n".join(lines) + "\n"


def format_diary_for_prompt(diary_memories: list[str], role: str) -> str:
    """Format agent diary for injection into node system prompt."""
    if not diary_memories:
        return ""

    lines = [f"\n## Lessons From Past {role.title()} Runs"]
    for m in diary_memories:
        lines.append(f"- {m}")
    return "\n".join(lines) + "\n"
