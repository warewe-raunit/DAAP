"""
DAAP Memory Reader — pre-run reads from Mem0.

Called before the master agent runs (to inject user context into the
system prompt) and before each node executes (to enrich system prompts
with agent diary learnings).
"""

from daap.memory.client import DaapMemory


def load_user_context_for_master(
    memory: DaapMemory,
    user_id: str,
    user_prompt: str = "",
) -> dict | None:
    """
    Load user context from memory for the master agent system prompt.

    Returns a dict with keys: profile, preferences, recent_runs.
    Returns None if the user has no memories (first-time user).
    """
    context = memory.get_user_context(user_id, query=user_prompt or None)

    has_content = any([
        context.get("profile"),
        context.get("preferences"),
        context.get("recent_runs"),
    ])

    return context if has_content else None


def load_agent_context_for_node(
    memory: DaapMemory,
    agent_role: str,
    task_description: str,
) -> str:
    """
    Load agent diary learnings for a specific node role.

    Returns a markdown string to prepend to the node's system prompt.
    Returns empty string if no learnings exist — caller uses as-is.
    """
    learnings = memory.get_agent_learnings(agent_role, task_description)

    if not learnings:
        return ""

    lines = ["## Learnings from past runs (use these to improve your work):"]
    for learning in learnings:
        lines.append(f"- {learning}")

    return "\n".join(lines)


def format_user_context_for_prompt(context: dict) -> str:
    """
    Format a user context dict into a readable string for system prompts.

    Returns empty string if context is None or empty.
    """
    if not context:
        return ""

    sections = []

    if context.get("profile"):
        sections.append("**User Profile:**")
        for fact in context["profile"]:
            if fact:
                sections.append(f"  - {fact}")

    if context.get("preferences"):
        sections.append("**User Preferences:**")
        for pref in context["preferences"]:
            if pref:
                sections.append(f"  - {pref}")

    if context.get("recent_runs"):
        sections.append("**Recent Run History:**")
        for run in context["recent_runs"]:
            if run:
                sections.append(f"  - {run}")

    return "\n".join(sections)
