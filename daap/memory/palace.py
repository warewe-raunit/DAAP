"""
DaapMemory — high-level memory API.

Single entry point for all memory operations. Wraps reader/writer
so callers don't need to know about Mem0 internals or scopes.
"""

import logging
from daap.memory import reader, writer

logger = logging.getLogger(__name__)


class DaapMemory:
    """
    Facade over Mem0 for DAAP-specific operations.

    Usage:
        mem = DaapMemory()
        profile = mem.get_user_profile("alice")
        mem.remember_run("alice", topology, result)  # non-blocking
    """

    def __init__(self):
        self.available = reader.memory_is_available()
        if not self.available:
            logger.warning("Memory unavailable — operating in degraded mode")

    # ========================================================================
    # Reads (blocking, fast)
    # ========================================================================

    def get_user_profile(self, user_id: str) -> list[str]:
        """Get user's profile facts (ICP, product, preferences)."""
        if not self.available:
            return []
        return reader.load_user_profile(user_id)

    def get_past_runs(self, user_id: str, query: str) -> list[str]:
        """Get past runs semantically similar to query."""
        if not self.available:
            return []
        return reader.load_master_history(user_id, query)

    def get_agent_wisdom(self, user_id: str, role: str, task: str | None = None) -> list[str]:
        """Get diary entries for a specific role."""
        if not self.available:
            return []
        return reader.load_agent_diary(user_id, role, query=task)

    # ========================================================================
    # Writes (non-blocking, fire-and-forget)
    # ========================================================================

    def remember_profile(
        self,
        user_id: str,
        user_prompt: str,
        clarifications: list[tuple[str, str]] | None = None,
    ):
        """Remember user profile from initial conversation."""
        if not self.available:
            return
        writer.fire_and_forget(
            writer.write_profile_async(user_id, user_prompt, clarifications)
        )

    def remember_run(
        self,
        user_id: str,
        topology: dict,
        execution_result: dict,
        user_rating: int | None = None,
    ):
        """Remember a run summary. Called after every run."""
        if not self.available:
            return
        writer.fire_and_forget(
            writer.write_run_summary_async(
                user_id, topology, execution_result, user_rating
            )
        )

    def remember_node_output(
        self,
        user_id: str,
        role: str,
        node_output: str,
        latency_seconds: float,
        model_used: str,
        success: bool,
    ):
        """Remember a node's observations for its diary."""
        if not self.available:
            return
        writer.fire_and_forget(
            writer.write_agent_diary_async(
                user_id, role, node_output, latency_seconds, model_used, success
            )
        )

    def remember_correction(
        self,
        user_id: str,
        rating: int,
        comment: str | None = None,
        topology_summary: str | None = None,
    ):
        """Remember user correction from low rating."""
        if not self.available:
            return
        writer.fire_and_forget(
            writer.write_correction_async(
                user_id, rating, comment, topology_summary
            )
        )

    # ========================================================================
    # Formatting helpers
    # ========================================================================

    def format_for_master_prompt(self, user_id: str, user_prompt: str) -> str:
        """
        Build full context block for master agent system prompt.

        Combines profile + past history into prompt-ready text.
        """
        if not self.available:
            return ""

        profile = reader.load_user_profile(user_id)
        history = reader.load_master_history(user_id, user_prompt)

        return (
            reader.format_profile_for_prompt(profile)
            + reader.format_history_for_prompt(history)
        )

    def format_for_node_prompt(self, user_id: str, role: str, task: str) -> str:
        """Build context block for node system prompt."""
        if not self.available:
            return ""

        diary = reader.load_agent_diary(user_id, role, query=task)
        return reader.format_diary_for_prompt(diary, role)


# Module-level singleton (for simple imports)
_default_memory: DaapMemory | None = None


def get_memory() -> DaapMemory:
    """Get or create default DaapMemory instance."""
    global _default_memory
    if _default_memory is None:
        _default_memory = DaapMemory()
    return _default_memory
