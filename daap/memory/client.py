"""
DAAP Memory Client — DAAP-specific wrapper around Mem0.

All mem0 internals are contained here. If we swap Mem0 for another
memory backend, only this file changes.

Memory is scoped by:
  user_id  — per-user memories (profile, preferences, run history)
  agent_id — per-agent-role memories (researcher, writer learnings)
  run_id   — per-run context (specific topology execution)
"""

from daap.memory.setup import create_memory_client


class DaapMemory:
    """
    DAAP's memory interface. Wraps Mem0.

    Instantiate once per server process and share across sessions.
    All read/write operations fail gracefully — memory is an
    optimization, not a hard dependency.
    """

    def __init__(self, mode: str = "production"):
        self.mem = create_memory_client(mode)

    # ------------------------------------------------------------------
    # User profile
    # ------------------------------------------------------------------

    def store_user_profile(self, user_id: str, profile_text: str) -> None:
        """
        Store user profile information as natural language.

        Mem0 auto-extracts individual facts. Pass complete sentences:
        "User sells project management SaaS. Target: mid-size construction
         companies (50-500 employees). ICP: operations managers."
        """
        self.mem.add(
            profile_text,
            user_id=user_id,
            metadata={"category": "profile"},
        )

    def store_user_preference(self, user_id: str, preference_text: str) -> None:
        """Store a user preference or correction."""
        self.mem.add(
            preference_text,
            user_id=user_id,
            metadata={"category": "preferences"},
        )

    def get_user_context(self, user_id: str, query: str | None = None) -> dict:
        """
        Return structured user context for the master agent system prompt.

        Returns dict with keys: profile, preferences, recent_runs.
        Each value is a list of memory strings.
        Returns empty lists if no memories exist.
        """
        def _search(q: str, limit: int) -> list[str]:
            results = self.mem.search(query=q, user_id=user_id, limit=limit)
            return [m.get("memory", "") for m in results.get("results", [])]

        q = query or ""
        return {
            "profile": _search(q or "user profile product ICP", 10),
            "preferences": _search(q or "user preferences style format tone", 5),
            "recent_runs": _search(q or "past run topology result cost latency", 5),
        }

    # ------------------------------------------------------------------
    # Run history
    # ------------------------------------------------------------------

    def store_run_result(
        self,
        user_id: str,
        run_summary: str,
        run_id: str | None = None,
    ) -> None:
        """
        Store a run result summary as natural language.

        Example:
        "Completed sales pipeline for construction companies.
         Found 18 leads, drafted 10 emails. Cost: $0.11. Latency: 43s.
         User rated 4/5. Emails were slightly too long."
        """
        self.mem.add(
            run_summary,
            user_id=user_id,
            run_id=run_id,
            metadata={"category": "runs"},
        )

    def store_user_rating(self, user_id: str, rating_text: str) -> None:
        """Store user feedback/rating for a run."""
        self.mem.add(
            rating_text,
            user_id=user_id,
            metadata={"category": "feedback"},
        )

    # ------------------------------------------------------------------
    # Agent diary (shared across users, scoped by role)
    # ------------------------------------------------------------------

    def store_agent_learning(self, agent_role: str, learning_text: str) -> None:
        """
        Store a learning from an agent role.

        agent_role: "researcher", "evaluator", "writer", etc.
        Learnings are shared across users — they capture what works
        for a given task type, not user-specific data.
        """
        self.mem.add(
            learning_text,
            agent_id=f"daap_{agent_role}",
            metadata={"category": "agent_diary"},
        )

    def get_agent_learnings(self, agent_role: str, query: str) -> list[str]:
        """
        Retrieve relevant learnings for an agent role.

        Used to enrich node system prompts before execution.
        Returns empty list if no learnings exist.
        """
        results = self.mem.search(
            query=query,
            agent_id=f"daap_{agent_role}",
            limit=5,
        )
        return [m.get("memory", "") for m in results.get("results", [])]

    # ------------------------------------------------------------------
    # Raw operations
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Direct add to Mem0. For cases not covered by specific methods."""
        self.mem.add(
            text,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata,
        )

    def search(
        self,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Direct search on Mem0."""
        return self.mem.search(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            limit=limit,
        )

    def get_all(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[dict]:
        """Get all memories for a scope."""
        return self.mem.get_all(user_id=user_id, agent_id=agent_id)
