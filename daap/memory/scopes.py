"""
Memory scoping helpers.

Mem0 supports three scope dimensions:
  - user_id: who this memory is about
  - agent_id: which agent role learned this
  - run_id: which specific run generated this (for cleanup/audit)

DAAP convention:
  user_id   = DAAP user_id (passed from session)
  agent_id  = node role ("researcher", "writer", etc.) OR "master" OR "profile"
  run_id    = topology_id when applicable

Memory types by agent_id:
  "profile"       → user-level facts (ICP, product, preferences)
  "master"        → master agent learnings (topology patterns that worked)
  <role>_diary    → per-role agent diaries (e.g., "researcher_diary")
"""


def profile_scope(user_id: str) -> dict:
    """Scope for user profile memories (ICP, product, preferences)."""
    return {
        "user_id": user_id,
        "agent_id": "profile",
    }


def master_scope(user_id: str) -> dict:
    """Scope for master agent learnings (topology patterns)."""
    return {
        "user_id": user_id,
        "agent_id": "master",
    }


def agent_diary_scope(user_id: str, role: str) -> dict:
    """Scope for per-role agent diary."""
    normalized = _normalize_role(role)
    return {
        "user_id": user_id,
        "agent_id": f"{normalized}_diary",
    }


def run_scope(user_id: str, topology_id: str) -> dict:
    """Scope for a single run (for targeted cleanup/audit)."""
    return {
        "user_id": user_id,
        "run_id": topology_id,
    }


def all_user_scope(user_id: str) -> dict:
    """Match all memories for a user (for export/delete)."""
    return {"user_id": user_id}


def _normalize_role(role: str) -> str:
    """
    Normalize free-text role to canonical key.
    Matches TopologyOptimizer._normalize_role in optimizer/bandit.py.
    """
    role_lower = role.lower().strip()
    if any(w in role_lower for w in ["research", "search", "find", "discover"]):
        return "researcher"
    if any(w in role_lower for w in ["evaluat", "scor", "rank", "qualif"]):
        return "evaluator"
    if any(w in role_lower for w in ["writ", "draft", "compos", "email"]):
        return "writer"
    if any(w in role_lower for w in ["personal", "custom", "tailor"]):
        return "personalizer"
    if any(w in role_lower for w in ["format", "clean", "transform"]):
        return "formatter"
    return role_lower.replace(" ", "_")
