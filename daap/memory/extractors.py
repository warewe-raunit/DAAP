"""
Structured extractors for memory writes.

Mem0 takes messages (list of dicts) or strings and runs its own LLM extraction.
We pre-structure inputs into clear statements so Mem0's extractor has better signal.

Each extractor returns a list of messages suitable for mem0.Memory.add().
"""

from typing import Any


def extract_profile_from_conversation(
    user_prompt: str,
    clarifications: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """
    Extract profile-worthy facts from initial user conversation.

    Args:
        user_prompt: original user request
        clarifications: list of (question, answer) from ask_user tool

    Returns messages ready for mem0.add(messages=...)
    """
    messages = [
        {"role": "user", "content": user_prompt},
    ]

    if clarifications:
        for q, a in clarifications:
            messages.append({"role": "assistant", "content": q})
            messages.append({"role": "user", "content": a})

    return messages


def extract_run_summary(
    topology: dict,
    execution_result: dict,
    user_rating: int | None = None,
) -> str:
    """
    Create a run summary string for memory storage.

    Succinct: 2-3 sentences. Mem0's extractor will pull out key facts.
    """
    node_count = len(topology.get("nodes", []))
    node_roles = [n.get("role", "unknown") for n in topology.get("nodes", [])]
    cost = execution_result.get("total_cost_usd") or execution_result.get("cost_usd", 0)
    latency = execution_result.get("total_latency_seconds") or execution_result.get("latency_seconds", 0)
    success = execution_result.get("success", False)

    parts = [
        f"Ran a {node_count}-node topology ({', '.join(node_roles)}) "
        f"in {latency:.1f}s at ${cost:.4f}.",
    ]

    if success:
        parts.append("Execution succeeded.")
    else:
        err = execution_result.get("error", "unknown error")
        parts.append(f"Execution failed: {err}.")

    if user_rating is not None:
        parts.append(f"User rated it {user_rating}/5.")

    return " ".join(parts)


def extract_agent_observation(
    role: str,
    node_output: str,
    latency_seconds: float,
    model_used: str,
    success: bool,
) -> str:
    """
    Create per-node observation for agent diary.

    Captures what the agent did, what worked, what to remember.
    Truncates output to keep storage efficient.
    """
    output_preview = node_output[:500] if node_output else "(empty)"
    outcome = "completed" if success else "failed"

    return (
        f"As {role} using {model_used}, {outcome} in {latency_seconds:.1f}s. "
        f"Output preview: {output_preview}"
    )


def extract_correction_from_rating(
    rating: int,
    comment: str | None = None,
    topology_summary: str | None = None,
) -> str | None:
    """
    When user gives low rating, extract explicit correction signal.

    Only creates memory for ratings ≤ 2 (explicit negative feedback).
    Returns None if rating is not low (no correction to capture).
    """
    if rating > 2:
        return None

    parts = [f"User rated a run {rating}/5 (low)."]
    if comment:
        parts.append(f"User said: \"{comment}\".")
    if topology_summary:
        parts.append(f"Context: {topology_summary}")
    parts.append("Avoid this pattern in future runs for this user.")

    return " ".join(parts)
