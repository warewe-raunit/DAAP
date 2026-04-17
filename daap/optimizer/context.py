"""
Context feature extraction for the contextual bandit.

Features must be:
  1. Available before execution (can't use results as features)
  2. Numeric (LinTS needs vectors)
  3. Normalized to roughly [0, 1] range (helps LinTS convergence)
"""

from dataclasses import dataclass


# Task type keywords → category index
TASK_CATEGORIES = {
    "lead": 0,       # lead generation
    "research": 1,   # market/company research
    "email": 2,      # email drafting
    "outreach": 3,   # full outreach pipeline
    "analysis": 4,   # data analysis
    "other": 5,
}
NUM_TASK_CATEGORIES = len(TASK_CATEGORIES)


@dataclass
class TaskContext:
    """Context features for one bandit decision."""
    task_category: int     # one-hot encoded below
    node_count: float      # normalized: n_nodes / 10
    has_parallel: float    # 1.0 if any node has parallel_instances > 1
    user_maturity: float   # min(user_total_runs / 50, 1.0)

    def to_vector(self) -> list[float]:
        """
        Convert to feature vector for LinTS.

        Returns fixed-length vector. Dimension = d.
        Structure:
          [0:6]  = one-hot task category (6 dims)
          [6]    = node_count (normalized)
          [7]    = has_parallel (binary)
          [8]    = user_maturity (0-1)
          [9]    = bias term (always 1.0)

        Total dimension: 10
        """
        one_hot = [0.0] * NUM_TASK_CATEGORIES
        if 0 <= self.task_category < NUM_TASK_CATEGORIES:
            one_hot[self.task_category] = 1.0

        return one_hot + [
            self.node_count,
            self.has_parallel,
            self.user_maturity,
            1.0,  # bias term
        ]


# Feature vector dimension
CONTEXT_DIM = 10


def extract_context(
    user_prompt: str,
    node_count: int,
    has_parallel: bool,
    user_total_runs: int,
) -> TaskContext:
    """
    Extract context features from a task before execution.

    Args:
        user_prompt: raw user request text
        node_count: number of nodes in proposed topology
        has_parallel: whether any node has parallel_instances > 1
        user_total_runs: how many runs this user has done total

    Returns:
        TaskContext with normalized features
    """
    category = _classify_task(user_prompt)

    return TaskContext(
        task_category=category,
        node_count=min(node_count / 10.0, 1.0),
        has_parallel=1.0 if has_parallel else 0.0,
        user_maturity=min(user_total_runs / 50.0, 1.0),
    )


def _classify_task(prompt: str) -> int:
    """Classify user prompt into task category via keyword matching."""
    prompt_lower = prompt.lower()

    if any(w in prompt_lower for w in ["email", "draft", "write", "message"]):
        return TASK_CATEGORIES["email"]
    if any(w in prompt_lower for w in ["lead", "prospect", "find companies", "find people"]):
        return TASK_CATEGORIES["lead"]
    if any(w in prompt_lower for w in ["research", "analyze", "investigate", "market"]):
        return TASK_CATEGORIES["research"]
    if any(w in prompt_lower for w in ["outreach", "campaign", "pipeline", "automate"]):
        return TASK_CATEGORIES["outreach"]
    if any(w in prompt_lower for w in ["analysis", "data", "report", "compare"]):
        return TASK_CATEGORIES["analysis"]

    return TASK_CATEGORIES["other"]


def compute_reward(
    user_rating: int,
    actual_cost_usd: float,
    budget_usd: float,
    latency_seconds: float,
    timeout_seconds: float,
) -> float:
    """
    Compute bandit reward from run outcome.

    Reward = weighted combination of:
      - Quality (user rating, normalized to 0-1): weight 0.70
      - Cost efficiency (how far under budget): weight 0.15
      - Latency efficiency (how far under timeout): weight 0.15

    Returns:
        float in [0, 1] range
    """
    quality = (user_rating - 1) / 4.0  # 1→0, 2→0.25, 3→0.5, 4→0.75, 5→1.0

    if budget_usd > 0:
        cost_eff = max(0.0, 1.0 - (actual_cost_usd / budget_usd))
    else:
        cost_eff = 0.5  # no budget set — neutral

    if timeout_seconds > 0:
        latency_eff = max(0.0, 1.0 - (latency_seconds / timeout_seconds))
    else:
        latency_eff = 0.5

    reward = (
        0.70 * quality +
        0.15 * cost_eff +
        0.15 * latency_eff
    )

    return max(0.0, min(1.0, reward))
