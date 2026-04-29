"""
Linear Thompson Sampling for contextual bandits.

Reference: Agrawal & Goyal (2013) "Thompson Sampling for Contextual
Bandits with Linear Payoffs", ICML.

Each (node_role, arm) pair maintains its own LinTS model.
Arms = model tiers: ["fast", "smart", "powerful"]
Context = TaskContext feature vector.

No external ML libraries beyond numpy.
"""

import numpy as np
from dataclasses import dataclass, field


ARMS = ["fast", "smart", "powerful"]


@dataclass
class ArmState:
    """
    Bayesian linear regression state for one arm.

    Maintains:
      B: d×d precision matrix (inverse covariance). Init = identity × lambda.
      f: d×1 vector. Cumulative context-weighted rewards.
      mu: d×1 vector. Current mean estimate = B^-1 @ f.
      n_pulls: number of times this arm has been selected.
    """
    dim: int
    B: np.ndarray = field(default=None)
    f: np.ndarray = field(default=None)
    mu: np.ndarray = field(default=None)
    n_pulls: int = 0

    def __post_init__(self):
        if self.B is None:
            self.B = np.eye(self.dim, dtype=np.float64)
        if self.f is None:
            self.f = np.zeros(self.dim, dtype=np.float64)
        if self.mu is None:
            self.mu = np.zeros(self.dim, dtype=np.float64)


class LinTSBandit:
    """
    One contextual bandit for one node role.

    Example: role="researcher" has 3 arms: fast, smart, powerful.
    Given context features, picks which model tier to use.

    Usage:
        bandit = LinTSBandit(dim=10)
        arm = bandit.select_arm(context_vector)
        # ... execute with that arm ...
        bandit.update(context_vector, arm, reward)
    """

    def __init__(
        self,
        dim: int,
        v_squared: float = 1.0,
        arms: list[str] = None,
    ):
        """
        Args:
            dim: dimension of context feature vector
            v_squared: exploration parameter. Higher = more exploration.
            arms: list of arm names. Default = ["fast", "smart", "powerful"]
        """
        self.dim = dim
        self.v_squared = v_squared
        self.arms = arms or ARMS

        self.arm_states: dict[str, ArmState] = {
            arm: ArmState(dim=dim) for arm in self.arms
        }

    def select_arm(
        self,
        context: np.ndarray,
        rng: np.random.Generator = None,
        greedy: bool = False,
    ) -> str:
        """
        Select an arm using Thompson Sampling (default) or greedy argmax.

        greedy=True: pure exploitation — score = context @ mu (no sampling).
        greedy=False: Thompson Sampling — samples theta from posterior.

        Args:
            context: feature vector of shape (dim,)
            rng: numpy random generator (for reproducibility in tests)
            greedy: if True, use argmax(mu · context) — fully deterministic

        Returns:
            arm name (e.g., "fast", "smart", "powerful")
        """
        context = np.asarray(context, dtype=np.float64)
        assert context.shape == (self.dim,), f"Expected dim {self.dim}, got {context.shape}"

        if rng is None:
            rng = np.random.default_rng()

        best_arm = None
        best_score = -np.inf

        for arm_name, state in self.arm_states.items():
            B_inv = np.linalg.inv(state.B)
            mu = B_inv @ state.f

            if greedy:
                theta = mu
            else:
                try:
                    theta = rng.multivariate_normal(
                        mean=mu,
                        cov=self.v_squared * B_inv,
                    )
                except np.linalg.LinAlgError:
                    theta = mu

            score = context @ theta

            if score > best_score:
                best_score = score
                best_arm = arm_name

        return best_arm

    def update(self, context: np.ndarray, arm: str, reward: float):
        """
        Update the posterior after observing a reward.

        B_a ← B_a + context × context^T
        f_a ← f_a + reward × context

        Args:
            context: feature vector of shape (dim,)
            arm: which arm was pulled
            reward: observed reward (0-1 scale)
        """
        context = np.asarray(context, dtype=np.float64)
        assert arm in self.arm_states, f"Unknown arm: {arm}"

        state = self.arm_states[arm]
        state.B += np.outer(context, context)
        state.f += reward * context
        state.mu = np.linalg.inv(state.B) @ state.f
        state.n_pulls += 1

    def get_total_pulls(self) -> int:
        return sum(s.n_pulls for s in self.arm_states.values())

    def get_arm_stats(self) -> dict[str, dict]:
        stats = {}
        for arm_name, state in self.arm_states.items():
            B_inv = np.linalg.inv(state.B)
            mu = B_inv @ state.f
            stats[arm_name] = {
                "n_pulls": state.n_pulls,
                "mean_weights": mu.tolist(),
                "uncertainty": float(np.trace(B_inv)),
            }
        return stats


class TopologyOptimizer:
    """
    Top-level optimizer. Manages one LinTS bandit per node role.

    Usage:
        opt = TopologyOptimizer()

        # Before generating topology:
        recs = opt.recommend(context, roles=["researcher", "evaluator", "writer"])
        # recs = {"researcher": "fast", "evaluator": "smart", "writer": "smart"}

        # After user rates the run:
        opt.update(context, node_configs={"researcher": "fast", ...}, reward=0.8)
    """

    def __init__(self, dim: int = None, v_squared: float = 1.0):
        from daap.optimizer.context import CONTEXT_DIM
        self.dim = dim or CONTEXT_DIM
        self.v_squared = v_squared
        self.bandits: dict[str, LinTSBandit] = {}

    def _get_or_create_bandit(self, role: str) -> LinTSBandit:
        role_key = self._normalize_role(role)
        if role_key not in self.bandits:
            self.bandits[role_key] = LinTSBandit(
                dim=self.dim,
                v_squared=self.v_squared,
            )
        return self.bandits[role_key]

    def recommend(
        self,
        context: list[float],
        roles: list[str],
        rng: np.random.Generator = None,
        greedy: bool = True,
    ) -> dict[str, str]:
        """
        Get model tier recommendations for each node role.

        greedy=True (default): deterministic argmax — consistent output across runs.
        greedy=False: Thompson Sampling — stochastic exploration.

        Args:
            context: feature vector from extract_context().to_vector()
            roles: list of node roles in the topology
            rng: random generator for reproducibility (used only if greedy=False)
            greedy: if True, return argmax(mu · context) — no sampling noise

        Returns:
            {role: tier} dict. e.g., {"researcher": "fast", "writer": "smart"}
        """
        ctx = np.asarray(context, dtype=np.float64)
        recommendations = {}

        for role in roles:
            bandit = self._get_or_create_bandit(role)
            arm = bandit.select_arm(ctx, rng=rng, greedy=greedy)
            recommendations[role] = arm

        return recommendations

    def update(
        self,
        context: list[float],
        node_configs: dict[str, str],
        reward: float,
    ):
        """
        Update all role bandits after a run.

        Args:
            context: feature vector from extract_context().to_vector()
            node_configs: {role: tier} that was actually used
            reward: 0-1 scaled reward from user rating
        """
        ctx = np.asarray(context, dtype=np.float64)

        for role, tier in node_configs.items():
            bandit = self._get_or_create_bandit(role)
            bandit.update(ctx, tier, reward)

    def get_stats(self) -> dict:
        return {
            role: bandit.get_arm_stats()
            for role, bandit in self.bandits.items()
        }

    @staticmethod
    def _normalize_role(role: str) -> str:
        """
        Normalize free-text role to canonical key.

        "Lead Researcher" → "researcher"
        "Email Writer & Personalizer" → "writer"
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
