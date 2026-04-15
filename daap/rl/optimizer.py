"""Topology optimizer integrating Thompson Sampling and training-free GRPO context."""

from __future__ import annotations

import copy
import logging
import re
import time
from typing import Any

from daap.rl.bandit import ARM_CONFIGS, TIER_PROFILES, ThompsonBandit
from daap.rl.experience import ExperienceStore
from daap.rl.reward import compute_reward

logger = logging.getLogger(__name__)

VALID_MODEL_TIERS = {"fast", "smart", "powerful"}


class TopologyOptimizer:
    """Recommends topology overrides and learns online from run outcomes."""

    def __init__(self, db_path: str = "optimizer.db"):
        self.db_path = db_path
        self._bandit: ThompsonBandit | None = None
        self._experience_store: ExperienceStore | None = None

        self._pending: dict[str, dict[str, Any]] = {}
        self._pending_ratings: dict[str, int] = {}
        self._topology_to_exp: dict[str, str] = {}

    def _get_bandit(self) -> ThompsonBandit:
        if self._bandit is None:
            self._bandit = ThompsonBandit(db_path=self.db_path)
        return self._bandit

    def _get_experience_store(self) -> ExperienceStore:
        if self._experience_store is None:
            self._experience_store = ExperienceStore(self._get_bandit())
        return self._experience_store

    def classify_task_type(self, user_prompt: str) -> str:
        """
        Classify user_prompt into a task_type string.

        When DAAP_RL_USE_EMBEDDINGS=1: uses cosine similarity to pre-computed
        centroids via free/cheap OpenRouter embedding models. Falls through to
        keyword classification on any failure.

        Keyword classification (always available, zero cost):
        lead_generation | email_outreach | qualification | research |
        content | data_processing | general
        """
        import os
        if os.environ.get("DAAP_RL_USE_EMBEDDINGS") == "1":
            try:
                from daap.rl.embedder import get_centroid_classifier
                classifier = get_centroid_classifier(self.db_path)
                result = classifier.classify(user_prompt)
                if result is not None:
                    return result
                logger.debug("Embedding classify returned None — falling back to keywords")
            except Exception as exc:
                logger.debug("Embedding classify failed (%s) — falling back to keywords", exc)

        return self._classify_by_keywords(user_prompt)

    @staticmethod
    def _classify_by_keywords(user_prompt: str) -> str:
        """Keyword-based task classification. Zero cost, always works."""
        prompt = (user_prompt or "").lower()
        if any(token in prompt for token in ("email", "outreach", "subject line", "cold message", "follow-up")):
            return "email_outreach"
        if any(token in prompt for token in ("qualify", "qualification", "score lead", "ranking")):
            return "qualification"
        if any(token in prompt for token in ("lead", "prospect", "contact list", "pipeline")):
            return "lead_generation"
        if any(token in prompt for token in ("research", "analyze", "investigate", "web search")):
            return "research"
        if any(token in prompt for token in ("blog", "article", "content", "copywriting", "landing page")):
            return "content"
        if any(token in prompt for token in ("csv", "spreadsheet", "transform", "clean data", "parse")):
            return "data_processing"
        return "general"

    def build_task_fingerprint(self, task_type: str, topo_dict: dict[str, Any]) -> str:
        nodes = topo_dict.get("nodes") if isinstance(topo_dict.get("nodes"), list) else []

        role_tokens: list[str] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            raw_role = str(node.get("role", "")).strip().lower()
            if not raw_role:
                continue
            normalized = re.sub(r"[^a-z0-9]+", "_", raw_role).strip("_")
            if normalized:
                role_tokens.append(normalized)

        roles = sorted(set(role_tokens))
        role_part = "|".join(roles) if roles else "none"
        normalized_task = (task_type or "general").strip().lower() or "general"
        return f"{normalized_task}:{role_part}"

    def recommend_overrides(self, topo_dict: dict[str, Any], task_type: str | None = None) -> dict[str, Any]:
        if not isinstance(topo_dict, dict):
            return topo_dict

        try:
            working_copy = copy.deepcopy(topo_dict)
            prompt = str(working_copy.get("user_prompt", ""))
            selected_task_type = task_type or self.classify_task_type(prompt)
            fingerprint = self.build_task_fingerprint(selected_task_type, working_copy)
            arm_id = self._get_bandit().select_arm(selected_task_type)

            updated = self._apply_arm_overrides(working_copy, arm_id)
            topology_id = str(updated.get("topology_id") or topo_dict.get("topology_id") or "unknown")

            self._pending[topology_id] = {
                "arm_id": arm_id,
                "task_fingerprint": fingerprint,
                "task_type": selected_task_type,
                "selected_at": time.time(),
            }
            return updated
        except Exception as exc:
            logger.warning("RL recommend_overrides failed (non-fatal): %s", exc)
            return topo_dict

    def _apply_arm_overrides(self, topo_dict: dict[str, Any], arm_id: str) -> dict[str, Any]:
        config = ARM_CONFIGS.get(arm_id)
        if config is None:
            return topo_dict

        profile = str(config["profile"])
        instance_count = int(config["instance_count"])

        nodes = topo_dict.get("nodes")
        if not isinstance(nodes, list):
            return topo_dict

        for idx, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue

            if self._has_model_override(node):
                continue

            tier = self._resolve_node_tier(node, nodes, idx, profile)
            if tier in VALID_MODEL_TIERS:
                node["model_tier"] = tier

            instance_cfg = node.get("instance_config")
            if not isinstance(instance_cfg, dict):
                instance_cfg = {}
                node["instance_config"] = instance_cfg

            try:
                current_parallel = int(instance_cfg.get("parallel_instances", 1))
            except Exception:
                current_parallel = 1

            if current_parallel > 1:
                instance_cfg["parallel_instances"] = instance_count
                if instance_count > 1 and not instance_cfg.get("consolidation"):
                    instance_cfg["consolidation"] = "merge"

        return topo_dict

    def record_outcome(
        self,
        topo_dict: dict[str, Any] | None,
        result: Any,
        task_type: str | None = None,
    ) -> None:
        try:
            topo = topo_dict if isinstance(topo_dict, dict) else {}
            normalized = self._normalize_result(result, topo)
            topology_id = str(normalized.get("topology_id") or topo.get("topology_id") or "unknown")

            pending = self._pending.pop(topology_id, None)
            if pending is None:
                return

            selected_task_type = task_type or str(pending.get("task_type", "general"))
            arm_id = str(pending.get("arm_id", "budget_1"))
            fingerprint = str(pending.get("task_fingerprint", "general:none"))

            rating = self._pending_ratings.pop(topology_id, None)
            reward = compute_reward(normalized, topology=topo, rating=rating)

            outcome_payload = dict(normalized)
            outcome_payload["task_type"] = selected_task_type
            outcome_payload["task_fingerprint"] = fingerprint
            outcome_payload["rating"] = rating
            outcome_payload["rating_applied"] = rating is not None

            bandit = self._get_bandit()
            bandit.update(selected_task_type, arm_id, reward)
            exp_id = bandit.log_experience(
                task_fingerprint=fingerprint,
                arm_id=arm_id,
                reward=reward,
                topology_json=topo,
                outcome_json=outcome_payload,
            )
            self._topology_to_exp[topology_id] = exp_id
        except Exception as exc:
            logger.warning("RL record_outcome failed (non-fatal): %s", exc)

    def record_rating(self, topology_id: str, rating: int) -> None:
        try:
            normalized_topology_id = str(topology_id or "").strip()
            if not normalized_topology_id:
                return

            clamped_rating = max(1, min(5, int(rating)))
            self._pending_ratings[normalized_topology_id] = clamped_rating
            self._retroactive_rating_update(normalized_topology_id, clamped_rating)
        except Exception as exc:
            logger.warning("RL record_rating failed (non-fatal): %s", exc)

    def _retroactive_rating_update(self, topology_id: str, rating: int) -> None:
        exp_id = self._topology_to_exp.get(topology_id)
        bandit = self._get_bandit()

        exp = bandit.get_experience(exp_id) if exp_id else None
        if exp is None:
            exp = bandit.find_latest_experience_by_topology_id(topology_id)
            if exp is not None:
                self._topology_to_exp[topology_id] = str(exp["exp_id"])

        if exp is None:
            return

        outcome = exp.get("outcome_json")
        topology = exp.get("topology_json")
        if not isinstance(outcome, dict):
            return

        if outcome.get("rating_applied"):
            self._pending_ratings.pop(topology_id, None)
            return

        task_type = str(outcome.get("task_type") or "")
        if not task_type:
            task_fingerprint = str(exp.get("task_fingerprint", "general:none"))
            task_type = task_fingerprint.split(":", 1)[0] if ":" in task_fingerprint else task_fingerprint
            task_type = task_type or "general"

        old_reward = float(exp.get("reward", 0.0))
        new_reward = compute_reward(outcome, topology=topology if isinstance(topology, dict) else None, rating=rating)
        delta = new_reward - old_reward

        bandit.adjust(task_type, str(exp.get("arm_id", "budget_1")), delta)

        outcome["rating"] = int(rating)
        outcome["rating_applied"] = True
        bandit.update_experience(str(exp["exp_id"]), new_reward, outcome_json=outcome)

        self._pending_ratings.pop(topology_id, None)

    def get_rl_prompt_section(
        self,
        task_type: str,
        topo_dict: dict[str, Any] | None = None,
    ) -> str | None:
        try:
            topo = topo_dict if isinstance(topo_dict, dict) else {"nodes": []}
            normalized_task = (task_type or "general").strip().lower() or "general"
            fingerprint = self.build_task_fingerprint(normalized_task, topo)
            return self._get_experience_store().build_rl_context_section(fingerprint)
        except Exception as exc:
            logger.warning("RL prompt insight fetch failed (non-fatal): %s", exc)
            return None

    @staticmethod
    def _has_model_override(node: dict[str, Any]) -> bool:
        operator_override = node.get("operator_override")
        if not isinstance(operator_override, dict):
            return False
        model_map = operator_override.get("model_map")
        return isinstance(model_map, dict) and bool(model_map)

    def _resolve_node_tier(
        self,
        node: dict[str, Any],
        nodes: list[Any],
        index: int,
        profile: str,
    ) -> str:
        role = str(node.get("role", "")).lower()
        agent_mode = str(node.get("agent_mode", "")).lower()
        has_tools = isinstance(node.get("tools"), list) and len(node.get("tools")) > 0

        if profile == "budget":
            return TIER_PROFILES["budget"]["default"]

        if profile == "balanced":
            if has_tools or agent_mode == "react":
                return TIER_PROFILES["balanced"]["tool_or_react"]
            if any(key in role for key in ("write", "email", "content", "summar", "evaluat", "score", "rank")):
                return TIER_PROFILES["balanced"]["writing_or_eval"]
            return TIER_PROFILES["balanced"]["default"]

        if profile == "quality":
            if index == len(nodes) - 1:
                return TIER_PROFILES["quality"]["synthesis"]
            if has_tools or agent_mode == "react":
                return TIER_PROFILES["quality"]["tool_or_react"]
            return TIER_PROFILES["quality"]["middle"]

        return "smart"

    @staticmethod
    def _normalize_result(result: Any, topo_dict: dict[str, Any]) -> dict[str, Any]:
        if isinstance(result, dict):
            data = dict(result)
        else:
            data = {
                "topology_id": getattr(result, "topology_id", None),
                "success": getattr(result, "success", False),
                "error": getattr(result, "error", None),
                "final_output": getattr(result, "final_output", ""),
                "latency_seconds": getattr(result, "latency_seconds", None),
                "total_latency_seconds": getattr(result, "total_latency_seconds", None),
                "total_input_tokens": getattr(result, "total_input_tokens", 0),
                "total_output_tokens": getattr(result, "total_output_tokens", 0),
            }

        if data.get("latency_seconds") is None and data.get("total_latency_seconds") is not None:
            data["latency_seconds"] = data.get("total_latency_seconds")

        if data.get("topology_id") is None:
            data["topology_id"] = topo_dict.get("topology_id", "unknown")

        data["success"] = bool(data.get("success", False))

        try:
            data["latency_seconds"] = float(data.get("latency_seconds", 0.0) or 0.0)
        except Exception:
            data["latency_seconds"] = 0.0

        try:
            data["total_input_tokens"] = int(data.get("total_input_tokens", 0) or 0)
        except Exception:
            data["total_input_tokens"] = 0

        try:
            data["total_output_tokens"] = int(data.get("total_output_tokens", 0) or 0)
        except Exception:
            data["total_output_tokens"] = 0

        return data
