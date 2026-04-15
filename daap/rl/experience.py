"""Experience store and training-free GRPO insight extraction."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from typing import Any

from daap.rl.bandit import ThompsonBandit

logger = logging.getLogger(__name__)

GRPO_MODEL = "google/gemini-2.5-flash-lite"
MIN_EXPERIENCES_FOR_GRPO = 5
GRPO_RECOMPUTE_INTERVAL = 10
GRPO_CACHE_TTL_SECONDS = 3600
GRPO_MAX_EXPERIENCES = 30


class ExperienceStore:
    """Builds cached RL prompt insights by comparing high/low reward trajectories."""

    def __init__(self, bandit: ThompsonBandit):
        self.bandit = bandit
        self._cache: dict[str, dict[str, Any]] = {}

    def get_grpo_insights(self, task_fingerprint: str) -> str | None:
        experiences = self.bandit.get_experiences(task_fingerprint, limit=GRPO_MAX_EXPERIENCES)
        if len(experiences) < MIN_EXPERIENCES_FOR_GRPO:
            return None

        now = time.time()
        cached = self._cache.get(task_fingerprint)
        if cached is not None:
            cache_age = now - float(cached["computed_at"])
            n_exp = int(cached["n_exp"])
            if cache_age < GRPO_CACHE_TTL_SECONDS and (len(experiences) - n_exp) < GRPO_RECOMPUTE_INTERVAL:
                return str(cached["insights"])

        try:
            insights = self._extract_insights_via_llm(experiences, task_fingerprint)
        except Exception as exc:
            logger.warning("GRPO insight extraction failed (non-fatal): %s", exc)
            insights = None

        if not insights:
            return str(cached["insights"]) if cached is not None else None

        self._cache[task_fingerprint] = {
            "insights": insights,
            "computed_at": now,
            "n_exp": len(experiences),
        }
        return insights

    def _extract_insights_via_llm(self, experiences: list[dict[str, Any]], task_fingerprint: str) -> str:
        ranked = sorted(experiences, key=lambda item: float(item.get("reward", 0.0)), reverse=True)
        split_size = max(1, int(len(ranked) * 0.4))

        high = ranked[:split_size]
        low = ranked[-split_size:]

        payload = {
            "task_fingerprint": task_fingerprint,
            "high_reward": [self._summarize_exp(item) for item in high],
            "low_reward": [self._summarize_exp(item) for item in low],
        }

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            return self._fallback_insights(high, low)

        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()

        try:
            from openai import OpenAI
        except Exception:
            logger.warning("openai package unavailable; using fallback GRPO insights")
            return self._fallback_insights(high, low)

        client = OpenAI(api_key=api_key, base_url=base_url)

        prompt = (
            "Compare high-reward vs low-reward DAAP topology outcomes for the same task fingerprint. "
            "Output only 2-4 concise bullet points. Each bullet should describe a practical pattern "
            "about arm selection, instance count, model tier mix, latency-cost tradeoff, or success correlation. "
            "Do not add headings or disclaimers.\n\n"
            f"DATA:\n{json.dumps(payload, indent=2)}"
        )

        response = client.chat.completions.create(
            model=GRPO_MODEL,
            temperature=0.1,
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an RL analyst. Return terse actionable bullets only. "
                        "No markdown headers, no prose paragraphs."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        text = ""
        if response.choices:
            message = response.choices[0].message
            text = (message.content or "").strip()

        cleaned = self._normalize_bullets(text)
        if cleaned:
            return cleaned

        return self._fallback_insights(high, low)

    def build_rl_context_section(self, task_fingerprint: str) -> str | None:
        insights = self.get_grpo_insights(task_fingerprint)
        if not insights:
            return None
        return "## RL Optimization Insights\n" + insights

    @staticmethod
    def _summarize_exp(exp: dict[str, Any]) -> dict[str, Any]:
        topology = exp.get("topology_json")
        outcome = exp.get("outcome_json")

        topology_dict = topology if isinstance(topology, dict) else {}
        outcome_dict = outcome if isinstance(outcome, dict) else {}

        nodes = topology_dict.get("nodes") if isinstance(topology_dict.get("nodes"), list) else []
        node_count = len(nodes)
        tiers = [
            str(node.get("model_tier", ""))
            for node in nodes
            if isinstance(node, dict) and node.get("model_tier")
        ]

        return {
            "arm_id": exp.get("arm_id"),
            "reward": round(float(exp.get("reward", 0.0)), 6),
            "topology_id": outcome_dict.get("topology_id", topology_dict.get("topology_id")),
            "success": bool(outcome_dict.get("success", False)),
            "latency_seconds": outcome_dict.get("latency_seconds"),
            "total_input_tokens": outcome_dict.get("total_input_tokens"),
            "total_output_tokens": outcome_dict.get("total_output_tokens"),
            "node_count": node_count,
            "tiers": tiers,
        }

    @staticmethod
    def _normalize_bullets(text: str) -> str:
        if not text:
            return ""

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        bullets: list[str] = []

        for line in lines:
            if line.startswith(("-", "*", "•")):
                content = line.lstrip("-*• ").strip()
                if content:
                    bullets.append(f"- {content}")

        if not bullets:
            merged = " ".join(lines)
            chunks = [chunk.strip() for chunk in merged.split(".") if chunk.strip()]
            for chunk in chunks:
                bullets.append(f"- {chunk}")
                if len(bullets) >= 4:
                    break

        return "\n".join(bullets[:4])

    @staticmethod
    def _fallback_insights(high: list[dict[str, Any]], low: list[dict[str, Any]]) -> str:
        high_arms = Counter(str(item.get("arm_id", "unknown")) for item in high)
        low_arms = Counter(str(item.get("arm_id", "unknown")) for item in low)

        high_best = high_arms.most_common(1)[0][0] if high_arms else "unknown"
        low_best = low_arms.most_common(1)[0][0] if low_arms else "unknown"

        high_success = [1.0 if (item.get("outcome_json") or {}).get("success") else 0.0 for item in high]
        low_success = [1.0 if (item.get("outcome_json") or {}).get("success") else 0.0 for item in low]
        high_success_rate = (sum(high_success) / len(high_success)) if high_success else 0.0
        low_success_rate = (sum(low_success) / len(low_success)) if low_success else 0.0

        high_latencies = [
            float((item.get("outcome_json") or {}).get("latency_seconds", 0.0) or 0.0)
            for item in high
        ]
        low_latencies = [
            float((item.get("outcome_json") or {}).get("latency_seconds", 0.0) or 0.0)
            for item in low
        ]
        high_latency = (sum(high_latencies) / len(high_latencies)) if high_latencies else 0.0
        low_latency = (sum(low_latencies) / len(low_latencies)) if low_latencies else 0.0

        bullets = [
            f"- High-reward runs most often used arm {high_best}; low-reward runs clustered around {low_best}.",
            f"- Success rate differs materially: high group {high_success_rate:.0%} vs low group {low_success_rate:.0%}.",
            f"- Average latency trend: high group {high_latency:.1f}s vs low group {low_latency:.1f}s.",
        ]

        if high_best != "unknown":
            bullets.append(
                "- Prefer this dominant high-reward arm as default unless user constraints require lower cost or latency."
            )

        return "\n".join(bullets[:4])
