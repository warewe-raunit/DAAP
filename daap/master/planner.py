"""
Plan-and-Execute planner for DAAP master agent.

Single LLM call that classifies user intent and produces a step-by-step
execution plan BEFORE the ReAct executor runs. The executor receives a
concrete plan via hint injection instead of deliberating from scratch.

Pattern from beam.ai plan-and-execute: separate planning from execution
to reduce greedy decisions, token waste, and accumulated tool chatter.

Intent classification:
  TOPOLOGY — task needs multi-agent pipeline (2+ of: external data,
             distinct stages, parallel processing, specialized roles)
  ANSWER   — specific, fully-specified, one-shot direct response
  CLARIFY  — vague or missing key information; ask before acting
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

INTENT_TOPOLOGY = "TOPOLOGY"
INTENT_ANSWER   = "ANSWER"
INTENT_CLARIFY  = "CLARIFY"

_VALID_INTENTS = {INTENT_TOPOLOGY, INTENT_ANSWER, INTENT_CLARIFY}

# Hard ceiling on tool calls — TOPOLOGY path max:
#   delegate_to_architect (1) + ask_user (1) + execute (1) + buffer (3) = 6
#   (architect is single-shot via response_format:json_schema — no internal retries)
MAX_PLANNER_ITERS = 12

_PLANNER_PROMPT = """\
You are a routing planner for DAAP, an AI orchestration system.
Given a user message, output a JSON plan. No explanation. No markdown fences.

Schema: {"intent": "TOPOLOGY"|"ANSWER"|"CLARIFY", "steps": [...], "max_iters": <int>}

TOPOLOGY — needs 2+ of: external data, distinct stages, parallel processing, specialized roles.
  steps: ["delegate_to_architect", "ask_user", "execute_pending_topology"], max_iters: 6
ANSWER — specific, fully-specified, one-shot task. steps: [], max_iters: 1
CLARIFY — vague, missing target/format/volume/intent. steps: [], max_iters: 1

Output JSON only."""


@dataclass
class PlanResult:
    intent: str
    steps: list[str] = field(default_factory=list)
    max_iters: int = 6

    def hint(self) -> str:
        """Guidance block for injection into the system prompt (not the user turn)."""
        steps_str = " → ".join(self.steps) if self.steps else "direct response"
        return (
            f"\n\n## Turn Plan\n"
            f"Intent: {self.intent}\n"
            f"Steps: {steps_str}\n"
            f"Iteration budget: {self.max_iters} tool calls max. Follow this plan exactly."
        )

    @classmethod
    def fallback(cls) -> "PlanResult":
        """Safe default when planner LLM call fails."""
        return cls(
            intent=INTENT_CLARIFY,
            steps=[],
            max_iters=1,
        )


async def plan_turn(user_message: str, model, formatter) -> PlanResult:
    """
    Single stateless LLM call → PlanResult. Never raises.

    Uses response_format: json_object (R4) so the planner output is always
    syntactically valid JSON — no freeform string parse failures.
    Output is NOT stored in agent memory — it only travels as a hint.
    """
    import os
    msgs = [
        {"role": "system", "content": _PLANNER_PROMPT},
        {"role": "user", "content": user_message[:2_000]},  # cap: no planner inflation
    ]
    try:
        # Build a constrained model reusing config from the master's model (R4).
        # response_format: json_object guarantees syntactically valid JSON output.
        # Cannot use master's model directly — response_format conflicts with tools.
        from daap.executor.tracked_model import TrackedOpenAIChatModel
        _mn = getattr(model, "model_name", "qwen/qwen3.6-plus")
        _ak = getattr(model, "api_key", os.environ.get("OPENROUTER_API_KEY", ""))
        _ck = getattr(model, "client_kwargs", {"base_url": "https://openrouter.ai/api/v1"})
        constrained = TrackedOpenAIChatModel(
            model_name=_mn,
            api_key=_ak,
            client_kwargs=_ck,
            stream=False,
            generate_kwargs={
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
        )

        formatted = formatter.format_messages(msgs)
        response = await constrained(formatted)
        text = (getattr(response, "content", None) or str(response)).strip()

        # Strip markdown fences if the model added them despite response_format
        if text.startswith("```"):
            inner = text.split("\n", 1)[-1]
            text = inner.rsplit("```", 1)[0].strip()

        data = json.loads(text)

        intent = data.get("intent", INTENT_CLARIFY)
        if intent not in _VALID_INTENTS:
            intent = INTENT_CLARIFY

        steps = data.get("steps", [])
        if not isinstance(steps, list):
            steps = []

        # Hard cap: min 1, max MAX_PLANNER_ITERS
        max_iters = max(1, min(int(data.get("max_iters", 6)), MAX_PLANNER_ITERS))

        return PlanResult(intent=intent, steps=steps, max_iters=max_iters)

    except Exception as exc:
        logger.debug("plan_turn fallback (planner error): %s", exc)
        return PlanResult.fallback()
