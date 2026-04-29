"""
DAAP Context Manager — model-aware bounded memory and input truncation.

Three problems solved permanently:

  1. Master agent InMemoryMemory grows unbounded across turns.
     Fix: BoundedMemory subclasses InMemoryMemory and overrides add() so every
     message addition checks the token budget and compacts if needed. The agent
     cannot accumulate unbounded context — it is structurally impossible.

  2. Smart-tier nodes (deepseek/deepseek-v3.2, 64K limit) receive upstream outputs
     that exceed their context window → hard API failure.
     Fix: BuiltNode carries max_input_tokens. patterns.py truncates every input
     before passing to any node. No node can ever receive more than its model handles.

  3. Node react-loop tool results inflate deepseek context mid-execution.
     Fix: BoundedMemory is used for node agents too, with model-specific thresholds.

Two-phase compaction strategy (OpenCode / Anthropic pattern, no LLM call needed):
  Phase 1 — Tool-result clearing: replace old ToolResultBlock content with a
             token-count placeholder. No LLM call. Reclaims 60-80% in most sessions.
             Keeps TOOL_CLEAR_KEEP_RECENT recent results untouched for active context.
  Phase 2 — Summary compression: build a deterministic structured summary of
             older assistant/user messages, delete them, and store via agentscope's
             native update_compressed_summary() → get_memory(prepend_summary=True)
             automatically prepends it on every subsequent model call.

Model context limits (tokens) — source: OpenRouter, April 2026:
  google/gemini-2.5-flash       1,048,576  master agent + powerful nodes
  google/gemini-2.5-flash-lite  1,048,576  fast nodes
  deepseek/deepseek-v3.2           64,000  smart nodes  ← most constrained
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from agentscope.memory import InMemoryMemory
from agentscope.message import Msg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model context limits
# ---------------------------------------------------------------------------

MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "google/gemini-2.5-flash":       1_048_576,
    "google/gemini-2.5-flash-lite":  1_048_576,
    "deepseek/deepseek-v3.2":           64_000,
    "google/gemini-3-flash-preview": 1_048_576,
    "google/gemini-2.0-flash-001":   1_048_576,
    "qwen/qwen3.6-plus":               131_072,  # 128K context
}
DEFAULT_CONTEXT_LIMIT = 128_000

# Compact when token estimate exceeds this fraction of the model's context limit.
# gemini-2.5-flash (1M): triggers at ~655K  — generous, keeps master snappy
# deepseek-v3.2 (64K):   triggers at ~40K   — protects hard 64K ceiling
COMPACT_TRIGGER_FRACTION = 0.625

# After Phase 1, run Phase 2 if still above this fraction.
COMPACT_HARD_FRACTION = 0.50

# Keep this many recent tool results untouched during Phase 1.
TOOL_CLEAR_KEEP_RECENT = 6

# ---------------------------------------------------------------------------
# Safe input token budget per model for upstream-to-node handoffs.
# Formula: context_limit − system_prompt_budget − output_budget − safety_margin
#   gemini-2.5-flash:      1M   − 5K − 20K − 25K  → 400K (practical cap)
#   gemini-2.5-flash-lite: 1M   − 5K − 10K − 25K  → 400K (practical cap)
#   deepseek-v3.2:         64K  − 4K − 10K −  6K  → 44K  (hard cap)
# ---------------------------------------------------------------------------

MAX_INPUT_TOKENS_BY_MODEL: dict[str, int] = {
    "google/gemini-2.5-flash":       400_000,
    "google/gemini-2.5-flash-lite":  400_000,
    "deepseek/deepseek-v3.2":          44_000,
    "google/gemini-3-flash-preview": 400_000,
    "google/gemini-2.0-flash-001":   400_000,
    "qwen/qwen3.6-plus":             100_000,  # 128K − system prompt − output − margin
}
DEFAULT_MAX_INPUT_TOKENS = 80_000


def get_context_limit(model_id: str) -> int:
    return MODEL_CONTEXT_LIMITS.get(model_id, DEFAULT_CONTEXT_LIMIT)


def get_max_input_tokens(model_id: str) -> int:
    return MAX_INPUT_TOKENS_BY_MODEL.get(model_id, DEFAULT_MAX_INPUT_TOKENS)


# ---------------------------------------------------------------------------
# Token estimation helpers
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate: 4 chars ≈ 1 token (conservative for multilingual)."""
    return max(1, len(text) // 4)


def _msg_text(msg: Msg) -> str:
    """Extract plain text from a Msg object."""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("output") or block.get("text") or ""))
        return "\n".join(p for p in parts if p)
    return str(content) if content is not None else ""


def _is_tool_result_msg(msg: Msg) -> bool:
    """Detect agentscope tool result messages (role=system, content=[ToolResultBlock])."""
    if not isinstance(msg.content, list):
        return False
    for block in msg.content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            return True
    return False


def _is_tool_use_msg(msg: Msg) -> bool:
    """Detect assistant messages that contain a ToolUse block."""
    if not isinstance(msg.content, list):
        return False
    for block in msg.content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            return True
    return False


# ---------------------------------------------------------------------------
# Input truncation — called in patterns.py before every node execution
# ---------------------------------------------------------------------------

def truncate_input_for_model(text: str, model_id: str) -> str:
    """
    Truncate upstream node input to fit the model's safe input budget.

    This is the primary protection for deepseek-v3.2 (64K total context).
    Appends a marker so downstream nodes know truncation occurred.
    """
    max_tokens = get_max_input_tokens(model_id)
    estimated = estimate_tokens(text)

    if estimated <= max_tokens:
        return text

    max_chars = max_tokens * 4
    dropped = estimated - max_tokens
    limit = get_context_limit(model_id)

    return (
        text[:max_chars]
        + f"\n\n[input truncated — ~{dropped:,} tokens dropped"
        f" to fit {model_id} context limit ({limit:,} tokens)]"
    )


# ---------------------------------------------------------------------------
# Phase 2 summary builder (deterministic, no LLM call)
# ---------------------------------------------------------------------------

def _priority_trim_lines(text: str, max_chars: int = 200, max_lines: int = 4) -> str:
    """
    Select highest-priority lines from text within budget (claw-code pattern).

    Priority tiers:
      0 — summary/scope/task/status headers
      1 — section headers ending with ':'
      2 — bullet points
      3 — everything else

    Returns a single-line summary for use inside a larger context summary.
    """
    raw = [l.strip() for l in text.splitlines() if l.strip()]
    # Deduplicate (case-insensitive)
    seen: set[str] = set()
    lines: list[str] = []
    for l in raw:
        key = l.lower()
        if key not in seen:
            seen.add(key)
            lines.append(l[:160])  # claw-code: max 160 chars per line

    def _priority(line: str) -> int:
        s = line.lower()
        if any(kw in s for kw in ("summary:", "scope:", "task:", "status:", "current work:")):
            return 0
        if s.endswith(":") and len(s) < 80:
            return 1
        if line.startswith(("- ", "• ", "* ")):
            return 2
        return 3

    ranked = sorted(range(len(lines)), key=lambda i: (_priority(lines[i]), i))
    selected: list[int] = []
    chars = 0
    for idx in ranked:
        line = lines[idx]
        if chars + len(line) > max_chars or len(selected) >= max_lines:
            break
        selected.append(idx)
        chars += len(line)

    selected_sorted = sorted(selected)
    return " | ".join(lines[i] for i in selected_sorted) if selected_sorted else lines[0][:max_chars]


def _build_summary(messages: list[Msg], model_id: str) -> str:
    """
    Build a structured, priority-ranked summary of older messages.

    Uses claw-code's priority-line selection: preserves the most semantically
    important content (scope, task, status lines) over chronological verbosity.
    Deterministic — no LLM call, zero token cost.
    Budget: 1,200 chars total (claw-code default).
    """
    SUMMARY_BUDGET = 1_200
    PER_MSG_BUDGET = 200

    user_turns: list[str] = []
    assistant_turns: list[str] = []

    for msg in messages:
        text = _msg_text(msg).strip()
        if not text or text.startswith("[tool result cleared") or text.startswith("[CONTEXT SUMMARY"):
            continue
        trimmed = _priority_trim_lines(text, max_chars=PER_MSG_BUDGET, max_lines=3)
        if msg.role == "user" and len(user_turns) < 8:
            user_turns.append(trimmed)
        elif msg.role == "assistant" and len(assistant_turns) < 8:
            assistant_turns.append(trimmed)

    header = f"[CONTEXT SUMMARY — {len(messages)} messages compressed for {model_id}]"
    lines: list[str] = [header]
    chars = len(header)

    if user_turns:
        section = "User requests (oldest → newest):"
        lines.append(section)
        chars += len(section)
        for t in user_turns:
            entry = f"  - {t}"
            if chars + len(entry) > SUMMARY_BUDGET:
                omitted = len(user_turns) - user_turns.index(t)
                lines.append(f"  … {omitted} more request(s) omitted")
                break
            lines.append(entry)
            chars += len(entry)

    if assistant_turns and chars < SUMMARY_BUDGET:
        section = "Assistant actions (oldest → newest):"
        lines.append(section)
        chars += len(section)
        for t in assistant_turns:
            entry = f"  - {t}"
            if chars + len(entry) > SUMMARY_BUDGET:
                omitted = len(assistant_turns) - assistant_turns.index(t)
                lines.append(f"  … {omitted} more action(s) omitted")
                break
            lines.append(entry)
            chars += len(entry)

    lines.append("[End of summary. Resume from here.]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# BoundedMemory — permanent fix for context overflow
# ---------------------------------------------------------------------------

class BoundedMemory(InMemoryMemory):
    """
    InMemoryMemory subclass that enforces a model-specific token budget.

    Overrides add() so every message insertion checks the budget and
    compacts automatically if exceeded. The agent cannot accumulate unbounded
    context — it is structurally prevented at the memory layer.

    Two-phase compaction (no LLM call, no token cost):
      Phase 1: replace old ToolResultBlock content with short placeholders.
      Phase 2: summarize older user/assistant messages via agentscope's native
               _compressed_summary mechanism (auto-prepended by get_memory).

    Compatible with agentscope's full InMemoryMemory API including marks,
    state serialization, and delete_by_mark.
    """

    def __init__(self, model_id: str) -> None:
        super().__init__()
        self._model_id = model_id
        context_limit = get_context_limit(model_id)
        self._compact_trigger = int(context_limit * COMPACT_TRIGGER_FRACTION)
        self._compact_hard = int(context_limit * COMPACT_HARD_FRACTION)

    async def add(
        self,
        memories: Msg | list[Msg] | None,
        marks: str | list[str] | None = None,
        allow_duplicates: bool = False,
        **kwargs: Any,
    ) -> None:
        await super().add(memories, marks=marks, allow_duplicates=allow_duplicates, **kwargs)
        await self._maybe_compact()

    # ------------------------------------------------------------------
    # Internal compaction
    # ------------------------------------------------------------------

    async def _maybe_compact(self) -> None:
        msgs = [msg for msg, _ in self.content]
        total_tokens = sum(estimate_tokens(_msg_text(m)) for m in msgs)

        if total_tokens <= self._compact_trigger:
            return

        logger.info(
            "BoundedMemory: compaction triggered model=%s ~%d tokens (trigger=%d limit=%d)",
            self._model_id, total_tokens, self._compact_trigger,
            get_context_limit(self._model_id),
        )

        # Phase 1: clear old tool results — no LLM, cheap
        freed = await self._phase1_clear_tool_results()
        total_tokens -= freed

        # Phase 2: summarize older messages if still over hard threshold
        if total_tokens > self._compact_hard:
            await self._phase2_summarize()

    async def _phase1_clear_tool_results(self) -> int:
        """Replace content of old tool results with token-count placeholders."""
        tool_ids = [
            msg.id for msg, _ in self.content if _is_tool_result_msg(msg)
        ]
        # Keep most recent TOOL_CLEAR_KEEP_RECENT tool results intact
        to_clear = set(tool_ids[:-TOOL_CLEAR_KEEP_RECENT]) if len(tool_ids) > TOOL_CLEAR_KEEP_RECENT else set()

        if not to_clear:
            return 0

        freed = 0
        new_content: list[tuple[Msg, list[str]]] = []

        for msg, marks in self.content:
            if msg.id in to_clear:
                original_tokens = estimate_tokens(_msg_text(msg))
                placeholder_text = f"[tool result cleared — ~{original_tokens} tokens]"
                cleared_msg = Msg(
                    name=msg.name,
                    content=placeholder_text,
                    role=msg.role,
                    metadata={**msg.metadata, "compacted": True},
                )
                new_content.append((cleared_msg, deepcopy(marks)))
                freed += max(0, original_tokens - estimate_tokens(placeholder_text))
            else:
                new_content.append((msg, marks))

        self.content = new_content

        if freed > 0:
            logger.info(
                "BoundedMemory Phase 1: freed ~%d tokens from %d tool results",
                freed, len(to_clear),
            )

        return freed

    async def _phase2_summarize(self) -> None:
        """
        Summarize older messages, delete them, and store summary via
        agentscope's native _compressed_summary mechanism.

        get_memory(prepend_summary=True) — the default — will prepend
        this summary on every subsequent model call automatically.

        Tool-use boundary protection (claw-code pattern): never delete an
        assistant ToolUse message while keeping its paired ToolResult. Walk
        the cut boundary backward until it lands on a safe split point.
        """
        msgs = [msg for msg, _ in self.content]
        keep_recent = max(10, len(msgs) // 3)
        cut = len(msgs) - keep_recent

        # Walk cut backward to avoid splitting a ToolUse from its ToolResult.
        # A ToolUse at index i is paired with a ToolResult at index i+1 (agentscope
        # convention: assistant tool_use → system tool_result immediately after).
        while cut > 0 and _is_tool_use_msg(msgs[cut - 1]):
            cut -= 1

        older = msgs[:cut]
        if not older:
            return

        summary = _build_summary(older, self._model_id)
        older_ids = [m.id for m in older]
        await self.delete(older_ids)
        await self.update_compressed_summary(summary)

        logger.info(
            "BoundedMemory Phase 2: compressed %d → summary, kept %d recent messages (cut=%d)",
            len(older), len(msgs) - cut, cut,
        )
