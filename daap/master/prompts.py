"""
DAAP Master Agent system prompt.

Patterns adopted from claw-code (https://github.com/instructkr/claw-code):
  1. Cache boundary marker — explicit separator between static (cacheable) and
     dynamic (session-specific) content, enabling prompt cache reuse.
  2. Hard section budgets — injected content capped at MAX_SECTION_CHARS with
     priority-based line trimming (summary > headers > bullets > other).
  3. Priority-trim for user context — preserves most important lines first.

Previously adopted (Claude Code / Anthropic research):
  - Critical rules FIRST (high attention region)
  - Compact runtime text (not JSON dump)
  - Conditional section injection
  - Volatile data after static boundary
"""

import json
from datetime import datetime

from daap.tools.registry import get_available_tool_names

# ---------------------------------------------------------------------------
# Section budget constants (from claw-code: 4K per section, 12K total)
# ---------------------------------------------------------------------------

MAX_USER_CONTEXT_CHARS = 4_000
MAX_RUNTIME_CHARS = 1_200   # compact text, already small; matches claw-code summary budget
MAX_LINE_CHARS = 160        # claw-code max chars per line


# ---------------------------------------------------------------------------
# Priority-based line trimmer (claw-code pattern)
# ---------------------------------------------------------------------------

def _priority_trim(text: str, max_chars: int, max_lines: int = 32) -> str:
    """
    Select highest-priority lines within budget.

    Priority tiers (claw-code):
      0 — summary/scope/task headers  (most important)
      1 — section headers ending in ':'
      2 — bullet points
      3 — everything else

    Returns original text if within budget. Appends omission notice if trimmed.
    """
    if len(text) <= max_chars:
        return text

    raw_lines = [l.rstrip() for l in text.splitlines()]

    # Deduplicate (case-insensitive) while preserving order
    seen: set[str] = set()
    lines: list[str] = []
    for l in raw_lines:
        key = l.strip().lower()
        if key and key not in seen:
            seen.add(key)
            lines.append(l[:MAX_LINE_CHARS])

    def _priority(line: str) -> int:
        s = line.strip().lower()
        if any(kw in s for kw in ("summary:", "scope:", "current work:", "task:", "status:")):
            return 0
        if s.endswith(":") and len(s) < 80:
            return 1
        if line.lstrip().startswith(("- ", "• ", "* ", "1.", "2.", "3.")):
            return 2
        return 3

    # Sort by (priority tier, original position) — stable within tier
    ranked = sorted(range(len(lines)), key=lambda i: (_priority(lines[i]), i))

    selected: list[int] = []
    chars_used = 0
    for idx in ranked:
        line = lines[idx]
        if chars_used + len(line) + 1 > max_chars or len(selected) >= max_lines:
            break
        selected.append(idx)
        chars_used += len(line) + 1  # +1 for newline

    omitted = len(lines) - len(selected)
    # Restore original order for readability
    selected_sorted = sorted(selected)
    result = "\n".join(lines[i] for i in selected_sorted)
    if omitted > 0:
        result += f"\n[… {omitted} line(s) omitted]"
    return result


# ---------------------------------------------------------------------------
# Runtime compactor
# ---------------------------------------------------------------------------

def _compact_runtime(runtime_context: dict) -> str:
    """Compact text summary of runtime snapshot — replaces full JSON dump."""
    if not runtime_context:
        return ""

    lines: list[str] = []

    exec_mode = runtime_context.get("execution_mode")
    if exec_mode:
        lines.append(f"execution_mode: {exec_mode}")

    flags = {k: v for k, v in runtime_context.get("feature_flags", {}).items() if v}
    if flags:
        lines.append(f"active_features: {', '.join(flags)}")

    caps = runtime_context.get("functional_capabilities", [])
    available_caps = [c["label"] for c in caps if c.get("available")]
    if available_caps:
        lines.append(f"capabilities: {', '.join(available_caps)}")

    gaps = runtime_context.get("known_gaps", [])
    if gaps:
        parts = []
        for g in gaps:
            cmd = g.get("install_cmd") or g.get("docs_url", "")
            parts.append(f"{g['label']} → `{cmd}`" if cmd else g["label"])
        lines.append(f"not_installed: {' | '.join(parts)}")

    servers = runtime_context.get("connected_mcp_servers", [])
    if servers:
        lines.append(f"mcp_servers: {', '.join(servers)}")

    master_tools = runtime_context.get("master_tools", [])
    if master_tools:
        lines.append(f"master_tools: {', '.join(master_tools)}")

    raw = "\n".join(lines)
    return _priority_trim(raw, MAX_RUNTIME_CHARS)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def get_master_system_prompt(
    available_tools: set[str] | None = None,
    user_context: dict | None = None,
    runtime_context: dict | None = None,
) -> str:
    """Build the master agent system prompt."""
    if available_tools is None:
        available_tools = get_available_tool_names()

    runtime_context = runtime_context or {}
    today = datetime.now().strftime("%Y-%m-%d")

    mcp_tools = sorted(t for t in available_tools if t.startswith("mcp://"))

    runtime_master_tools = runtime_context.get("master_tools", [])

    # --- Execute flow: conditional on tool availability ---
    if "execute_pending_topology" in runtime_master_tools:
        execute_flow = (
            "If approved → DAAP calls `execute_pending_topology` immediately.\n"
            "If cheaper → DAAP re-delegates to `delegate_to_architect` with the feedback, re-presents, re-asks.\n"
            "If cancel → DAAP acknowledges."
        )
    else:
        execute_flow = (
            "If approved → DAAP confirms in plain text with topology summary.\n"
            "DAAP does NOT claim execution — `execute_pending_topology` is not available."
        )

    # --- Dynamic sections (after static boundary) ---
    compact_rt = _compact_runtime(runtime_context)
    runtime_section = f"\n## Runtime\n{compact_rt}\n" if compact_rt else ""

    mcp_section = ""
    if mcp_tools:
        mcp_section = "\n**MCP tools:** " + "  ".join(f"`{t}`" for t in mcp_tools) + "\n"

    context_section = ""
    if user_context:
        body = user_context if isinstance(user_context, str) else json.dumps(user_context, indent=2)
        body = _priority_trim(body, MAX_USER_CONTEXT_CHARS)
        context_section = f"\n## User Context\n{body}\n"

    topo_store_enabled = runtime_context.get("feature_flags", {}).get("topology_store_enabled", True)
    saved_topos_section = ""
    if topo_store_enabled:
        saved_topos_section = (
            "\n## Saved Topologies\n"
            "DAAP calls `list_saved_topologies` → matches description → calls `rerun_topology(topology_id=<exact_id>)`.\n"
            "DAAP never guesses or constructs an ID. DAAP proactively offers when user asks 'what have I done?' or 'rerun'.\n"
        )

    # ---------------------------------------------------------------------------
    # STATIC SECTION — everything above this boundary is session-invariant and
    # eligible for prompt cache reuse. Dynamic content follows the boundary.
    # ---------------------------------------------------------------------------
    static_block = f"""DAAP is an AI System Architect and multi-agent orchestrator. Today: {today}.

## Core Rules

- DAAP clarifies in plain text when context is missing; DAAP stops the turn and waits for reply.
- Each turn does one of: (1) calls a tool, (2) asks a clarifying question, (3) delivers a final answer.
- DAAP calls `get_runtime_context` before answering capability or status questions.
- If the user provides a skill path, DAAP calls `register_skill` immediately.
- DAAP surfaces capability gaps with install_cmd when a required tool is missing. DAAP never fabricates tools.

## Decision: Ask vs Answer vs Topology

**Clarify** — vague request, missing target/format/volume, ambiguous intent. 1–3 questions max; user context avoids re-asking.
**Answer directly** — specific, fully-specified, one-shot task.
**delegate_to_architect** — task needs 2+ of: external data · distinct stages · parallel processing · specialized roles. DAAP gathers all requirements from the user first, then calls `delegate_to_architect`. DAAP does not write the topology JSON itself.
**ask_user** — structured option picker only (topology approval: Yes / Cheaper / Cancel).

## Plan Presentation & Approval

After `delegate_to_architect` returns, DAAP outputs a TEXT explanation then calls `ask_user`:
- One sentence per node · connection flow (A→B→C) · cost + latency + node count

Example:
```
- **search** (fast): Finds Reddit posts matching criteria.
- **rank** (smart): Filters by recency, extracts URLs.
2 nodes · ~$0.001 · ~15s
```

`ask_user`: "Proceed?", options: "Yes, execute", "Make cheaper", "Cancel".
{execute_flow}

If user's approval response is a question: DAAP answers it, then re-calls `ask_user` with same options."""

    # ---------------------------------------------------------------------------
    # DYNAMIC SECTION — session-specific, changes per call, not cached
    # ---------------------------------------------------------------------------
    dynamic_block = f"{saved_topos_section}{runtime_section}{mcp_section}{context_section}"

    suffix = "\nStyle: friendly · direct · non-technical · concise · honest about limitations."

    return static_block + "\n" + dynamic_block + suffix
