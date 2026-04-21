# Self-Aware Master Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the DAAP master agent self-aware of its own capabilities, detect tool gaps proactively or on failure, suggest exact install commands, and drop its narrow B2B-only identity in favour of a flexible DAAP platform assistant identity.

**Architecture:** Add `daap/master/capability_registry.py` (new) that maps capability categories to installed tools/MCPs and generates `functional_capabilities` + `known_gaps` structs. Enhance `build_master_runtime_snapshot()` in `runtime.py` to include those structs in the snapshot already injected into the prompt. Rewrite the system prompt in `prompts.py` with a broader identity, hybrid gap-detection rules, and user-friendly self-description instructions.

**Tech Stack:** Python 3.11+, existing AgentScope toolkit, existing DAAP tool registry (`daap/tools/registry.py`), pytest + pytest-asyncio for tests.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `daap/master/capability_registry.py` | **CREATE** | Capability entries, `build_functional_capabilities()`, `build_known_gaps()` |
| `daap/master/runtime.py` | **MODIFY** | Inject `functional_capabilities` + `known_gaps` into snapshot |
| `daap/master/prompts.py` | **MODIFY** | New identity, gap-detection rules, self-description format |
| `daap/tests/test_capability_registry.py` | **CREATE** | Unit tests for registry functions |
| `daap/tests/test_master_agent.py` | **MODIFY** | Add tests for new prompt sections + runtime snapshot keys |

---

## Task 1: Capability Registry — data + core functions

**Files:**
- Create: `daap/master/capability_registry.py`
- Create: `daap/tests/test_capability_registry.py`

### Step 1.1: Write failing tests for `build_functional_capabilities`

- [ ] Create `daap/tests/test_capability_registry.py`:

```python
"""Tests for daap.master.capability_registry."""
import pytest
from daap.master.capability_registry import (
    build_functional_capabilities,
    build_known_gaps,
    CAPABILITY_REGISTRY,
)


def test_builtin_tool_marked_available_when_installed():
    installed = {"WebSearch", "WebFetch", "CodeExecution"}
    caps = build_functional_capabilities(installed)
    web_search = next(c for c in caps if c["label"] == "Web search")
    assert web_search["available"] is True


def test_builtin_tool_marked_unavailable_when_missing():
    installed = set()
    caps = build_functional_capabilities(installed)
    web_search = next(c for c in caps if c["label"] == "Web search")
    assert web_search["available"] is False


def test_mcp_tool_marked_available_by_server_prefix():
    installed = {"WebSearch", "mcp://linkedin/search_people"}
    caps = build_functional_capabilities(installed)
    linkedin = next(c for c in caps if c["label"] == "LinkedIn")
    assert linkedin["available"] is True


def test_mcp_tool_marked_available_by_bare_server_name():
    installed = {"mcp://linkedin"}
    caps = build_functional_capabilities(installed)
    linkedin = next(c for c in caps if c["label"] == "LinkedIn")
    assert linkedin["available"] is True


def test_mcp_tool_marked_unavailable_when_not_installed():
    installed = {"WebSearch", "WebFetch"}
    caps = build_functional_capabilities(installed)
    linkedin = next(c for c in caps if c["label"] == "LinkedIn")
    assert linkedin["available"] is False


def test_all_entries_present_in_functional_capabilities():
    installed = set()
    caps = build_functional_capabilities(installed)
    assert len(caps) == len(CAPABILITY_REGISTRY)


def test_known_gaps_excludes_builtin_tools():
    installed = set()  # even with nothing installed, builtins not in gaps
    gaps = build_known_gaps(installed)
    gap_labels = {g["label"] for g in gaps}
    # Built-in labels should not appear in gaps
    assert "Web search" not in gap_labels
    assert "Web page reading" not in gap_labels
    assert "Code execution" not in gap_labels


def test_known_gaps_includes_missing_mcp_with_install_cmd():
    installed = {"WebSearch"}
    gaps = build_known_gaps(installed)
    linkedin_gap = next((g for g in gaps if g["label"] == "LinkedIn"), None)
    assert linkedin_gap is not None
    assert "install_cmd" in linkedin_gap
    assert "daap mcp add linkedin" in linkedin_gap["install_cmd"]


def test_known_gaps_empty_when_all_mcps_installed():
    installed = {
        "mcp://linkedin",
        "mcp://crunchbase",
        "mcp://gmail",
        "mcp://slack",
        "mcp://github",
        "mcp://hubspot",
    }
    gaps = build_known_gaps(installed)
    assert gaps == []


def test_known_gaps_has_task_keywords():
    installed = set()
    gaps = build_known_gaps(installed)
    for gap in gaps:
        assert "keywords" in gap
        assert len(gap["keywords"]) > 0
```

### Step 1.2: Run tests to verify they fail

- [ ] Run: `pytest daap/tests/test_capability_registry.py -v`
- Expected: `ModuleNotFoundError: No module named 'daap.master.capability_registry'`

### Step 1.3: Implement `capability_registry.py`

- [ ] Create `daap/master/capability_registry.py`:

```python
"""DAAP Capability Registry.

Maps capability categories to installed tools/MCPs and surfaces gaps with
exact install commands. Used by runtime.py to enrich the snapshot injected
into the master agent system prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CapabilityEntry:
    label: str
    task_keywords: list[str]
    builtin_tool: str | None = None
    mcp_server: str | None = None
    install_cmd: str | None = None
    docs_url: str | None = None


CAPABILITY_REGISTRY: list[CapabilityEntry] = [
    CapabilityEntry(
        label="Web search",
        task_keywords=["search", "find", "research", "look up"],
        builtin_tool="WebSearch",
    ),
    CapabilityEntry(
        label="Web page reading",
        task_keywords=["read", "fetch", "scrape", "page content", "website"],
        builtin_tool="WebFetch",
    ),
    CapabilityEntry(
        label="Deep web crawling",
        task_keywords=["crawl", "multi-page", "site crawl", "documentation"],
        builtin_tool="DeepCrawl",
    ),
    CapabilityEntry(
        label="File read/write",
        task_keywords=["read file", "write file", "csv", "save", "load file"],
        builtin_tool="ReadFile",
    ),
    CapabilityEntry(
        label="Code execution",
        task_keywords=["run code", "execute", "compute", "parse data", "script"],
        builtin_tool="CodeExecution",
    ),
    CapabilityEntry(
        label="LinkedIn",
        task_keywords=["linkedin", "people search", "profile", "prospect", "sales navigator"],
        mcp_server="linkedin",
        install_cmd="daap mcp add linkedin npx @daap/linkedin-mcp",
        docs_url="https://mcp.so/server/linkedin",
    ),
    CapabilityEntry(
        label="Crunchbase",
        task_keywords=["crunchbase", "funding", "investors", "startup data", "company data"],
        mcp_server="crunchbase",
        install_cmd="daap mcp add crunchbase npx @daap/crunchbase-mcp",
        docs_url="https://mcp.so/server/crunchbase",
    ),
    CapabilityEntry(
        label="Email sending",
        task_keywords=["send email", "gmail", "smtp", "email outreach", "send outreach"],
        mcp_server="gmail",
        install_cmd="daap mcp add gmail npx @modelcontextprotocol/server-gmail",
        docs_url="https://mcp.so/server/gmail",
    ),
    CapabilityEntry(
        label="Slack",
        task_keywords=["slack", "send message", "notify team", "post to channel"],
        mcp_server="slack",
        install_cmd="daap mcp add slack npx @modelcontextprotocol/server-slack",
        docs_url="https://mcp.so/server/slack",
    ),
    CapabilityEntry(
        label="GitHub",
        task_keywords=["github", "repository", "pull request", "issues", "code review"],
        mcp_server="github",
        install_cmd="daap mcp add github npx @modelcontextprotocol/server-github",
        docs_url="https://mcp.so/server/github",
    ),
    CapabilityEntry(
        label="HubSpot CRM",
        task_keywords=["hubspot", "crm", "contacts", "deals", "pipeline"],
        mcp_server="hubspot",
        install_cmd="daap mcp add hubspot npx @daap/hubspot-mcp",
        docs_url="https://mcp.so/server/hubspot",
    ),
]


def build_functional_capabilities(installed_tool_names: set[str]) -> list[dict]:
    """Return list of {label, available} for all registered capabilities.

    Args:
        installed_tool_names: full set of available tool names (builtins + MCP IDs).
    """
    results = []
    for entry in CAPABILITY_REGISTRY:
        if entry.builtin_tool is not None:
            available = entry.builtin_tool in installed_tool_names
        elif entry.mcp_server is not None:
            prefix = f"mcp://{entry.mcp_server}"
            available = any(
                t == prefix or t.startswith(prefix + "/")
                for t in installed_tool_names
            )
        else:
            available = False
        results.append({"label": entry.label, "available": available})
    return results


def build_known_gaps(installed_tool_names: set[str]) -> list[dict]:
    """Return gaps: MCP capabilities not yet installed, with install info.

    Builtin tools are excluded — they are always present when DAAP runs.
    """
    gaps = []
    for entry in CAPABILITY_REGISTRY:
        if entry.builtin_tool is not None:
            continue
        if entry.mcp_server is None:
            continue
        prefix = f"mcp://{entry.mcp_server}"
        available = any(
            t == prefix or t.startswith(prefix + "/")
            for t in installed_tool_names
        )
        if not available:
            gap: dict = {
                "label": entry.label,
                "keywords": entry.task_keywords,
            }
            if entry.install_cmd:
                gap["install_cmd"] = entry.install_cmd
            elif entry.docs_url:
                gap["docs_url"] = entry.docs_url
            gaps.append(gap)
    return gaps
```

### Step 1.4: Run tests to verify they pass

- [ ] Run: `pytest daap/tests/test_capability_registry.py -v`
- Expected: all 10 tests PASS

### Step 1.5: Commit

- [ ] Run:
```bash
git add daap/master/capability_registry.py daap/tests/test_capability_registry.py
git commit -m "feat(master): add capability registry with gap detection helpers"
```

---

## Task 2: Enhanced Runtime Snapshot

**Files:**
- Modify: `daap/master/runtime.py`
- Modify: `daap/tests/test_master_agent.py` (add 2 tests)

### Step 2.1: Write failing tests

- [ ] Add to end of `daap/tests/test_master_agent.py`:

```python
# ---------------------------------------------------------------------------
# Runtime snapshot capability tests
# ---------------------------------------------------------------------------

def test_runtime_snapshot_includes_functional_capabilities():
    """Runtime snapshot includes functional_capabilities list."""
    from daap.master.runtime import build_master_runtime_snapshot
    from unittest.mock import MagicMock

    toolkit = MagicMock()
    toolkit.tools = {}
    snapshot = build_master_runtime_snapshot(toolkit, execution_mode="script")

    assert "functional_capabilities" in snapshot
    caps = snapshot["functional_capabilities"]
    assert isinstance(caps, list)
    assert len(caps) > 0
    labels = {c["label"] for c in caps}
    assert "Web search" in labels
    assert "LinkedIn" in labels
    for cap in caps:
        assert "label" in cap
        assert "available" in cap
        assert isinstance(cap["available"], bool)


def test_runtime_snapshot_includes_known_gaps():
    """Runtime snapshot includes known_gaps list for uninstalled MCP capabilities."""
    from daap.master.runtime import build_master_runtime_snapshot
    from unittest.mock import MagicMock

    toolkit = MagicMock()
    toolkit.tools = {}
    snapshot = build_master_runtime_snapshot(toolkit, execution_mode="script")

    assert "known_gaps" in snapshot
    gaps = snapshot["known_gaps"]
    assert isinstance(gaps, list)
    # With no MCP servers connected, should have gaps
    gap_labels = {g["label"] for g in gaps}
    assert "LinkedIn" in gap_labels
    for gap in gaps:
        assert "label" in gap
        assert "keywords" in gap
        assert "install_cmd" in gap or "docs_url" in gap
```

### Step 2.2: Run tests to verify they fail

- [ ] Run: `pytest daap/tests/test_master_agent.py::test_runtime_snapshot_includes_functional_capabilities daap/tests/test_master_agent.py::test_runtime_snapshot_includes_known_gaps -v`
- Expected: FAIL — `KeyError: 'functional_capabilities'`

### Step 2.3: Implement snapshot enhancement

- [ ] Replace the body of `build_master_runtime_snapshot` in `daap/master/runtime.py`. Full file after change:

```python
"""Runtime capability snapshot helpers for the master agent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _safe_memory_status() -> dict[str, Any]:
    try:
        from daap.memory.observability import get_memory_status

        return get_memory_status()
    except Exception:
        return {"available": False, "reason": "memory observability unavailable"}


def _safe_connected_mcp_servers() -> list[str]:
    try:
        from daap.mcpx.manager import get_mcp_manager

        manager = get_mcp_manager()
        return manager.list_connected()
    except Exception:
        return []


def _safe_installed_tool_names() -> set[str]:
    try:
        from daap.tools.registry import get_available_tool_names

        return get_available_tool_names(include_mcp_placeholders=False)
    except Exception:
        return set()


def build_master_runtime_snapshot(
    toolkit,
    *,
    execution_mode: str,
    memory_enabled: bool | None = None,
    optimizer_enabled: bool | None = None,
    topology_store_enabled: bool | None = None,
    feedback_store_enabled: bool | None = None,
    session_store_enabled: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, serializable runtime snapshot for prompt grounding."""
    from daap.master.capability_registry import (
        build_functional_capabilities,
        build_known_gaps,
    )

    installed = _safe_installed_tool_names()

    snapshot: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agent_identity": "DAAP Master Agent",
        "agent_role": "orchestrator",
        "execution_mode": execution_mode,
        "master_tools": sorted(list(getattr(toolkit, "tools", {}).keys())),
        "connected_mcp_servers": _safe_connected_mcp_servers(),
        "memory_status": _safe_memory_status(),
        "feature_flags": {},
        "functional_capabilities": build_functional_capabilities(installed),
        "known_gaps": build_known_gaps(installed),
    }

    flags = snapshot["feature_flags"]
    if memory_enabled is not None:
        flags["memory_enabled"] = bool(memory_enabled)
    if optimizer_enabled is not None:
        flags["optimizer_enabled"] = bool(optimizer_enabled)
    if topology_store_enabled is not None:
        flags["topology_store_enabled"] = bool(topology_store_enabled)
    if feedback_store_enabled is not None:
        flags["feedback_store_enabled"] = bool(feedback_store_enabled)
    if session_store_enabled is not None:
        flags["session_store_enabled"] = bool(session_store_enabled)

    if isinstance(extra, dict):
        snapshot.update(extra)

    return snapshot
```

### Step 2.4: Run tests to verify they pass

- [ ] Run: `pytest daap/tests/test_master_agent.py::test_runtime_snapshot_includes_functional_capabilities daap/tests/test_master_agent.py::test_runtime_snapshot_includes_known_gaps -v`
- Expected: both PASS

### Step 2.5: Run full test suite to check no regressions

- [ ] Run: `pytest daap/tests/ -v --tb=short`
- Expected: all previously passing tests still PASS

### Step 2.6: Commit

- [ ] Run:
```bash
git add daap/master/runtime.py daap/tests/test_master_agent.py
git commit -m "feat(master): enrich runtime snapshot with functional_capabilities and known_gaps"
```

---

## Task 3: Rewrite System Prompt

**Files:**
- Modify: `daap/master/prompts.py`
- Modify: `daap/tests/test_master_agent.py` (add 4 tests)

### Step 3.1: Write failing tests

- [ ] Add to end of `daap/tests/test_master_agent.py`:

```python
# ---------------------------------------------------------------------------
# New identity + gap detection prompt tests
# ---------------------------------------------------------------------------

def test_prompt_identity_is_daap_platform_not_b2b_only():
    """Master agent identity is DAAP platform assistant, not narrowly B2B sales only."""
    prompt = get_master_system_prompt()
    # Should identify as DAAP platform assistant
    assert "DAAP" in prompt
    # Should NOT restrict identity to B2B sales automation only
    assert "expert AI assistant for B2B sales automation" not in prompt


def test_prompt_includes_gap_detection_rules():
    """Prompt includes hybrid gap detection: proactive if clear, reactive on failure."""
    prompt = get_master_system_prompt()
    assert "gap" in prompt.lower() or "missing" in prompt.lower() or "install" in prompt.lower()
    # Should instruct agent to surface gaps with install commands
    assert "daap mcp add" in prompt or "install" in prompt.lower()


def test_prompt_instructs_get_runtime_context_for_self_description():
    """Prompt tells agent to call get_runtime_context when asked about capabilities."""
    prompt = get_master_system_prompt()
    assert "get_runtime_context" in prompt
    assert "functional_capabilities" in prompt or "what can you do" in prompt.lower()


def test_prompt_gap_detection_uses_known_gaps_from_snapshot():
    """Prompt instructs agent to use known_gaps from the runtime snapshot for suggestions."""
    prompt = get_master_system_prompt(
        runtime_context={
            "master_tools": ["ask_user", "generate_topology", "get_runtime_context"],
            "functional_capabilities": [
                {"label": "Web search", "available": True},
                {"label": "LinkedIn", "available": False},
            ],
            "known_gaps": [
                {
                    "label": "LinkedIn",
                    "keywords": ["linkedin", "prospect"],
                    "install_cmd": "daap mcp add linkedin npx @daap/linkedin-mcp",
                }
            ],
        }
    )
    assert "known_gaps" in prompt or "LinkedIn" in prompt
```

### Step 3.2: Run tests to verify they fail

- [ ] Run: `pytest daap/tests/test_master_agent.py::test_prompt_identity_is_daap_platform_not_b2b_only daap/tests/test_master_agent.py::test_prompt_includes_gap_detection_rules daap/tests/test_master_agent.py::test_prompt_instructs_get_runtime_context_for_self_description daap/tests/test_master_agent.py::test_prompt_gap_detection_uses_known_gaps_from_snapshot -v`
- Expected: `test_prompt_identity_is_daap_platform_not_b2b_only` FAILS (old identity string still present), others may fail too.

### Step 3.3: Rewrite `prompts.py`

- [ ] Replace `daap/master/prompts.py` with:

```python
"""
DAAP Master Agent system prompts.

The system prompt is injected with:
  - The full TopologySpec JSON schema
  - Available tool names
  - Runtime snapshot (functional_capabilities, known_gaps, connected MCPs)
  - Optional user context from memory
"""

import json
from datetime import datetime

from daap.spec.schema import get_topology_json_schema
from daap.tools.registry import get_available_tool_names, get_tool_descriptions


def get_master_system_prompt(
    available_tools: set[str] | None = None,
    user_context: dict | None = None,
    runtime_context: dict | None = None,
) -> str:
    """Build the master agent system prompt."""
    if available_tools is None:
        available_tools = get_available_tool_names()

    schema = get_topology_json_schema()
    runtime_context = runtime_context or {}

    builtin_tools = {t for t in available_tools if not t.startswith("mcp://")}
    mcp_tools = sorted(t for t in available_tools if t.startswith("mcp://"))
    tools_list = ", ".join(sorted(builtin_tools))

    context_section = ""
    if user_context:
        if isinstance(user_context, str):
            context_section = user_context
        else:
            context_section = f"""
## User Context (from past interactions)
{json.dumps(user_context, indent=2)}
Use this to personalize your responses and any topologies you design.
"""

    mcp_section = ""
    if mcp_tools:
        mcp_lines = "\n".join(f"- `{t}`" for t in mcp_tools)
        mcp_section = f"""
## MCP Tools Available

These tools come from connected MCP servers. Use them in node `tools` arrays with a DAAP MCP tool ID.

Preferred format: `mcp://server_name/tool_name`
Backward-compatible alias: `mcp://server_name` (only when a server exposes exactly one tool).

{mcp_lines}
"""

    runtime_section = ""
    runtime_master_tools = runtime_context.get("master_tools", [])
    known_gaps = runtime_context.get("known_gaps", [])
    functional_caps = runtime_context.get("functional_capabilities", [])

    if runtime_context:
        runtime_section = f"""
## Runtime Snapshot (authoritative — use this as source of truth)
```json
{json.dumps(runtime_context, indent=2)}
```
Treat this snapshot and tool outputs as the source of truth for current capabilities.
"""

    skill_hint = """
## Skills

If the user mentions a file path that looks like a skill directory (for example, `/path/to/skill` or `./my-skill`), call `register_skill` with that path to wire it up immediately. Do not ask the user to run a command.
"""

    execute_flow = """
## Approval and Execution Flow

Call `ask_user` with:
- Question: "Would you like to proceed with this topology?"
- Options: "Yes, execute it", "Make it cheaper", "Cancel"

If user approves (selects execute / says yes / confirms):
- Call `execute_pending_topology` immediately. Do not wait for further input.

If user says make cheaper:
- Call `generate_topology` with a revised lower-cost design, then present the new plan and ask for approval again.

If user cancels:
- Acknowledge and offer to help with something else.

Never tell the user to "type approve" or any keyword. You handle execution directly via `execute_pending_topology`.
"""
    if "execute_pending_topology" not in runtime_master_tools:
        execute_flow = """
## Approval and Execution Flow

Call `ask_user` with:
- Question: "Would you like to proceed with this topology?"
- Options: "Yes, execute it", "Make it cheaper", "Cancel"

If user approves:
- Confirm approval in plain text and provide the final topology summary.
- Do NOT claim you executed anything unless `execute_pending_topology` exists in your runtime tools.
"""

    today = datetime.now().strftime("%Y-%m-%d")

    gap_detection_section = f"""
## Capability Gap Detection

You know what you can and cannot do from the runtime snapshot (`functional_capabilities` and `known_gaps`).

### When to check for gaps

**Proactive** — before attempting: if the user's request clearly requires a capability listed in `known_gaps` (match by keywords), surface the gap immediately. Do not attempt a workaround silently.

**Reactive** — after a tool call fails because a capability is missing: surface the gap and stop.

**Degrade gracefully** — if the task is only partially blocked (some tools available): complete what you can using available tools, then note what was not possible and why.

### How to surface a gap

If the gap has an `install_cmd` in the snapshot:
> "To do this I need **[label]**, which isn't installed yet. Add it with:
> ```
> [install_cmd]
> ```
> Once added, restart DAAP and ask me again."

If the gap has only a `docs_url`:
> "To do this I need **[label]**. I don't have an exact install command — check the docs: [docs_url]"

If the gap is unknown (no matching entry):
> "I don't have a tool for [description of what's needed]. You may be able to add an MCP server for this — check https://mcp.so"

Never make up tool names or claim you executed something using a tool that is not installed.
"""

    self_description_section = """
## Self-Description (when user asks "what can you do?")

Call `get_runtime_context` first. Then format the `functional_capabilities` list into a plain, grouped response:

**Available right now:**
- [label] for each entry where available=true

**Not installed (add to unlock):**
- [label] — `[install_cmd]` for each entry where available=false (if install_cmd exists)

Keep it short and non-technical. Do not explain DAAP internals, topology engine, or architecture. Users just want to know what tasks they can ask for.
"""

    return f"""You are DAAP, an AI assistant that helps you get things done.
Today's date: {today}

You can answer questions directly, run tasks yourself, or design and run multi-agent workflows — whatever fits the job. You know exactly what you can do right now from your runtime snapshot.
{context_section}{runtime_section}{mcp_section}{skill_hint}{gap_detection_section}{self_description_section}
## Core Operating Contract

- You are a conversational, tool-using orchestrator.
- If you need user input, call `ask_user`. Never ask clarification questions in plain text.
- If the task needs a multi-agent workflow, call `generate_topology` with complete JSON.
- Never invent capabilities, infrastructure, integrations, or execution state.
- If asked about runtime capabilities/status, call `get_runtime_context` first before answering.
- Never end a turn with a promise of future action. Each turn must either:
    1) call a tool that makes progress, or
    2) deliver a final direct answer.

## Decision Policy: Direct Answer vs ask_user vs generate_topology

Answer directly when the task is specific, self-contained, and completable in one response:
- One email, one strategy answer, one summary, one brainstorm
- Explaining something, answering a question about DAAP

Use `ask_user` when missing context would materially change the plan or output:
- Target audience, product, or preferences unknown
- Request is ambiguous ("help me with outreach")
- You are presenting a plan and need an explicit user decision

Use `generate_topology` when the task needs multiple specialized agents and at least two of:
- External data gathering
- Distinct stages (research → evaluate → write)
- Parallel processing across many items
- Specialized roles with clear handoffs

Do not call `generate_topology` for simple one-shot tasks.

## Clarification Discipline

- Ask only what you cannot reliably infer or retrieve.
- Keep clarification lightweight: 1-4 questions max.
- Use structured options so users can answer quickly.
- If user context is present, use it to avoid re-asking stable facts.

## Topology Design Rules (When Calling generate_topology)

Available tools for agents: {tools_list}
MCP tools use the format: `mcp://service_name/tool_name` (example: `mcp://linkedin/search_people`).

### Tool Descriptions (what each tool actually does)

{get_tool_descriptions()}

Model tiers — pick the right tier per node (cost shown per 1M tokens):
- "fast":     Gemini 2.5 Flash Lite ($0.10 in / $0.40 out) — search, extract, format, tool-heavy nodes
- "smart":    DeepSeek V3.2 ($0.26 in / $0.38 out) — evaluate, score, rank, write quality output, analysis
- "powerful": Gemini 2.5 Flash ($0.30 in / $2.50 out) — complex synthesis, multi-source reasoning (use sparingly)

Cost rules: default "fast" for tool nodes; "smart" for judgment/writing; "powerful" only when "smart" is insufficient.
A 4-node all-fast topology costs ~$0.002. All-smart ~$0.005. Keep parallel_instances low (2-4 max).

Agent modes:
- "react": iterative tool loop for research/exploration. Typical `max_react_iterations`: 3-5 simple, 8-10 deep.
- "single": one-pass generation with no tool loop.

Design constraints:
- One node = one responsibility.
- Typical topology size: 2-4 nodes. More than 6 is usually over-engineered.
- `parallel_instances > 1` only when work is truly parallelizable, and MUST include consolidation.
- DAG only. No cycles.
- First node has no edge inputs (receives user prompt directly).
- Last node produces final deliverable output.
- `topology_id` format: `topo-` + 8 hex chars, `version` must be 1.
- `agent_mode="react"` REQUIRES at least one tool.
- `agent_mode="single"` should be used for deterministic write/format/summarize steps.

Consolidation strategy guidance:
- "merge": combine independent outputs
- "deduplicate": remove overlap in list-like outputs
- "rank": quality-sort competing outputs
- "vote": majority decision for classification outputs

## Critical: Writing Node system_prompt Quality

Single-mode writing nodes (`agent_mode="single"`) get exactly ONE LLM call.
Their `system_prompt` must force immediate output, not planning language.

Bad writer prompt:
"You are an email writer. Write personalized cold emails for the leads provided."

Good writer prompt:
"You are an email writer. You will receive leads. Immediately output complete cold emails now: subject, body, sign-off. Do not describe your plan. Start with Email 1 immediately."

Rule: writing/formatting prompts must start with output instructions like:
"Output X now", "Write X immediately", "Return X directly".

## Critical: Research Node system_prompt Quality

Research nodes (`agent_mode="react"`) need a concrete multi-step search strategy.

Bad research prompt:
"Find B2B leads for a SaaS company targeting construction companies."

Good research prompt specifies:
1) exact search queries and where to search
2) how to refine search after first pass
3) exact output schema and minimum result count

## Validation and Retry Discipline for generate_topology

If `generate_topology` returns schema/validation/resolution errors:
1. Read every error.
2. Fix ALL errors in the JSON.
3. Call `generate_topology` again in the same conversation.

Do not ask the user to hand-edit topology JSON.
Retry up to 3 times. If still failing, use `ask_user` for targeted missing constraints.

## TopologySpec JSON Schema

Generate JSON that matches this schema exactly:
```json
{json.dumps(schema, indent=2)}
```

## Plan Presentation and Approval Flow

After `generate_topology` succeeds, output a TEXT explanation in the same response turn BEFORE calling `ask_user`. The user cannot see the raw topology JSON — your explanation is the only thing they see.

Your text explanation must include:
1. What each node does (one sentence per node, plain language)
2. How nodes connect (A feeds B, B feeds C)
3. Estimated cost in USD and estimated latency in seconds
4. Total node count

Example format:
```
Here's the plan:
- **reddit_search** (fast model): Searches Reddit for posts matching your criteria using web_search.
- **filter_and_rank** (smart model): Filters results by recency and relevance, extracts post URLs.

2 nodes · ~$0.001 · ~15s
```

Only AFTER this explanation, call `ask_user` with the approval question.

If the user's answer to the approval question is itself a question (e.g. "what does node X do?"):
- Answer their question in plain text
- Then call `ask_user` again with the same approval options

{execute_flow}

## Response Quality Gate

Before finalizing a direct answer or plan summary, check:
- Correctness: did you satisfy every explicit requirement?
- Grounding: are claims based on user context, tool results, or clearly-labeled assumptions?
- Format: did you match requested output format?
- Cost realism: is the topology quality/cost tradeoff appropriate?

## Your Style

- Friendly, clear, and direct
- Non-technical language for explanations — assume the user is not an engineer
- Honest about what you can and cannot do
- Concise and action-oriented
- Always use `ask_user` for clarification questions"""
```

### Step 3.4: Run new prompt tests

- [ ] Run: `pytest daap/tests/test_master_agent.py::test_prompt_identity_is_daap_platform_not_b2b_only daap/tests/test_master_agent.py::test_prompt_includes_gap_detection_rules daap/tests/test_master_agent.py::test_prompt_instructs_get_runtime_context_for_self_description daap/tests/test_master_agent.py::test_prompt_gap_detection_uses_known_gaps_from_snapshot -v`
- Expected: all 4 PASS

### Step 3.5: Run full test suite

- [ ] Run: `pytest daap/tests/ -v --tb=short`
- Expected: all tests PASS (including pre-existing prompt tests)

### Step 3.6: Commit

- [ ] Run:
```bash
git add daap/master/prompts.py daap/tests/test_master_agent.py
git commit -m "feat(master): rewrite identity + add gap detection and self-description to prompt"
```

---

## Task 4: Smoke Test End-to-End

**Files:**
- No file changes — manual verification only

### Step 4.1: Start the CLI and verify self-description

- [ ] Run: `python scripts/chat.py`
- [ ] Type: `what can you do?`
- Expected: agent calls `get_runtime_context`, then responds with grouped list of available capabilities. Does NOT describe architecture or topology engine.

### Step 4.2: Verify gap detection — proactive

- [ ] In same session, type: `find me leads on LinkedIn`
- Expected: agent detects LinkedIn gap from snapshot, responds with:
  > "To do this I need **LinkedIn**, which isn't installed yet. Add it with:
  > `daap mcp add linkedin npx @daap/linkedin-mcp`
  > Once added, restart DAAP and ask me again."
- Does NOT attempt to fake LinkedIn capability using web_search.

### Step 4.3: Verify graceful degradation — partial gap

- [ ] Type: `research companies on Crunchbase and write a summary`
- Expected: agent notes Crunchbase is not installed, offers to do the research part with WebSearch instead (degrade gracefully), asks if that's OK.

### Step 4.4: Verify direct task (no topology)

- [ ] Type: `write me a one-sentence value proposition for a project management SaaS`
- Expected: agent answers directly in one turn. Does NOT call generate_topology.

### Step 4.5: Verify topology still works

- [ ] Type: `find 10 B2B SaaS companies that raised Series A in the last 6 months, research each one, and write a personalized cold email for each`
- Expected: agent asks clarifying questions (your product, ICP), then generates a topology (research + evaluate + write nodes). No gap triggered because WebSearch covers research.

### Step 4.6: Final regression — run full test suite

- [ ] Run: `pytest daap/tests/ -v`
- Expected: all tests PASS

### Step 4.7: Commit

- [ ] Run:
```bash
git add -A
git commit -m "test(master): verify self-aware agent smoke tests pass"
```

---

## Self-Review

**Spec coverage:**
- [x] Self-aware identity (DAAP platform, not B2B only) → Task 3 prompt rewrite
- [x] Direct answer for simple tasks → existing Decision Policy kept + broadened
- [x] Topology for complex tasks → existing logic preserved
- [x] Gap detection: proactive if intent clear → Task 3 gap detection section
- [x] Gap detection: reactive on failure → Task 3 gap detection section
- [x] Degrade gracefully for partial gaps → Task 3 gap detection section
- [x] Exact command when known → `install_cmd` in registry + prompt template
- [x] Docs URL when unknown → `docs_url` fallback in registry + prompt template
- [x] Self-description: surface + functional grouping → Task 3 self-description section
- [x] Self-description: no architecture explanation → prompt explicitly says "non-technical"
- [x] Hybrid gap check → proactive if keywords match, reactive on failure
- [x] Runtime snapshot carries `functional_capabilities` + `known_gaps` → Task 2

**Placeholder scan:** No TBDs or TODOs in any step. All code blocks complete.

**Type consistency:** `build_functional_capabilities` returns `list[dict]` with keys `label`, `available` everywhere. `build_known_gaps` returns `list[dict]` with keys `label`, `keywords`, and `install_cmd` or `docs_url` everywhere. Consistent across registry, runtime, tests.
