"""
DAAP Master Agent Tools — delegate_to_architect and ask_user.

Both registered via AgentScope Toolkit. The agent calls these instead of
producing plain text — the API layer uses tool calls as reliable signals,
not fragile heuristics (no question-mark detection, no guessing).

State model:
  delegate_to_architect → returns topology+estimate via ToolResponse.metadata
                          and stores in per-toolkit-instance slot
  ask_user state         — per-toolkit-instance closures (see create_master_toolkit)
"""

import asyncio
import json
import logging
import os
import re as _re
import uuid
from datetime import datetime, timezone

from agentscope.message import TextBlock
from agentscope.tool import Toolkit, ToolResponse

logger = logging.getLogger(__name__)

from daap.master.runtime import build_master_runtime_snapshot
from daap.spec.estimator import estimate_topology, _format_cost
from daap.spec.resolver import resolve_topology
from daap.spec.schema import TopologySpec
from daap.spec.validator import validate_topology
from daap.skills.manager import (
    SkillValidationError,
    apply_configured_skills,
    get_skill_manager,
)
from daap.tools.registry import get_available_tool_names

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MASTER_MODEL = "qwen/qwen3.6-plus"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _content_to_text(content: object) -> str:
    """Extract plain text from AgentScope/OpenAI-ish message content."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "content" in block:
                    parts.append(str(block.get("content", "")))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def _strip_markdown_fence(text: str) -> str:
    """Strip one surrounding markdown code fence if present."""
    text = text.strip()
    if text.startswith("```"):
        text = _re.sub(r"^```(?:json)?\s*", "", text)
        text = _re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_topology_json(text: str) -> dict | None:
    """Parse a topology JSON object from raw or lightly wrapped model text."""
    text = _strip_markdown_fence(text)
    decoder = json.JSONDecoder()

    candidates = [0]
    candidates.extend(i for i, ch in enumerate(text) if ch == "{")
    seen: set[int] = set()
    for start in candidates:
        if start in seen:
            continue
        seen.add(start)
        try:
            obj, _end = decoder.raw_decode(text[start:].lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "nodes" in obj and "edges" in obj:
            return obj

    try:
        from json_repair import repair_json
        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, dict) and "nodes" in repaired and "edges" in repaired:
            return repaired
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Constrained-decoding repair (R4)
# ---------------------------------------------------------------------------

async def _repair_topology_json(broken_json: str) -> dict | None:
    """
    Attempt to repair malformed topology JSON using constrained decoding.

    Tries json_schema mode first (schema-enforced output), falls back to
    json_object mode (valid JSON guaranteed, schema checked by Pydantic).
    Returns parsed dict on success, None if all attempts fail.

    This repair step drops JSON parse failures from ~38% to near-zero
    (https://openreview.net/pdf?id=FKOaJqKoio) by constraining the model's
    token sampling to valid JSON token sequences.
    """
    from agentscope.formatter import OpenAIChatFormatter
    from daap.executor.tracked_model import TrackedOpenAIChatModel
    from daap.spec.schema import get_topology_json_schema

    schema = get_topology_json_schema()
    prompt = (
        "The following topology JSON is malformed or incomplete. "
        "Return a corrected, complete TopologySpec JSON object. "
        "Output ONLY valid JSON — no markdown, no explanation.\n\n"
        f"{broken_json[:8_000]}"
    )

    for response_format in [
        {
            "type": "json_schema",
            "json_schema": {"name": "TopologySpec", "strict": False, "schema": schema},
        },
        {"type": "json_object"},
    ]:
        try:
            model = TrackedOpenAIChatModel(
                model_name=OPENROUTER_MASTER_MODEL,
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
                client_kwargs={"base_url": OPENROUTER_BASE_URL},
                stream=False,
                generate_kwargs={"temperature": 0.3, "response_format": response_format},
            )
            formatter = OpenAIChatFormatter()
            msgs = [{"role": "user", "content": prompt}]
            response = await model(formatter.format_messages(msgs))
            text = _content_to_text(getattr(response, "content", None) or str(response))
            repaired = _extract_topology_json(text)
            if repaired is not None:
                return repaired
            return json.loads(_strip_markdown_fence(text))
        except Exception as exc:
            logger.debug("_repair_topology_json attempt failed (%s): %s", response_format["type"], exc)
            continue
    return None


# ---------------------------------------------------------------------------
# Tool: delegate_to_architect
# ---------------------------------------------------------------------------

async def _delegate_to_architect_impl(
    current_topology_json: str,
    user_feedback: str,
    *,
    user_context: object = None,
    runtime_context: dict | None = None,
    operator_config: dict | None = None,
) -> ToolResponse:
    """Internal implementation. Accepts grounding context invisible to the LLM.

    user_context / runtime_context are forwarded to generate_topology so the
    architect designs against today's date, connected MCP servers, known
    capability gaps, and the user's domain — not a comma-separated tool list.
    """
    from daap.master.topology_agent import generate_topology

    available_tools = get_available_tool_names()
    user_msg = (
        f"Current Topology: {current_topology_json}\n\n"
        f"Requested Changes: {user_feedback}"
    )

    # Single-shot structured generation — response_format:json_schema makes
    # the model emit a TopologySpec-shaped JSON object on the first attempt.
    try:
        topology_dict, raw_text = await generate_topology(
            user_msg,
            available_tools,
            operator_config=operator_config,
            user_context=user_context,
            runtime_context=runtime_context,
        )
    except Exception as e:
        logger.exception("Topology Architect generation failed")
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Architect failed: {e}. Retry or simplify the request.",
        )])

    # Fallback: constrained-decoding repair for the rare case where the
    # provider returned wrapped or partial output despite json_schema mode.
    if topology_dict is None:
        logger.warning("Architect returned non-parseable topology output: %r", raw_text[:1000])
        topology_dict = await _repair_topology_json(raw_text)
        if topology_dict is None:
            return ToolResponse(content=[TextBlock(
                type="text",
                text=(
                    "Architect returned malformed JSON that could not be repaired. "
                    f"Raw architect output started with: {raw_text[:300]!r}"
                ),
            )])

    # Sanitize topology_id — always regenerate if missing or non-canonical
    _TOPO_ID_RE = _re.compile(r"^topo-[0-9a-f]{8}$")
    if not _TOPO_ID_RE.match(topology_dict.get("topology_id", "")):
        topology_dict["topology_id"] = "topo-" + uuid.uuid4().hex[:8]
    if "created_at" not in topology_dict or not topology_dict["created_at"]:
        topology_dict["created_at"] = datetime.now(timezone.utc).isoformat()

    submitted_json = json.dumps(topology_dict, indent=2)

    # Pydantic parse
    try:
        topology = TopologySpec.model_validate(topology_dict)
    except Exception as e:
        return ToolResponse(content=[TextBlock(
            type="text",
            text=(
                f"Topology schema validation failed:\n{e}\n\n"
                f"Submitted JSON:\n```json\n{submitted_json}\n```"
            ),
        )])

    # Validate
    available_models = {"fast", "smart", "powerful"}
    validation = validate_topology(topology, available_tools, available_models)
    if not validation.is_valid:
        return ToolResponse(content=[TextBlock(
            type="text",
            text=(
                f"Topology validation failed:\n{validation.error_summary}\n\n"
                f"Submitted JSON:\n```json\n{submitted_json}\n```"
            ),
        )])

    # Resolve
    resolved = resolve_topology(topology)
    if isinstance(resolved, list):
        error_text = "\n".join(f"- {e.message}" for e in resolved)
        return ToolResponse(content=[TextBlock(
            type="text",
            text=(
                f"Topology resolution failed:\n{error_text}\n\n"
                f"Submitted JSON:\n```json\n{submitted_json}\n```"
            ),
        )])

    # Estimate
    estimate = estimate_topology(resolved)
    estimate_data = {
        "total_cost_usd": estimate.total_cost_usd,
        "total_latency_seconds": estimate.total_latency_seconds,
        "min_viable_cost_usd": estimate.min_viable_cost_usd,
        "within_budget": estimate.within_budget,
        "within_timeout": estimate.within_timeout,
    }

    return ToolResponse(
        content=[TextBlock(
            type="text",
            text=(
                f"Topology is valid and ready for execution.\n\n"
                f"{estimate.user_facing_summary}\n\n"
                f"Estimated cost: {_format_cost(estimate.total_cost_usd)}\n"
                f"Estimated time: ~{estimate.total_latency_seconds:.0f} seconds\n"
                f"Minimum viable cost: {_format_cost(estimate.min_viable_cost_usd)}\n\n"
                f"Present this plan and cost estimate to the user. "
                f"Ask if they want to proceed, want it cheaper, or want changes."
            ),
        )],
        metadata={"topology": topology_dict, "estimate": estimate_data},
    )


async def delegate_to_architect(
    current_topology_json: str,
    user_feedback: str,
) -> ToolResponse:
    """Design or modify a multi-agent execution topology via the Topology Architect.

    Call this when the task needs multiple specialized agents. NOT for one-shot tasks.
    Gather all requirements from the user before calling — DAAP does not write the JSON.

    Args:
        current_topology_json: Existing topology JSON string, or "{}" if building from scratch.
        user_feedback: Full description of what to build or what to change. Be specific.

    Returns:
        On success: execution plan summary with cost estimate.
        On failure: specific errors — gather more requirements from the user and retry.
    """
    # Module-level shim — script callers (no session). Sessions use the
    # closure-bound version registered inside create_master_toolkit /
    # create_session_scoped_toolkit, which forwards user/runtime context.
    return await _delegate_to_architect_impl(current_topology_json, user_feedback)


# ---------------------------------------------------------------------------
# Tool: register_skill
# ---------------------------------------------------------------------------

def register_skill(directory: str, targets: str = "all") -> str:
    """
    Register a skill directory for this session and persist it for future runs.

    Args:
        directory: Absolute or relative path to the skill directory.
        targets: "master", "subagent", or "all".
    """
    manager = get_skill_manager()
    try:
        name, was_new = manager.add_skill(directory, targets=targets, persist=True)
        if not was_new:
            return f"Skill '{name}' already registered."
        target_display = targets if targets != "all" else "master, subagent"
        return f"Skill '{name}' registered [{target_display}]."
    except SkillValidationError as exc:
        return f"Failed to register skill: {exc}"
    except Exception as exc:
        return f"Failed to register skill: {exc}"


# ---------------------------------------------------------------------------
# Toolkit factory
# ---------------------------------------------------------------------------

def create_master_toolkit() -> Toolkit:
    """Create the Toolkit for the master agent with per-instance ask_user state.

    For CLI/script use (single session). The API always uses session-scoped
    toolkits from create_session_scoped_toolkit() in sessions.py instead.

    The returned Toolkit has two extra attributes for script-layer coordination:
        toolkit.get_pending_questions() -> list | None
        toolkit.resolve_pending_questions(answers: list) -> None
    """
    toolkit = Toolkit()
    get_skill_manager().bind_toolkit(toolkit, target="master")
    apply_configured_skills(toolkit, target="master")

    # Per-toolkit ask_user state — no module-level globals, safe for concurrent toolkits.
    _state: dict = {"questions": None, "event": None, "answers": None}

    # Per-toolkit topology handoff slot — replaces the prior ContextVar.
    # Closure-bound delegate_to_architect writes the validated topology +
    # estimate here from its own ToolResponse metadata; the agent layer
    # reads it back via toolkit.get_last_topology_result().
    _last_topology: dict = {"topology": None, "estimate": None}

    # Per-toolkit architect grounding context — set by agent factory after the
    # runtime snapshot is built. Invisible to the LLM; forwarded to the
    # architect so it designs against today's date, MCP servers, gaps, user
    # memory, etc. instead of a bare tool-name list.
    _architect_ctx: dict = {
        "user_context": None,
        "runtime_context": None,
        "operator_config": None,
    }

    async def delegate_to_architect(
        current_topology_json: str,
        user_feedback: str,
    ) -> ToolResponse:
        """Design or modify a multi-agent execution topology via the Topology Architect.

        Call this when the task needs multiple specialized agents. NOT for one-shot tasks.
        Gather all requirements from the user before calling — DAAP does not write the JSON.

        Args:
            current_topology_json: Existing topology JSON string, or "{}" if building from scratch.
            user_feedback: Full description of what to build or what to change. Be specific.
        """
        result = await _delegate_to_architect_impl(
            current_topology_json,
            user_feedback,
            user_context=_architect_ctx.get("user_context"),
            runtime_context=_architect_ctx.get("runtime_context"),
            operator_config=_architect_ctx.get("operator_config"),
        )
        meta = result.metadata or {}
        if meta.get("topology") is not None:
            _last_topology["topology"] = meta.get("topology")
            _last_topology["estimate"] = meta.get("estimate")
        return result

    def get_last_topology_result() -> dict:
        return {
            "topology": _last_topology["topology"],
            "estimate": _last_topology["estimate"],
        }

    def clear_last_topology_result() -> None:
        _last_topology["topology"] = None
        _last_topology["estimate"] = None

    def set_architect_context(
        *,
        user_context: object = None,
        runtime_context: dict | None = None,
        operator_config: dict | None = None,
    ) -> None:
        """Inject grounding context the architect should see on its next call."""
        _architect_ctx["user_context"] = user_context
        _architect_ctx["runtime_context"] = runtime_context
        _architect_ctx["operator_config"] = operator_config

    async def ask_user(questions_json: str) -> ToolResponse:
        """Present structured options to the user and wait for their choice.

        Use ONLY for structured option-picker flows where the user must choose
        between labeled alternatives — not for open-ended clarification questions.

        Primary use cases:
        - Topology approval: "Proceed / Make it cheaper / Cancel"
        - Any other explicit multiple-choice decision

        For open-ended clarification (missing product, audience, format, etc.),
        ask in plain text instead and let the user reply in the next turn.

        Args:
            questions_json: A JSON string containing an array of questions.
                Each question has:
                - "question": the question text
                - "options": array of options, each with "label" and "description"
                - "multi_select": boolean, true if user can pick multiple options

                Keep it to 1-4 questions.
                Mark the recommended option with "(Recommended)" in the label.

        Returns:
            The user's selected answers.
        """
        try:
            questions = json.loads(questions_json)
        except json.JSONDecodeError as e:
            return ToolResponse(content=[TextBlock(type="text", text=f"Invalid questions JSON: {e}")])

        if not isinstance(questions, list) or len(questions) == 0:
            return ToolResponse(content=[TextBlock(
                type="text",
                text="questions_json must be a non-empty JSON array of question objects.",
            )])

        _state["questions"] = questions
        _state["event"] = asyncio.Event()
        _state["answers"] = None

        await _state["event"].wait()

        answers = _state["answers"]
        _state["questions"] = None
        _state["event"] = None
        _state["answers"] = None

        if answers is None:
            return ToolResponse(content=[TextBlock(
                type="text",
                text="The user did not provide answers. Proceed with what you know or ask again.",
            )])

        answer_lines = []
        for i, q in enumerate(questions):
            a = answers[i] if i < len(answers) else "(no answer)"
            answer_lines.append(f"Q: {q.get('question', '')}\nA: {a}")

        return ToolResponse(content=[TextBlock(
            type="text",
            text="User's answers:\n\n" + "\n\n".join(answer_lines),
        )])

    def get_pending_questions() -> list | None:
        """Return pending ask_user questions for this toolkit instance."""
        return _state["questions"]

    def resolve_pending_questions(answers: list) -> None:
        """Deliver answers for this toolkit instance's pending ask_user call."""
        _state["answers"] = answers
        if _state["event"] is not None:
            _state["event"].set()

    async def get_runtime_context() -> ToolResponse:
        """Return current master runtime capabilities and infrastructure snapshot.

        Call this before answering questions like:
        - what tools/capabilities you currently have
        - whether memory, MCP, or execution features are available
        - what infrastructure you are operating with right now
        """
        snapshot = build_master_runtime_snapshot(
            toolkit,
            execution_mode="script",
        )
        return ToolResponse(content=[TextBlock(
            type="text",
            text=json.dumps(snapshot, indent=2),
        )])

    toolkit.register_tool_function(delegate_to_architect)
    toolkit.register_tool_function(ask_user)
    toolkit.register_tool_function(register_skill)
    toolkit.register_tool_function(get_runtime_context)

    # Attach state accessors for script-layer coordination (not part of Toolkit API)
    toolkit.get_pending_questions = get_pending_questions
    toolkit.resolve_pending_questions = resolve_pending_questions
    toolkit.set_architect_context = set_architect_context
    toolkit.get_last_topology_result = get_last_topology_result
    toolkit.clear_last_topology_result = clear_last_topology_result

    return toolkit
