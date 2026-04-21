"""
DAAP Master Agent Tools — generate_topology and ask_user.

Both registered via AgentScope Toolkit. The agent calls these instead of
producing plain text — the API layer uses tool calls as reliable signals,
not fragile heuristics (no question-mark detection, no guessing).

State model:
  _topology_result_var  — ContextVar (task-scoped): isolates concurrent sessions
  ask_user state        — per-toolkit-instance closures (see create_master_toolkit)
                          API uses session-scoped toolkits from sessions.py instead
"""

import asyncio
import json
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

from agentscope.message import TextBlock
from agentscope.tool import Toolkit, ToolResponse

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


# ---------------------------------------------------------------------------
# Module-level shared state
# ---------------------------------------------------------------------------

# ContextVar — each asyncio Task (i.e. each session's agent task) gets its own
# copy, so concurrent sessions never overwrite each other's topology results.
_topology_result_var: ContextVar[dict] = ContextVar(
    "daap_topology_result",
    default={"topology": None, "estimate": None},
)

# Note: ask_user state is now per-toolkit-instance (see create_master_toolkit).
# The API always uses session-scoped toolkits (create_session_scoped_toolkit in
# sessions.py) and never reads these globals. Scripts use create_master_toolkit
# which exposes get/resolve via toolkit attributes.


def get_last_topology_result() -> dict:
    """API layer calls this to retrieve the most recent validated topology."""
    return _topology_result_var.get()


def clear_last_topology_result() -> None:
    _topology_result_var.set({"topology": None, "estimate": None})




# ---------------------------------------------------------------------------
# Tool: generate_topology
# ---------------------------------------------------------------------------

async def generate_topology(topology_json: str) -> ToolResponse:
    """Design and validate a multi-agent execution topology.

    Call this tool when the user's task requires multiple specialized agents
    working together. For example: research + evaluation + content creation,
    or tasks that need to find data from multiple sources then process it.

    Do NOT call this for simple tasks you can handle directly — writing a
    single email, answering a question, brainstorming, or summarizing.

    Before calling this tool, make sure you have gathered enough information
    from the user to design a good topology. Ask clarifying questions first
    if needed (their product, ICP, target audience, preferences, etc).

    Args:
        topology_json: A complete TopologySpec as a JSON string. Must include:
            - topology_id: unique ID (format: "topo-" + 8 hex chars)
            - version: 1
            - created_at: ISO 8601 timestamp
            - user_prompt: the user's original request (verbatim)
            - nodes: list of agent nodes (each with node_id, role, model_tier,
              system_prompt, tools, inputs, outputs, instance_config, agent_mode)
            - edges: list of connections between nodes
            - constraints: execution limits (or omit for defaults)

            Model tiers: "fast" (search/extract), "smart" (reasoning/writing),
            "powerful" (complex planning — rarely needed).
            Agent modes: "react" (iterative tool use), "single" (one-pass, no tools).
            Consolidation (when parallel_instances > 1): "merge", "deduplicate", "rank", "vote".

    Returns:
        On success: the execution plan summary with cost estimate.
        On validation failure: specific errors with fix suggestions.
    """
    # 1. Parse JSON
    try:
        topology_dict = json.loads(topology_json)
    except json.JSONDecodeError as e:
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Invalid JSON: {e}. Provide valid JSON for the topology.",
        )])

    # Auto-fill topology_id and created_at if not provided
    if "topology_id" not in topology_dict or not topology_dict["topology_id"]:
        topology_dict["topology_id"] = "topo-" + uuid.uuid4().hex[:8]
    if "created_at" not in topology_dict or not topology_dict["created_at"]:
        topology_dict["created_at"] = datetime.now(timezone.utc).isoformat()

    # 2. Pydantic parse
    try:
        topology = TopologySpec.model_validate(topology_dict)
    except Exception as e:
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Topology schema validation failed:\n{e}\n\nFix the issues and call generate_topology again.",
        )])

    # 3. Validate
    available_tools = get_available_tool_names()
    available_models = {"fast", "smart", "powerful"}
    validation = validate_topology(topology, available_tools, available_models)

    if not validation.is_valid:
        return ToolResponse(content=[TextBlock(
            type="text",
            text=(
                "Topology has validation errors. Fix ALL of these and call generate_topology again:\n\n"
                + validation.error_summary
            ),
        )])

    # 4. Resolve
    resolved = resolve_topology(topology)
    if isinstance(resolved, list):
        error_text = "\n".join(f"- {e.message}" for e in resolved)
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Topology passed validation but failed resolution:\n{error_text}\nFix and retry.",
        )])

    # 5. Estimate
    estimate = estimate_topology(resolved)

    # Store for API layer extraction
    estimate_data = {
        "total_cost_usd": estimate.total_cost_usd,
        "total_latency_seconds": estimate.total_latency_seconds,
        "min_viable_cost_usd": estimate.min_viable_cost_usd,
        "within_budget": estimate.within_budget,
        "within_timeout": estimate.within_timeout,
    }
    _topology_result_var.set({
        "topology": topology_dict,
        "estimate": estimate_data,
    })

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


# ---------------------------------------------------------------------------
# Tool: ask_user
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

    async def ask_user(questions_json: str) -> ToolResponse:
        """Ask the user clarifying questions before proceeding.

        Call this tool when you need more information from the user to do your
        job well. DO NOT guess or make assumptions about their product, audience,
        preferences, or requirements. Ask instead.

        Use this BEFORE calling generate_topology if the user's request is
        missing critical details like:
        - What their product/service does
        - Who their target audience is (industry, company size, job titles)
        - How many leads/results they want
        - What format they want output in (emails, report, list)
        - Any specific preferences or constraints

        Also use this when presenting a topology plan to offer the user
        choices: proceed, make cheaper, modify, or cancel.

        Args:
            questions_json: A JSON string containing an array of questions.
                Each question has:
                - "question": the question text
                - "options": array of options, each with "label" and "description"
                - "multi_select": boolean, true if user can pick multiple options

                Keep it to 1-4 questions. Don't interrogate the user.
                If you recommend a specific option, add "(Recommended)" to the label.

        Returns:
            The user's answers to your questions.
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

    toolkit.register_tool_function(generate_topology)
    toolkit.register_tool_function(ask_user)
    toolkit.register_tool_function(register_skill)
    toolkit.register_tool_function(get_runtime_context)

    # Attach state accessors for script-layer coordination (not part of Toolkit API)
    toolkit.get_pending_questions = get_pending_questions
    toolkit.resolve_pending_questions = resolve_pending_questions

    return toolkit
