"""
DAAP Session Management — in-memory session store with session-scoped toolkit.

Each session holds a live master agent, conversation history, and pending
topology state. The session-scoped ask_user closure fixes the concurrency
bug from Section 4's module-level state.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field

from agentscope.message import TextBlock
from agentscope.tool import Toolkit, ToolResponse

from daap.master.tools import generate_topology as generate_topology_tool


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """A single user conversation session."""
    session_id: str
    created_at: float
    user_id: str = "default"

    master_agent: object | None = None          # ReActAgent (typed as object to avoid circular)
    conversation: list[dict] = field(default_factory=list)

    # Topology state — stored as raw dict after generate_topology succeeds
    pending_topology: dict | None = None        # raw TopologySpec dict
    pending_estimate: dict | None = None        # estimate data dict

    # Execution state
    is_executing: bool = False
    execution_result: dict | None = None
    execution_progress: dict | None = None

    # Optional model selection (set at session creation)
    master_operator_config: dict | None = None
    subagent_operator_config: dict | None = None

    # Token usage tracking — reset per turn, read by ws_handler
    token_tracker: object | None = None         # TokenTracker instance

    # ask_user state — set by session-scoped closure, read by ws_handler
    pending_questions: list | None = None
    _resolve_answers: object | None = None      # callable injected by create_session_scoped_toolkit


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    In-memory session store.

    Phase 1: dict in memory — sessions lost on server restart.
    Phase 3: Redis-backed sessions via AgentScope Runtime.
    """

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create_session(self) -> Session:
        session_id = str(uuid.uuid4())[:8]
        session = Session(session_id=session_id, created_at=time.time())
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[dict]:
        return [
            {
                "session_id": s.session_id,
                "created_at": s.created_at,
                "message_count": len(s.conversation),
                "has_pending_topology": s.pending_topology is not None,
                "is_executing": s.is_executing,
            }
            for s in self._sessions.values()
        ]


# ---------------------------------------------------------------------------
# Session-scoped toolkit factory
# ---------------------------------------------------------------------------

def create_session_scoped_toolkit(session: Session) -> Toolkit:
    """
    Build a Toolkit with ask_user scoped to this specific session.

    The session-scoped ask_user closure:
    - Stores pending questions on the Session object (not module-level state)
    - Uses a per-session asyncio.Event — no cross-session interference
    - Attaches session._resolve_answers for the WebSocket handler to call

    generate_topology is wrapped per session so we can inject user-selected
    subagent model/operator settings before topology validation.
    """
    toolkit = Toolkit()

    def _merge_operator_config(existing: dict | None, selected: dict) -> dict:
        existing_cfg = existing if isinstance(existing, dict) else {}
        merged = dict(existing_cfg)

        for key in ("provider", "base_url", "api_key_env"):
            selected_val = selected.get(key)
            if selected_val is not None:
                merged[key] = selected_val

        existing_model_map = (
            existing_cfg.get("model_map")
            if isinstance(existing_cfg.get("model_map"), dict)
            else {}
        )
        selected_model_map = (
            selected.get("model_map")
            if isinstance(selected.get("model_map"), dict)
            else {}
        )

        model_map = dict(existing_model_map)
        model_map.update({k: v for k, v in selected_model_map.items() if v})
        merged["model_map"] = model_map
        return merged

    async def generate_topology(topology_json: str) -> ToolResponse:
        """Session-scoped topology generation with optional operator/model injection."""
        selected = session.subagent_operator_config
        if not selected:
            return await generate_topology_tool(topology_json)

        try:
            topology_dict = json.loads(topology_json)
        except json.JSONDecodeError:
            return await generate_topology_tool(topology_json)

        # Enforce user-selected default operator/model map for all subagents.
        topology_dict["operator_config"] = _merge_operator_config(
            topology_dict.get("operator_config"),
            selected,
        )

        # Node-level overrides still exist, but user selection should win.
        for node in topology_dict.get("nodes", []):
            if not isinstance(node, dict):
                continue
            if node.get("operator_override") is not None:
                node["operator_override"] = _merge_operator_config(
                    node.get("operator_override"),
                    selected,
                )

        return await generate_topology_tool(json.dumps(topology_dict))

    toolkit.register_tool_function(generate_topology)

    # Per-session state captured by the closure
    _answer_event: asyncio.Event = asyncio.Event()
    _state: dict = {"user_answers": None}

    async def ask_user(questions_json: str) -> ToolResponse:
        """Ask the user clarifying questions before proceeding.

        Call this tool when you need more information from the user to do
        your job well. DO NOT guess or make assumptions. Ask instead.

        Use BEFORE calling generate_topology if the user's request is
        missing: product/service details, target audience, preferences,
        volume, or output format. Also use to present a topology plan
        and get approval: proceed, make cheaper, modify, or cancel.

        Args:
            questions_json: JSON array of question objects. Each has:
                - "question": the question text
                - "options": array of {"label": str, "description": str}
                - "multi_select": boolean

                Keep to 1-4 questions. List recommended option first
                and mark with "(Recommended)".

        Returns:
            The user's answers to your questions.
        """
        try:
            questions = json.loads(questions_json)
        except json.JSONDecodeError as exc:
            return ToolResponse(content=[TextBlock(
                type="text", text=f"Invalid questions JSON: {exc}",
            )])

        if not isinstance(questions, list) or len(questions) == 0:
            return ToolResponse(content=[TextBlock(
                type="text",
                text="questions_json must be a non-empty JSON array.",
            )])

        # Store on session — WebSocket handler polls this
        session.pending_questions = questions
        _answer_event.clear()
        _state["user_answers"] = None

        # Pause until WebSocket handler calls _resolve_answers()
        await _answer_event.wait()

        # Clear pending state
        session.pending_questions = None
        answers = _state["user_answers"]

        if answers is None:
            return ToolResponse(content=[TextBlock(
                type="text",
                text="User did not provide answers. Proceed with what you know or ask again.",
            )])

        lines = []
        for i, q in enumerate(questions):
            a = answers[i] if i < len(answers) else "(no answer)"
            lines.append(f"Q: {q.get('question', '')}\nA: {a}")

        return ToolResponse(content=[TextBlock(
            type="text",
            text="User's answers:\n\n" + "\n\n".join(lines),
        )])

    def resolve_answers(answers: list[str]) -> None:
        _state["user_answers"] = answers
        _answer_event.set()

    # Attach resolver to session so ws_handler can call it
    session._resolve_answers = resolve_answers

    toolkit.register_tool_function(ask_user)

    async def get_execution_status() -> ToolResponse:
        """Check the status of the current or most recent topology execution.

        Call this when the user asks about execution progress, whether the
        topology is running, or what the results were. This is the ONLY way
        to check execution state — do NOT say you cannot check it.

        Returns:
            Current execution status and results (if available).
        """
        progress = session.execution_progress or {}

        if session.is_executing:
            completed = int(progress.get("completed_nodes", 0))
            total = int(progress.get("total_nodes", 0))
            remaining = int(progress.get("remaining_nodes", max(total - completed, 0)))
            percent = int(progress.get("percent_complete", 0))
            current_node = progress.get("current_node") or "initializing"
            return ToolResponse(content=[TextBlock(
                type="text",
                text=(
                    "Topology is currently executing.\n"
                    f"Progress: {completed}/{total} nodes complete ({percent}%).\n"
                    f"Current node: {current_node}\n"
                    f"Remaining nodes: {remaining}"
                ),
            )])

        result = session.execution_result
        if result is None:
            pending = session.pending_topology
            if pending:
                nodes = [n.get("node_id") for n in pending.get("nodes", [])]
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text=f"No execution yet. A topology is pending approval: nodes={nodes}. User must type 'approve' to execute.",
                )])
            return ToolResponse(content=[TextBlock(
                type="text",
                text="No topology has been executed in this session yet.",
            )])

        success = result.get("success", False)
        if success:
            completed = int(progress.get("completed_nodes", 0))
            total = int(progress.get("total_nodes", 0))
            progress_line = f"Progress: {completed}/{total} nodes completed\n" if total else ""
            return ToolResponse(content=[TextBlock(
                type="text",
                text=(
                    f"Last execution: SUCCESS\n"
                    f"Topology: {result.get('topology_id', 'unknown')}\n"
                    f"Latency: {result.get('latency_seconds', 0):.1f}s\n"
                    f"{progress_line}"
                    f"Output:\n{result.get('final_output', '')}"
                ),
            )])
        else:
            return ToolResponse(content=[TextBlock(
                type="text",
                text=(
                    f"Last execution: FAILED\n"
                    f"Error: {result.get('error', 'unknown error')}"
                ),
            )])

    toolkit.register_tool_function(get_execution_status)
    return toolkit
