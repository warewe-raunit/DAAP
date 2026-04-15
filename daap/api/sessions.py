"""
DAAP Session Management — in-memory session store with session-scoped toolkit.

Each session holds a live master agent, conversation history, and pending
topology state. The session-scoped ask_user closure fixes the concurrency
bug from Section 4's module-level state.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field

from agentscope.message import TextBlock
from agentscope.tool import Toolkit, ToolResponse

from daap.master.tools import generate_topology as generate_topology_tool

logger = logging.getLogger(__name__)


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

    # WebSocket send callback — set by ws_handler so agent tools can stream progress
    _ws_send: object | None = None              # async callable: (dict) -> None


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

def create_session_scoped_toolkit(
    session: Session,
    topology_store=None,
    daap_memory=None,
) -> Toolkit:
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

    # ------------------------------------------------------------------
    # Topology persistence tools
    # ------------------------------------------------------------------

    if topology_store is not None:

        async def load_topology(topology_id: str, version: int | None = None) -> ToolResponse:
            """Load a saved topology into session.pending_topology for editing or rerun."""
            stored = topology_store.get_topology(topology_id, version=version)
            if stored is None:
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text=f"Topology '{topology_id}' not found in store.",
                )])
            session.pending_topology = stored.spec
            session.pending_estimate = None
            return ToolResponse(content=[TextBlock(
                type="text",
                text=(
                    f"Loaded topology '{topology_id}' (v{stored.version}) into session. "
                    f"Name: '{stored.name}'. Ready to edit or rerun."
                ),
            )])

        async def persist_topology(topology_id: str, save_mode: str) -> ToolResponse:
            """Persist session.pending_topology to the topology store."""
            if session.pending_topology is None:
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text="No pending topology to save. Generate or load one first.",
                )])

            if save_mode not in {"overwrite", "new_version"}:
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text="Invalid save_mode. Use 'overwrite' or 'new_version'.",
                )])

            overwrite = save_mode == "overwrite"
            spec_to_save = dict(session.pending_topology)
            spec_to_save["topology_id"] = topology_id

            stored = topology_store.save_topology(
                spec=spec_to_save,
                user_id=session.user_id,
                overwrite=overwrite,
            )
            session.pending_topology = stored.spec
            return ToolResponse(content=[TextBlock(
                type="text",
                text=(
                    f"Topology saved: id='{stored.topology_id}', "
                    f"version={stored.version}, name='{stored.name}'."
                ),
            )])

        async def rerun_topology(
            topology_id: str,
            user_prompt: str | None = None,
        ) -> ToolResponse:
            """Load and execute a saved topology with optional prompt override."""
            from daap.executor.engine import execute_topology
            from daap.spec.resolver import resolve_topology
            from daap.spec.schema import TopologySpec

            stored = topology_store.get_topology(topology_id)
            if stored is None:
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text=f"Topology '{topology_id}' not found.",
                )])
            if stored.deleted_at is not None:
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text=f"Topology '{topology_id}' has been deleted and cannot be rerun.",
                )])

            prompt = user_prompt or stored.spec.get("user_prompt", "")

            try:
                spec = TopologySpec.model_validate(stored.spec)
                resolved = resolve_topology(spec)
                if isinstance(resolved, list):
                    errors = "; ".join(error.message for error in resolved)
                    return ToolResponse(content=[TextBlock(
                        type="text",
                        text=f"Topology resolution failed: {errors}",
                    )])

                result = await execute_topology(
                    resolved=resolved,
                    user_prompt=prompt,
                    tracker=session.token_tracker,
                )

                result_payload = {
                    "topology_id": result.topology_id,
                    "final_output": result.final_output,
                    "success": result.success,
                    "error": result.error,
                    "latency_seconds": result.total_latency_seconds,
                    "total_input_tokens": result.total_input_tokens,
                    "total_output_tokens": result.total_output_tokens,
                }
                session.execution_result = result_payload

                topology_store.save_run(
                    topology_id=topology_id,
                    topology_version=stored.version,
                    user_id=session.user_id,
                    result=result_payload,
                    user_prompt=prompt,
                )

                if result.success:
                    return ToolResponse(content=[TextBlock(
                        type="text",
                        text=(
                            f"Rerun complete. Latency: {result.total_latency_seconds:.1f}s. "
                            f"Tokens: {result.total_input_tokens} in / {result.total_output_tokens} out.\n"
                            f"Output:\n{result.final_output}"
                        ),
                    )])

                return ToolResponse(content=[TextBlock(
                    type="text",
                    text=f"Rerun failed: {result.error}",
                )])
            except Exception as exc:
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text=f"Rerun error: {exc}",
                )])

        toolkit.register_tool_function(load_topology)
        toolkit.register_tool_function(persist_topology)
        toolkit.register_tool_function(rerun_topology)

    # ------------------------------------------------------------------
    # execute_pending_topology — always registered
    # Lets the agent execute a newly-generated topology after getting
    # user approval via ask_user, without any hardcoded keyword detection.
    # ------------------------------------------------------------------

    async def execute_pending_topology() -> ToolResponse:
        """Execute the pending topology that was just approved by the user.

        Call this immediately after the user confirms they want to proceed
        via ask_user. Do NOT call this without user approval.

        Streams progress to the client (if connected via WebSocket) and
        returns the full execution result so you can summarise it.

        Returns:
            Execution result including output, latency, and token usage.
        """
        if session.pending_topology is None:
            return ToolResponse(content=[TextBlock(
                type="text",
                text="No pending topology to execute. Call generate_topology first.",
            )])

        if session.is_executing:
            return ToolResponse(content=[TextBlock(
                type="text",
                text="A topology is already executing. Wait for it to finish.",
            )])

        from daap.spec.schema import TopologySpec
        from daap.spec.resolver import resolve_topology
        from daap.executor.engine import execute_topology as _execute_topology

        topo_dict = session.pending_topology
        topo_id = topo_dict.get("topology_id", "unknown")
        user_prompt = topo_dict.get("user_prompt", "")
        ws_send = session._ws_send  # None in CLI, set by ws_handler for WebSocket

        try:
            spec = TopologySpec.model_validate(topo_dict)
            resolved = resolve_topology(spec)
            if isinstance(resolved, list):
                errors = "; ".join(e.message for e in resolved)
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text=f"Topology resolution failed: {errors}",
                )])

            total_nodes = len(resolved.nodes)
            session.is_executing = True
            session.execution_progress = {
                "topology_id": topo_id,
                "status": "running",
                "total_nodes": total_nodes,
                "completed_nodes": 0,
                "remaining_nodes": total_nodes,
                "percent_complete": 0,
                "current_node": None,
            }

            if ws_send:
                await ws_send({
                    "type": "executing",
                    "topology_id": topo_id,
                    "total_nodes": total_nodes,
                })

            completed_nodes = 0

            def _on_node_start(node_id: str, model_id: str, step_num: int, total: int) -> None:
                session.execution_progress.update({
                    "current_node": node_id,
                    "completed_nodes": completed_nodes,
                    "remaining_nodes": max(total_nodes - completed_nodes, 0),
                    "percent_complete": int((completed_nodes / total_nodes) * 100) if total_nodes else 0,
                })
                if ws_send:
                    import asyncio
                    asyncio.get_event_loop().create_task(ws_send({
                        "type": "progress",
                        "event": "node_start",
                        "topology_id": topo_id,
                        "node_id": node_id,
                        "model_id": model_id,
                        "step_num": step_num,
                        "total_steps": total,
                        "completed_nodes": completed_nodes,
                        "total_nodes": total_nodes,
                    }))

            def _on_node_complete(nr) -> None:
                nonlocal completed_nodes
                completed_nodes += 1
                pct = int((completed_nodes / total_nodes) * 100) if total_nodes else 100
                session.execution_progress.update({
                    "completed_nodes": completed_nodes,
                    "remaining_nodes": max(total_nodes - completed_nodes, 0),
                    "percent_complete": pct,
                    "current_node": nr.node_id,
                })
                if ws_send:
                    import asyncio
                    asyncio.get_event_loop().create_task(ws_send({
                        "type": "progress",
                        "event": "node_complete",
                        "topology_id": topo_id,
                        "node_id": nr.node_id,
                        "completed_nodes": completed_nodes,
                        "total_nodes": total_nodes,
                        "percent_complete": pct,
                        "node_latency_seconds": nr.latency_seconds,
                    }))

            result = await _execute_topology(
                resolved=resolved,
                user_prompt=user_prompt,
                tracker=session.token_tracker,
                on_node_start=_on_node_start,
                on_node_complete=_on_node_complete,
                daap_memory=daap_memory,
            )

            session.execution_result = {
                "topology_id": result.topology_id,
                "final_output": result.final_output,
                "success": result.success,
                "error": result.error,
                "latency_seconds": result.total_latency_seconds,
                "total_input_tokens": result.total_input_tokens,
                "total_output_tokens": result.total_output_tokens,
            }

            # Write run + agent learnings to memory (non-fatal)
            if daap_memory is not None and result.success:
                try:
                    from daap.memory.writer import (
                        write_run_to_memory,
                        write_agent_learnings_from_run,
                    )
                    write_run_to_memory(
                        memory=daap_memory,
                        user_id=session.user_id,
                        topology_summary=user_prompt,
                        execution_result=result,
                    )
                    topology_nodes = topo_dict.get("nodes", [])
                    write_agent_learnings_from_run(
                        memory=daap_memory,
                        execution_result=result,
                        topology_nodes=topology_nodes,
                    )
                except Exception as _mem_exc:
                    logger.warning("Memory write failed (non-fatal): %s", _mem_exc)

            # Auto-save topology + run
            if topology_store is not None:
                try:
                    saved = topology_store.save_topology(
                        spec=topo_dict,
                        user_id=session.user_id,
                        overwrite=True,
                    )
                    topology_store.save_run(
                        topology_id=saved.topology_id,
                        topology_version=saved.version,
                        user_id=session.user_id,
                        result=session.execution_result,
                        user_prompt=user_prompt,
                    )
                except Exception as _save_exc:
                    logger.warning("Auto-save failed (non-fatal): %s", _save_exc)

            session.pending_topology = None
            session.pending_estimate = None

            if ws_send:
                await ws_send({
                    "type": "result",
                    "output": result.final_output,
                    "latency_seconds": result.total_latency_seconds,
                    "models_used": result.models_used,
                    "usage": {
                        "input_tokens": result.total_input_tokens,
                        "output_tokens": result.total_output_tokens,
                        "total_tokens": result.total_input_tokens + result.total_output_tokens,
                    },
                })

            if result.success:
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text=(
                        f"Execution complete. Ran in {result.total_latency_seconds:.1f}s. "
                        f"Tokens: {result.total_input_tokens} in / {result.total_output_tokens} out.\n"
                        f"Output:\n{result.final_output}"
                    ),
                )])
            else:
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text=f"Execution failed: {result.error}",
                )])

        except Exception as exc:
            logger.exception("execute_pending_topology tool failed for %s", topo_id)
            return ToolResponse(content=[TextBlock(
                type="text",
                text=f"Execution error: {exc}",
            )])
        finally:
            session.is_executing = False

    toolkit.register_tool_function(execute_pending_topology)

    return toolkit
