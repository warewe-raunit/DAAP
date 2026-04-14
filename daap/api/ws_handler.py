"""
DAAP WebSocket Handler — manages the master agent conversation loop over WebSocket.

Message protocol (JSON over WebSocket):

Client → Server:
  {"type": "message", "content": "..."}          user chat message
  {"type": "answer", "answers": ["..."]}          answers to ask_user questions
  {"type": "approve"}                             approve topology execution
  {"type": "make_cheaper"}                        request cheaper topology
  {"type": "cancel"}                              cancel pending topology

Server → Client:
  {"type": "response", "content": "..."}          agent text response
  {"type": "questions", "questions": [...]}        structured ask_user questions
  {"type": "plan", "summary": "...", ...}          topology plan awaiting approval
  {"type": "executing", "topology_id": "..."}      execution started
  {"type": "result", "output": "...", ...}         execution complete
  {"type": "error", "message": "..."}              error
"""

import asyncio
import json
import logging
import re

from fastapi import WebSocket, WebSocketDisconnect
from agentscope.message import Msg

from daap.api.sessions import Session
from daap.master.tools import clear_last_topology_result, get_last_topology_result
from daap.spec.schema import TopologySpec
from daap.spec.resolver import resolve_topology
from daap.executor.engine import execute_topology

logger = logging.getLogger(__name__)


_APPROVE_TEXT_COMMANDS = {
    "approve",
    "run",
    "run it",
    "go ahead",
    "execute",
    "execute now",
}
_CHEAPER_TEXT_COMMANDS = {
    "cheaper",
    "make cheaper",
    "make it cheaper",
    "reduce cost",
}
_CANCEL_TEXT_COMMANDS = {
    "cancel",
    "abort",
    "stop",
}
_TOPOLOGY_ID_RE = re.compile(r"\btopo-[0-9a-fA-F]{8}\b")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg_text(msg: Msg) -> str:
    return msg.content if isinstance(msg.content, str) else str(msg.content)


def _normalize_text_command(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _extract_topology_id(text: str) -> str | None:
    m = _TOPOLOGY_ID_RE.search(text or "")
    return m.group(0).lower() if m else None


def _detect_text_command(text: str) -> tuple[str, str | None] | None:
    """
    Detect command-like chat text for pending topology control.

    Returns:
      ("approve", topology_id|None)
      ("make_cheaper", None)
      ("cancel", None)
      None
    """
    normalized = _normalize_text_command(text)
    if not normalized:
        return None

    requested_topology_id = _extract_topology_id(normalized)

    if (
        normalized in _APPROVE_TEXT_COMMANDS
        or normalized.startswith("approve ")
        or normalized.startswith("execute ")
        or normalized.startswith("run topo-")
    ):
        return ("approve", requested_topology_id)

    if normalized in _CHEAPER_TEXT_COMMANDS:
        return ("make_cheaper", None)

    if normalized in _CANCEL_TEXT_COMMANDS:
        return ("cancel", None)

    return None


async def _run_agent_with_question_pump(
    websocket: WebSocket,
    session: Session,
    user_text: str,
) -> str:
    """
    Send user_text to the master agent and handle ask_user pauses.

    Runs the agent as a background task and polls session.pending_questions.
    When the agent calls ask_user, questions appear on the session; we send
    them to the client, wait for {"type":"answer"}, and resume the agent.

    Returns the agent's final response text.
    """
    agent_msg = Msg(name="user", content=user_text, role="user")
    agent_task = asyncio.create_task(session.master_agent(agent_msg))

    while not agent_task.done():
        if session.pending_questions is not None:
            # Agent is paused waiting for user input
            await websocket.send_json({
                "type": "questions",
                "questions": session.pending_questions,
            })

            # Wait for client answer
            raw = await websocket.receive_text()
            data = json.loads(raw)
            if data.get("type") == "answer":
                session._resolve_answers(data.get("answers", []))
            # else: unexpected message type — ignore, agent stays paused

        await asyncio.sleep(0.05)

    response_msg = agent_task.result()
    return _msg_text(response_msg)


async def _check_and_send_topology(
    websocket: WebSocket,
    session: Session,
    response_text: str,
) -> bool:
    """
    Check if generate_topology was called this turn. If so, store it and
    send a plan message to the client.

    Returns True if a topology was found, False otherwise.
    """
    topo_result = get_last_topology_result()
    if topo_result.get("topology") is None:
        return False

    session.pending_topology = topo_result["topology"]
    session.pending_estimate = topo_result["estimate"]
    clear_last_topology_result()

    est = session.pending_estimate or {}
    tracker = getattr(session, "token_tracker", None)
    usage = tracker.to_dict() if tracker else {}
    await websocket.send_json({
        "type": "plan",
        "summary": response_text,
        "cost_usd": est.get("total_cost_usd", 0),
        "latency_seconds": est.get("total_latency_seconds", 0),
        "min_cost_usd": est.get("min_viable_cost_usd", 0),
        "usage": usage,
    })
    return True


async def _run_make_cheaper_flow(
    websocket: WebSocket,
    session: Session,
) -> None:
    clear_last_topology_result()
    cheaper_prompt = "Make the topology cheaper. Reduce cost while keeping it functional."
    session.conversation.append({"role": "user", "content": cheaper_prompt})

    tracker = session.token_tracker
    if tracker:
        tracker.reset()

    response_text = await _run_agent_with_question_pump(websocket, session, cheaper_prompt)
    session.conversation.append({"role": "assistant", "content": response_text})

    usage = tracker.to_dict() if tracker else {}
    if not await _check_and_send_topology(websocket, session, response_text):
        await websocket.send_json({
            "type": "response",
            "content": response_text,
            "usage": usage,
        })


async def _execute_pending_topology(
    websocket: WebSocket,
    session: Session,
    daap_memory=None,
    topology_store=None,
) -> None:
    """Execute the currently pending topology and stream progress updates."""
    if session.is_executing:
        await websocket.send_json({
            "type": "error",
            "message": "A topology is already executing in this session.",
        })
        return

    if session.pending_topology is None:
        await websocket.send_json({
            "type": "error",
            "message": "No topology pending for approval.",
        })
        return

    topo_id = session.pending_topology.get("topology_id", "unknown")
    user_prompt = session.pending_topology.get(
        "user_prompt",
        session.conversation[0]["content"] if session.conversation else "",
    )

    session.execution_progress = {
        "topology_id": topo_id,
        "status": "preparing",
        "total_nodes": 0,
        "completed_nodes": 0,
        "remaining_nodes": 0,
        "percent_complete": 0,
        "current_node": None,
    }

    try:
        # Parse dict -> TopologySpec -> ResolvedTopology
        topology_spec = TopologySpec.model_validate(session.pending_topology)
        resolved = resolve_topology(topology_spec)
        if isinstance(resolved, list):
            errors = "; ".join(e.message for e in resolved)
            raise ValueError(f"Topology resolution failed: {errors}")

        total_nodes = len(resolved.nodes)
        session.is_executing = True
        session.execution_progress.update({
            "status": "running",
            "total_nodes": total_nodes,
            "remaining_nodes": total_nodes,
        })

        await websocket.send_json({
            "type": "executing",
            "topology_id": topo_id,
            "total_nodes": total_nodes,
        })

        callback_tasks: list[asyncio.Task] = []
        loop = asyncio.get_running_loop()
        completed_nodes = 0

        def _queue_progress(payload: dict) -> None:
            callback_tasks.append(loop.create_task(websocket.send_json(payload)))

        def _on_node_start(node_id: str, model_id: str, step_num: int, total_steps: int) -> None:
            remaining_nodes = max(total_nodes - completed_nodes, 0)
            percent_complete = int((completed_nodes / total_nodes) * 100) if total_nodes else 0
            session.execution_progress.update({
                "status": "running",
                "current_node": node_id,
                "completed_nodes": completed_nodes,
                "remaining_nodes": remaining_nodes,
                "percent_complete": percent_complete,
                "step_num": step_num,
                "total_steps": total_steps,
            })
            _queue_progress({
                "type": "progress",
                "event": "node_start",
                "topology_id": topo_id,
                "node_id": node_id,
                "model_id": model_id,
                "step_num": step_num,
                "total_steps": total_steps,
                "completed_nodes": completed_nodes,
                "total_nodes": total_nodes,
                "remaining_nodes": remaining_nodes,
                "percent_complete": percent_complete,
            })

        def _on_node_complete(node_result) -> None:
            nonlocal completed_nodes
            completed_nodes += 1
            remaining_nodes = max(total_nodes - completed_nodes, 0)
            percent_complete = int((completed_nodes / total_nodes) * 100) if total_nodes else 100
            session.execution_progress.update({
                "status": "running",
                "current_node": node_result.node_id,
                "completed_nodes": completed_nodes,
                "remaining_nodes": remaining_nodes,
                "percent_complete": percent_complete,
            })
            _queue_progress({
                "type": "progress",
                "event": "node_complete",
                "topology_id": topo_id,
                "node_id": node_result.node_id,
                "completed_nodes": completed_nodes,
                "total_nodes": total_nodes,
                "remaining_nodes": remaining_nodes,
                "percent_complete": percent_complete,
                "node_latency_seconds": node_result.latency_seconds,
            })

        result = await execute_topology(
            resolved=resolved,
            user_prompt=user_prompt,
            tracker=session.token_tracker,
            on_node_start=_on_node_start,
            on_node_complete=_on_node_complete,
        )

        if callback_tasks:
            await asyncio.gather(*callback_tasks, return_exceptions=True)

        session.execution_result = {
            "topology_id": result.topology_id,
            "final_output": result.final_output,
            "success": result.success,
            "error": result.error,
            "latency_seconds": result.total_latency_seconds,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
        }

        # Auto-save topology + run to persistent store.
        if topology_store is not None:
            try:
                saved = topology_store.save_topology(
                    spec=session.pending_topology,
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
            except Exception as exc:
                logger.warning("Auto-save topology failed (non-fatal): %s", exc)

        if result.success:
            session.execution_progress.update({
                "status": "success",
                "current_node": None,
                "completed_nodes": total_nodes,
                "remaining_nodes": 0,
                "percent_complete": 100,
            })

            await websocket.send_json({
                "type": "result",
                "output": result.final_output,
                "latency_seconds": result.total_latency_seconds,
                "models_used": result.models_used,
                "usage": {
                    "input_tokens": result.total_input_tokens,
                    "output_tokens": result.total_output_tokens,
                    "total_tokens": result.total_input_tokens + result.total_output_tokens,
                },
                "progress": {
                    "completed_nodes": total_nodes,
                    "total_nodes": total_nodes,
                    "percent_complete": 100,
                },
            })

            # Write run + agent learnings to memory (optional)
            if daap_memory:
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
                    topology_nodes = (
                        session.pending_topology.get("nodes", [])
                        if session.pending_topology else []
                    )
                    write_agent_learnings_from_run(
                        memory=daap_memory,
                        execution_result=result,
                        topology_nodes=topology_nodes,
                    )
                except Exception as exc:
                    logger.warning("Failed to write run to memory: %s", exc)
        else:
            session.execution_progress.update({
                "status": "failed",
                "error": result.error,
            })
            await websocket.send_json({
                "type": "error",
                "message": f"Execution failed: {result.error}",
            })

    except Exception as exc:
        session.execution_progress.update({
            "status": "failed",
            "error": str(exc),
        })
        await websocket.send_json({
            "type": "error",
            "message": f"Execution error: {exc}",
        })
    finally:
        # Always release the lock — guards against CancelledError and any
        # other BaseException subclass that bypasses the except block.
        session.is_executing = False


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

async def handle_websocket(
    websocket: WebSocket,
    session: Session,
    daap_memory=None,
    topology_store=None,
) -> None:
    """
    Handle a full WebSocket conversation with the master agent.

    The caller (routes.py) is responsible for verifying the session exists
    and has a master_agent before calling this function.

    daap_memory: optional DaapMemory instance for post-run writes.
                 None = memory disabled (first-time users, missing keys, etc.)
    """
    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")

            # ------------------------------------------------------------------
            if msg_type == "message":
                user_text = data.get("content", "")
                session.conversation.append({"role": "user", "content": user_text})

                # If a topology is pending, allow natural-language control commands
                # so users can execute/cheapen/cancel without special client message types.
                if session.pending_topology is not None:
                    cmd = _detect_text_command(user_text)
                    if cmd is not None:
                        action, requested_topology_id = cmd

                        if action == "approve":
                            pending_id = str(
                                session.pending_topology.get("topology_id", "")
                            ).lower()
                            if (
                                requested_topology_id is not None
                                and pending_id
                                and requested_topology_id != pending_id
                            ):
                                await websocket.send_json({
                                    "type": "error",
                                    "message": (
                                        f"Requested topology '{requested_topology_id}' does not match "
                                        f"pending topology '{pending_id}'."
                                    ),
                                })
                                continue

                            await _execute_pending_topology(
                                websocket,
                                session,
                                daap_memory=daap_memory,
                                topology_store=topology_store,
                            )
                            continue

                        if action == "make_cheaper":
                            await _run_make_cheaper_flow(websocket, session)
                            continue

                        if action == "cancel":
                            session.pending_topology = None
                            session.pending_estimate = None
                            await websocket.send_json({
                                "type": "response",
                                "content": "Cancelled. How else can I help?",
                            })
                            continue

                clear_last_topology_result()

                tracker = session.token_tracker
                if tracker:
                    tracker.reset()

                response_text = await _run_agent_with_question_pump(
                    websocket, session, user_text
                )
                session.conversation.append({"role": "assistant", "content": response_text})

                usage = tracker.to_dict() if tracker else {}
                if not await _check_and_send_topology(websocket, session, response_text):
                    await websocket.send_json({
                        "type": "response",
                        "content": response_text,
                        "usage": usage,
                    })

            # ------------------------------------------------------------------
            elif msg_type == "approve":
                await _execute_pending_topology(
                    websocket,
                    session,
                    daap_memory=daap_memory,
                    topology_store=topology_store,
                )

            # ------------------------------------------------------------------
            elif msg_type == "make_cheaper":
                await _run_make_cheaper_flow(websocket, session)

            # ------------------------------------------------------------------
            elif msg_type == "cancel":
                session.pending_topology = None
                session.pending_estimate = None
                await websocket.send_json({
                    "type": "response",
                    "content": "Cancelled. How else can I help?",
                })

            # ------------------------------------------------------------------
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type!r}",
                })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
