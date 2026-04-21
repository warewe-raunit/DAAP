"""
DAAP WebSocket Handler — manages the master agent conversation loop over WebSocket.

Message protocol (JSON over WebSocket):

Client → Server:
  {"type": "message", "content": "..."}                      user chat message
  {"type": "answer", "answers": ["..."]}                     answers to ask_user questions
  {"type": "permission_response", "granted": true/false}     file access approval
  {"type": "make_cheaper"}                                   request cheaper topology
  {"type": "cancel"}                                         cancel pending topology

Server → Client:
  {"type": "response", "content": "..."}                     agent text response
  {"type": "questions", "questions": [...]}                  structured ask_user questions
  {"type": "permission_request", "filepath": "...",
   "operation": "read"|"write"}                              out-of-cwd file access request
  {"type": "plan", "summary": "...", ...}                    topology plan awaiting approval
  {"type": "executing", "topology_id": "..."}                execution started (sent by agent tool)
  {"type": "result", "output": "...", ...}                   execution complete (sent by agent tool)
  {"type": "error", "message": "..."}                        error
"""

import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect
from agentscope.message import Msg

from daap.api.sessions import Session

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg_text(msg: Msg) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p).strip()
    return str(content)


def _normalize_text_command(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _detect_text_command(text: str) -> tuple[str, str | None] | None:
    """
    Detect command-like chat text for pending topology control.

    Returns:
      ("make_cheaper", None)
      ("cancel", None)
      None
    """
    normalized = _normalize_text_command(text)
    if not normalized:
        return None

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

    Runs the agent as a background task. When the agent calls ask_user it sets
    session._questions_event; we wait on that event instead of polling, so
    questions are dispatched immediately with no sleep overhead.

    Returns the agent's final response text.
    """
    agent_msg = Msg(name="user", content=user_text, role="user")
    agent_task = asyncio.create_task(session.master_agent(agent_msg))

    try:
        while not agent_task.done():
            if session.pending_questions is not None:
                # If the agent just generated a topology in this same turn,
                # send the plan spec FIRST so the user sees what they are
                # approving before the approval question appears.
                if session.topology_just_generated:
                    session.topology_just_generated = False
                    est = session.pending_estimate or {}
                    topo = session.pending_topology or {}
                    raw_nodes = [n for n in topo.get("nodes", []) if isinstance(n, dict)]
                    node_parts = []
                    for n in raw_nodes:
                        nid = n.get("node_id", "")
                        instances = n.get("instance_config", {}).get("parallel_instances", 1)
                        node_parts.append(f"{nid} ×{instances}" if instances > 1 else nid)
                    tracker = getattr(session, "token_tracker", None)
                    usage_snapshot = tracker.to_dict() if tracker else {}
                    await websocket.send_json({
                        "type": "plan",
                        "summary": f"{len(node_parts)} node(s): {', '.join(node_parts)}",
                        "topology": topo,
                        "cost_usd": est.get("total_cost_usd", 0),
                        "latency_seconds": est.get("total_latency_seconds", 0),
                        "min_cost_usd": est.get("min_viable_cost_usd", 0),
                        "usage": usage_snapshot,
                    })

                # Agent is paused waiting for user input — send once, then loop
                # until we receive a valid "answer" message (ignore anything else
                # to prevent resending the same questions on unexpected messages).
                await websocket.send_json({
                    "type": "questions",
                    "questions": session.pending_questions,
                })
                while session.pending_questions is not None:
                    raw = await websocket.receive_text()
                    data = json.loads(raw)
                    if data.get("type") == "answer":
                        session._resolve_answers(data.get("answers", []))
                        # Agent will clear pending_questions after waking; give it
                        # a tick to do so before the outer loop re-checks.
                        await asyncio.sleep(0)
                        break
                    logger.debug(
                        "Unexpected WS message while waiting for answer: %r",
                        data.get("type"),
                    )

            elif session.pending_permission is not None:
                # Subagent wants to access a file outside cwd — ask user
                perm = session.pending_permission
                await websocket.send_json({
                    "type": "permission_request",
                    "filepath": perm["filepath"],
                    "operation": perm["operation"],
                })
                raw = await websocket.receive_text()
                data = json.loads(raw)
                if data.get("type") == "permission_response":
                    session._resolve_permission(bool(data.get("granted", False)))
                else:
                    # Unexpected message — deny by default
                    session._resolve_permission(False)

            else:
                # Nothing pending — wait for agent to signal questions ready
                # (or finish). Fall back to a short sleep if no event available.
                questions_event = getattr(session, "_questions_event", None)
                if questions_event is not None:
                    waiter = asyncio.ensure_future(questions_event.wait())
                    done, _ = await asyncio.wait(
                        {agent_task, waiter},
                        timeout=0.5,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if waiter not in done:
                        waiter.cancel()
                        try:
                            await waiter
                        except asyncio.CancelledError:
                            pass
                else:
                    await asyncio.sleep(0.05)
    except Exception:
        agent_task.cancel()
        raise

    response_msg = agent_task.result()
    return _msg_text(response_msg)


async def _check_and_send_topology(
    websocket: WebSocket,
    session: Session,
    response_text: str,
) -> bool:
    """
    Check if generate_topology was called this turn. If so, send a plan
    message to the client.

    generate_topology stores results directly on the session (inside the agent
    task) to avoid cross-task ContextVar isolation issues. We just read the flag.

    Returns True if a topology was found, False otherwise.
    """
    if not session.topology_just_generated:
        return False

    session.topology_just_generated = False

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
    session.topology_just_generated = False
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


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

async def handle_websocket(
    websocket: WebSocket,
    session: Session,
    daap_memory=None,
    topology_store=None,
    persist_fn=None,        # callable(session_id: str) -> None; called after state changes
) -> None:
    """
    Handle a full WebSocket conversation with the master agent.

    The caller (routes.py) is responsible for verifying the session exists
    and has a master_agent before calling this function.

    daap_memory: optional DaapMemory instance for post-run writes.
    persist_fn:  optional callable to flush session state to persistent store
                 after each turn and execution.
    """
    await websocket.accept()

    # Expose WebSocket send to agent tools (e.g. execute_pending_topology)
    # so they can stream progress without the handler needing to drive it.
    session._ws_send = websocket.send_json

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")

            # ------------------------------------------------------------------
            if msg_type == "message":
                user_text = data.get("content", "")
                session.conversation.append({"role": "user", "content": user_text})

                # Allow make_cheaper / cancel as convenience text shortcuts.
                # Approval is now handled by the agent via execute_pending_topology tool.
                if session.pending_topology is not None:
                    cmd = _detect_text_command(user_text)
                    if cmd is not None:
                        action, _ = cmd

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

                session.topology_just_generated = False

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

                if persist_fn is not None:
                    persist_fn(session.session_id)

            # ------------------------------------------------------------------
            elif msg_type == "make_cheaper":
                await _run_make_cheaper_flow(websocket, session)
                if persist_fn is not None:
                    persist_fn(session.session_id)

            # ------------------------------------------------------------------
            elif msg_type == "cancel":
                session.pending_topology = None
                session.pending_estimate = None
                await websocket.send_json({
                    "type": "response",
                    "content": "Cancelled. How else can I help?",
                })
                if persist_fn is not None:
                    persist_fn(session.session_id)

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
