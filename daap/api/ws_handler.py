"""
DAAP WebSocket Handler — manages the master agent conversation loop over WebSocket.

Message protocol (JSON over WebSocket):

Client → Server:
  {"type": "message", "content": "..."}          user chat message
  {"type": "answer", "answers": ["..."]}          answers to ask_user questions
  {"type": "make_cheaper"}                        request cheaper topology
  {"type": "cancel"}                              cancel pending topology

Server → Client:
  {"type": "response", "content": "..."}          agent text response
  {"type": "questions", "questions": [...]}        structured ask_user questions
  {"type": "plan", "summary": "...", ...}          topology plan awaiting approval
  {"type": "executing", "topology_id": "..."}      execution started (sent by agent tool)
  {"type": "result", "output": "...", ...}         execution complete (sent by agent tool)
  {"type": "error", "message": "..."}              error
"""

import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect
from agentscope.message import Msg

from daap.api.sessions import Session
from daap.master.tools import clear_last_topology_result, get_last_topology_result

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
    return msg.content if isinstance(msg.content, str) else str(msg.content)


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
