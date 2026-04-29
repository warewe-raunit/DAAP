import sys
code = '''def _cancel_active_task(session) -> None:
    if getattr(session, "active_task", None) and not session.active_task.done():
        session.active_task.cancel()

async def _state_monitor(session) -> None:
    """Background task to push pending state (questions/permissions) to the WebSocket."""
    import asyncio
    try:
        while True:
            # 1. Questions
            if session.pending_questions is not None and getattr(session, "_questions_event", None) and session._questions_event.is_set():
                if getattr(session, "topology_just_generated", False):
                    session.topology_just_generated = False
                    est = getattr(session, "pending_estimate", {}) or {}
                    topo = getattr(session, "pending_topology", {}) or {}
                    raw_nodes = [n for n in topo.get("nodes", []) if isinstance(n, dict)]
                    node_parts = []
                    for n in raw_nodes:
                        nid = n.get("node_id", "")
                        instances = n.get("instance_config", {}).get("parallel_instances", 1)
                        node_parts.append(f"{nid} ×{instances}" if instances > 1 else nid)
                    tracker = getattr(session, "token_tracker", None)
                    usage_snapshot = tracker.to_dict() if tracker else {}
                    if getattr(session, "_ws_send", None):
                        await session._ws_send({
                            "type": "plan",
                            "summary": f"{len(node_parts)} node(s): {', '.join(node_parts)}",
                            "topology": topo,
                            "cost_usd": est.get("total_cost_usd", 0),
                            "latency_seconds": est.get("total_latency_seconds", 0),
                            "min_cost_usd": est.get("min_viable_cost_usd", 0),
                            "usage": usage_snapshot,
                        })

                if getattr(session, "_ws_send", None):
                    await session._ws_send({
                        "type": "questions",
                        "questions": session.pending_questions,
                    })
                session._questions_event.clear()

            # 2. Permissions
            if hasattr(session, "_perm_sent") and getattr(session, "_perm_sent"):
                if session.pending_permission is None:
                    session._perm_sent = False
            elif session.pending_permission is not None:
                perm = session.pending_permission
                if getattr(session, "_ws_send", None):
                    await session._ws_send({
                        "type": "permission_request",
                        "filepath": perm.get("filepath", ""),
                        "operation": perm.get("operation", ""),
                    })
                session._perm_sent = True

            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        pass


async def _run_agent_turn(
    session,
    user_text: str,
    persist_fn=None,
) -> None:
    """Run a single master agent turn in the background."""
    import asyncio
    try:
        from daap.master.agent import _inject_plan_hint
        from daap.master.planner import plan_turn
        from agentscope.message import Msg

        _model = getattr(session.master_agent, "_daap_model", None)
        _formatter = getattr(session.master_agent, "_daap_formatter", None)
        if _model is not None and _formatter is not None:
            _plan = await plan_turn(user_text, _model, _formatter)
            _inject_plan_hint(session.master_agent, _plan)

        agent_msg = Msg(name="user", content=user_text, role="user")
        response_msg = await session.master_agent(agent_msg)
        response_text = _msg_text(response_msg)

        session.conversation.append({"role": "assistant", "content": response_text})

        if getattr(session, "topology_just_generated", False):
            session.topology_just_generated = False
            est = getattr(session, "pending_estimate", {}) or {}
            tracker = getattr(session, "token_tracker", None)
            usage = tracker.to_dict() if tracker else {}
            if getattr(session, "_ws_send", None):
                await session._ws_send({
                    "type": "plan",
                    "summary": response_text,
                    "cost_usd": est.get("total_cost_usd", 0),
                    "latency_seconds": est.get("total_latency_seconds", 0),
                    "min_cost_usd": est.get("min_viable_cost_usd", 0),
                    "usage": usage,
                })
        else:
            tracker = getattr(session, "token_tracker", None)
            usage = tracker.to_dict() if tracker else {}
            if getattr(session, "_ws_send", None):
                await session._ws_send({
                    "type": "response",
                    "content": response_text,
                    "usage": usage,
                })

        if persist_fn is not None:
            persist_fn(session.session_id)
            
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        if getattr(session, "_ws_send", None):
            await session._ws_send({"type": "error", "message": str(exc)})
    finally:
        session.active_task = None


async def _run_pending_topology_direct_task(session, persist_fn=None) -> None:
    """Execute the pending topology in the background."""
    import asyncio
    try:
        toolkit = getattr(session.master_agent, "_daap_toolkit", None)
        tool = getattr(toolkit, "tools", {}).get("execute_pending_topology") if toolkit else None
        execute_fn = getattr(tool, "original_func", None)
        if not callable(execute_fn):
            raise RuntimeError("execute_pending_topology tool is not available on this session.")

        result = await execute_fn()
        response_text = _tool_response_text(result)

        session.conversation.append({
            "role": "assistant",
            "content": response_text or "Execution finished.",
        })
        if session.execution_result is None and response_text:
            if getattr(session, "_ws_send", None):
                await session._ws_send({
                    "type": "response",
                    "content": response_text,
                })
        if persist_fn is not None:
            persist_fn(session.session_id)
            
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        if getattr(session, "_ws_send", None):
            await session._ws_send({"type": "error", "message": str(exc)})
    finally:
        session.active_task = None


async def _run_make_cheaper_flow_task(session, persist_fn=None) -> None:
    import asyncio
    try:
        session.topology_just_generated = False
        cheaper_prompt = "Make the topology cheaper. Reduce cost while keeping it functional."
        session.conversation.append({"role": "user", "content": cheaper_prompt})

        tracker = getattr(session, "token_tracker", None)
        if tracker:
            tracker.reset()

        await _run_agent_turn(session, cheaper_prompt, persist_fn)
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

async def handle_websocket(
    websocket: WebSocket,
    session,
    daap_memory=None,
    topology_store=None,
    persist_fn=None,
) -> None:
    import asyncio
    await websocket.accept()
    session._ws_send = websocket.send_json
    session.active_task = None
    
    monitor_task = asyncio.create_task(_state_monitor(session))
    
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "message":
                user_text = data.get("content", "")
                
                cmd = _detect_text_command(user_text)
                if session.pending_topology is not None and cmd is not None:
                    action, _ = cmd
                    if action == "approve":
                        if session.pending_questions and getattr(session, "_resolve_answers", None):
                            session._resolve_answers(["yes"])
                        else:
                            _cancel_active_task(session)
                            session.topology_just_generated = False
                            session.active_task = asyncio.create_task(_run_pending_topology_direct_task(session, persist_fn))
                        continue
                    elif action == "cancel":
                        if session.pending_questions and getattr(session, "_resolve_answers", None):
                            session._resolve_answers(["cancel"])
                        else:
                            _cancel_active_task(session)
                            session.pending_topology = None
                            session.pending_estimate = None
                            await websocket.send_json({"type": "response", "content": "Cancelled. How else can I help?"})
                        continue
                    elif action == "make_cheaper":
                        if session.pending_questions and getattr(session, "_resolve_answers", None):
                            session._resolve_answers(["make cheaper"])
                        else:
                            _cancel_active_task(session)
                            session.active_task = asyncio.create_task(_run_make_cheaper_flow_task(session, persist_fn))
                        continue
                
                if getattr(session, "pending_questions", None):
                    if getattr(session, "_resolve_answers", None):
                        session._resolve_answers([user_text])
                    continue
                
                session.conversation.append({"role": "user", "content": user_text})
                
                _cancel_active_task(session)
                tracker = getattr(session, "token_tracker", None)
                if tracker:
                    tracker.reset()
                
                session.active_task = asyncio.create_task(_run_agent_turn(session, user_text, persist_fn))
                
            elif msg_type == "answer":
                if getattr(session, "_resolve_answers", None):
                    session._resolve_answers(data.get("answers", []))
                    
            elif msg_type == "permission_response":
                if getattr(session, "_resolve_permission", None):
                    session._resolve_permission(bool(data.get("granted", False)))
                    
            elif msg_type == "make_cheaper":
                if getattr(session, "pending_questions", None) and getattr(session, "_resolve_answers", None):
                    session._resolve_answers(["make cheaper"])
                else:
                    _cancel_active_task(session)
                    session.active_task = asyncio.create_task(_run_make_cheaper_flow_task(session, persist_fn))
                    if persist_fn is not None:
                        persist_fn(session.session_id)
                        
            elif msg_type == "cancel":
                if getattr(session, "pending_questions", None) and getattr(session, "_resolve_answers", None):
                    session._resolve_answers(["cancel"])
                else:
                    _cancel_active_task(session)
                    session.pending_topology = None
                    session.pending_estimate = None
                    await websocket.send_json({"type": "response", "content": "Cancelled. How else can I help?"})
                    if persist_fn is not None:
                        persist_fn(session.session_id)
                        
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
    finally:
        monitor_task.cancel()
        _cancel_active_task(session)
'''
with open('c:/Users/aman/DAAP/daap/api/ws_handler.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
with open('c:/Users/aman/DAAP/daap/api/ws_handler.py', 'w', encoding='utf-8') as f:
    f.writelines(lines[:120])
    f.write(code)
