"""
DAAP CLI Chat — real-time terminal interface to the master agent.

Usage:
  python scripts/chat.py

Commands during chat:
  approve       — approve the pending topology and execute it
  cheaper       — ask agent to make topology cheaper
  cancel        — cancel pending topology
  quit / exit   — end session

Requires: OPENROUTER_API_KEY in .env or environment.
"""

import asyncio
import json
import sys
import os

# Force UTF-8 output — prevents cp1252 crash on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
except ImportError:
    pass

from agentscope.message import Msg
from daap.api.sessions import Session, SessionManager, create_session_scoped_toolkit
from daap.master.agent import create_master_agent_with_toolkit
from daap.master.tools import clear_last_topology_result, get_last_topology_result
from daap.spec.schema import TopologySpec
from daap.spec.resolver import resolve_topology
from daap.executor.engine import execute_topology
from daap.topology.store import TopologyStore
from daap.tools.token_tracker import TokenTracker


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"

def _print_agent(text: str):
    print(f"\n{CYAN}{BOLD}DAAP:{RESET} {text}\n")

def _print_system(text: str):
    print(f"{DIM}[{text}]{RESET}")

def _print_success(text: str):
    print(f"\n{GREEN}{BOLD}{text}{RESET}\n")

def _print_warn(text: str):
    print(f"\n{YELLOW}{text}{RESET}\n")

def _print_error(text: str):
    print(f"\n{RED}{text}{RESET}\n")

def _print_plan(topology: dict, estimate: dict, usage: dict):
    nodes = [n.get("node_id") for n in topology.get("nodes", [])]
    cost  = estimate.get("total_cost_usd", 0) if estimate else 0
    lat   = estimate.get("total_latency_seconds", 0) if estimate else 0
    print(f"\n{YELLOW}{BOLD}== TOPOLOGY PLAN =={RESET}")
    print(f"  Nodes    : {' -> '.join(nodes)}")
    print(f"  Est cost : ${cost:.4f}")
    print(f"  Est time : {lat:.0f}s")
    if usage:
        models = ", ".join(usage.get("models_used", []))
        print(f"  Planning : {usage.get('total_tokens', 0)} tokens ({models})")
    print(f"\n{YELLOW}Commands: approve | cheaper | cancel{RESET}\n")

def _print_result(output: str, latency: float, models: list, usage: dict):
    print(f"\n{GREEN}{BOLD}== EXECUTION RESULT =={RESET}")
    print(f"{output}")
    print(f"\n{DIM}Latency: {latency:.1f}s | Models: {', '.join(models)}")
    if usage:
        print(f"Tokens: {usage.get('input_tokens',0)} in / {usage.get('output_tokens',0)} out / {usage.get('total_tokens',0)} total{RESET}")

def _print_usage(usage: dict):
    if not usage or not usage.get("total_tokens"):
        return
    models = ", ".join(usage.get("models_used", []))
    print(f"{DIM}  [{usage.get('total_tokens',0)} tokens | {models}]{RESET}")

def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# ask_user: prompt user in terminal
# ---------------------------------------------------------------------------

async def _handle_ask_user(session: Session) -> None:
    """Wait for pending_questions, then prompt user interactively."""
    while session.pending_questions is None:
        await asyncio.sleep(0.05)

    questions = session.pending_questions
    print(f"\n{YELLOW}{BOLD}DAAP needs more info:{RESET}")
    answers = []

    for i, q in enumerate(questions, 1):
        print(f"\n  {BOLD}Q{i}:{RESET} {q.get('question', '')}")
        options = q.get("options", [])
        if options:
            for j, opt in enumerate(options):
                label = opt.get("label", "")
                desc  = opt.get("description", "")
                marker = " (Recommended)" if "Recommended" in label else ""
                print(f"    {j+1}. {label}{marker}" + (f" — {desc}" if desc else ""))
            print(f"    {DIM}Enter number or type your answer:{RESET} ", end="", flush=True)
        else:
            print(f"    {DIM}Your answer:{RESET} ", end="", flush=True)

        raw = await asyncio.get_event_loop().run_in_executor(None, input, "")
        raw = raw.strip()

        # If numeric, map to option label
        if options and raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                raw = options[idx]["label"]

        answers.append(raw)

    session._resolve_answers(answers)


# ---------------------------------------------------------------------------
# Agent turn runner
# ---------------------------------------------------------------------------

async def _run_agent_turn(session: Session) -> tuple[str, dict]:
    """
    Run one agent turn. Handles ask_user questions interactively.
    Returns (response_text, usage_dict).
    """
    tracker = session.token_tracker
    if tracker:
        tracker.reset()

    # Peek at last message from conversation to send to agent
    last_user = next(
        (m["content"] for m in reversed(session.conversation) if m["role"] == "user"),
        "",
    )
    msg = Msg(name="user", content=last_user, role="user")
    agent_task = asyncio.create_task(session.master_agent(msg))

    # Pump ask_user questions while agent runs
    while not agent_task.done():
        if session.pending_questions is not None:
            await _handle_ask_user(session)
        await asyncio.sleep(0.05)

    response_msg = agent_task.result()
    text = _extract_text(response_msg.content)
    usage = tracker.to_dict() if tracker else {}
    return text, usage


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

async def main():
    print(f"\n{BOLD}DAAP Sales Automation CLI{RESET}")
    print("Type your request, or 'quit' to exit.")
    print(f"{DIM}--------------------------------------------------{RESET}\n")

    # Init session
    session_mgr = SessionManager()
    session = session_mgr.create_session()
    topology_store = TopologyStore()
    session.token_tracker = TokenTracker()
    toolkit = create_session_scoped_toolkit(session, topology_store=topology_store)
    session.master_agent = create_master_agent_with_toolkit(
        toolkit,
        tracker=session.token_tracker,
    )
    _print_system(f"Session {session.session_id} ready | Model: google/gemini-2.0-flash-001")

    while True:
        # Get user input
        try:
            print(f"{BOLD}You:{RESET} ", end="", flush=True)
            user_input = await asyncio.get_event_loop().run_in_executor(None, input, "")
            user_input = user_input.strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower().strip().lstrip("/")

        # Fuzzy approve detection — catches typos and natural language
        _approve_triggers = (
            "approve", "approv", "yes run", "yes execute", "execute it",
            "run it", "go ahead", "proceed", "i approve", "looks good",
            "let's go", "lets go", "do it", "start execution", "run the topology",
        )
        _is_approve = any(t in cmd for t in _approve_triggers)

        # ------------------------------------------------------------------
        # Built-in commands
        if cmd in ("quit", "exit", "bye"):
            print("Bye!")
            break

        if _is_approve:
            if session.pending_topology is None:
                _print_warn("No topology pending. Ask me to build one first.")
                continue

            topo_id = session.pending_topology.get("topology_id", "?")
            user_prompt = session.pending_topology.get("user_prompt", "")
            _print_system(f"Executing topology {topo_id}...")

            try:
                topology_spec = TopologySpec.model_validate(session.pending_topology)
                resolved = resolve_topology(topology_spec)
                if isinstance(resolved, list):
                    errors = "; ".join(e.message for e in resolved)
                    _print_error(f"Resolution failed: {errors}")
                    continue

                total_steps = len(resolved.execution_order)

                def _on_node_start(node_id, model_id, step_num, total):
                    print(f"{DIM}[{step_num}/{total}] Running {BOLD}{node_id}{RESET}{DIM} ({model_id})...{RESET}", flush=True)

                def _on_node_complete(nr):
                    tok = nr.input_tokens + nr.output_tokens
                    print(f"{GREEN}[done] {nr.node_id} — {nr.latency_seconds:.1f}s | {tok} tokens{RESET}", flush=True)

                result = await execute_topology(
                    resolved=resolved,
                    user_prompt=user_prompt,
                    tracker=session.token_tracker,
                    on_node_start=_on_node_start,
                    on_node_complete=_on_node_complete,
                )

                session.pending_topology = None
                session.pending_estimate = None

                # Auto-save topology + run history (non-fatal on storage errors).
                try:
                    saved = topology_store.save_topology(
                        spec=topology_spec.model_dump(),
                        user_id=session.user_id,
                        overwrite=True,
                    )
                    topology_store.save_run(
                        topology_id=saved.topology_id,
                        topology_version=saved.version,
                        user_id=session.user_id,
                        result={
                            "topology_id": result.topology_id,
                            "final_output": result.final_output,
                            "success": result.success,
                            "error": result.error,
                            "latency_seconds": result.total_latency_seconds,
                            "total_input_tokens": result.total_input_tokens,
                            "total_output_tokens": result.total_output_tokens,
                        },
                        user_prompt=user_prompt,
                    )
                except Exception as _save_exc:
                    _print_warn(f"Auto-save failed (run still completed): {_save_exc}")

                if result.success:
                    # Inject result into conversation so agent can discuss it
                    summary = (
                        f"[Execution complete]\n"
                        f"Topology ran successfully in {result.total_latency_seconds:.1f}s.\n"
                        f"Nodes: {', '.join(nr.node_id for nr in result.node_results)}\n"
                        f"Tokens used: {result.total_input_tokens} in / {result.total_output_tokens} out\n"
                        f"Final output:\n{result.final_output}"
                    )
                    session.conversation.append({"role": "user", "content": summary})
                    session.conversation.append({"role": "assistant", "content": "Execution complete. Here are the results above."})

                if result.success:
                    _print_result(
                        result.final_output,
                        result.total_latency_seconds,
                        result.models_used,
                        {
                            "input_tokens": result.total_input_tokens,
                            "output_tokens": result.total_output_tokens,
                            "total_tokens": result.total_input_tokens + result.total_output_tokens,
                        },
                    )
                else:
                    _print_error(f"Execution failed: {result.error}")
            except Exception as exc:
                _print_error(f"Error: {exc}")
            continue

        if cmd == "cheaper":
            user_input = "Make the topology cheaper. Reduce cost while keeping it functional."

        if cmd == "cancel":
            session.pending_topology = None
            session.pending_estimate = None
            _print_system("Topology cancelled.")
            continue

        # ------------------------------------------------------------------
        # Normal message to agent
        clear_last_topology_result()
        session.conversation.append({"role": "user", "content": user_input})

        _print_system("Thinking...")
        response_text, usage = await _run_agent_turn(session)
        session.conversation.append({"role": "assistant", "content": response_text})

        # Check if topology was generated this turn
        topo_result = get_last_topology_result()
        if topo_result.get("topology") is not None:
            session.pending_topology = topo_result["topology"]
            session.pending_estimate = topo_result["estimate"]
            clear_last_topology_result()
            _print_agent(response_text)
            _print_plan(session.pending_topology, session.pending_estimate, usage)
        else:
            _print_agent(response_text)
            _print_usage(usage)


if __name__ == "__main__":
    asyncio.run(main())
