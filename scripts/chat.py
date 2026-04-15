"""
DAAP CLI Chat — real-time terminal interface to the master agent.

Usage:
    python scripts/chat.py [--raw-output]

Commands during chat:
    raw          — show raw model output (including JSON)
    clean        — hide raw JSON/code blocks (default)
    help         — show command help
    approve      — approve latest topology plan
  cheaper       — ask agent to make topology cheaper
  cancel        — cancel pending topology
  quit / exit   — end session

Requires: OPENROUTER_API_KEY in .env or environment.
"""

import argparse
import asyncio
import json
import re
import shutil
import sys
import os
import textwrap

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

if not sys.stdout.isatty():
    RESET = BOLD = CYAN = GREEN = YELLOW = RED = DIM = ""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAAP terminal chat")
    parser.add_argument(
        "--raw-output",
        action="store_true",
        help="Show raw model output (including JSON blobs and fenced blocks).",
    )
    return parser.parse_args()


def _terminal_width() -> int:
    cols = shutil.get_terminal_size((100, 24)).columns
    return max(72, min(cols, 140))


def _print_header(raw_output: bool):
    mode = "RAW" if raw_output else "CLEAN"
    print(f"\n{BOLD}DAAP CLI{RESET} {DIM}[{mode} mode]{RESET}")
    print(f"{DIM}{'-' * min(_terminal_width(), 100)}{RESET}")
    print(f"{DIM}Commands: /help /approve /cheaper /cancel /raw /clean /quit{RESET}\n")


def _print_wrapped(text: str, indent: int = 2):
    width = _terminal_width() - indent
    prefix = " " * indent
    for paragraph in text.splitlines() or [""]:
        line = paragraph.rstrip()
        if not line:
            print("")
            continue
        wrapped = textwrap.wrap(
            line,
            width=max(40, width),
            replace_whitespace=False,
            drop_whitespace=False,
        )
        if not wrapped:
            print(prefix)
            continue
        for part in wrapped:
            print(f"{prefix}{part}")


def _looks_like_json_blob(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 2:
        return False
    if (stripped[0], stripped[-1]) not in {("{", "}"), ("[", "]")}:
        return False
    try:
        json.loads(stripped)
        return True
    except Exception:
        return False


def _summarize_json_blob(text: str) -> str:
    try:
        payload = json.loads(text)
    except Exception:
        return "Structured data generated."

    if isinstance(payload, dict):
        keys = ", ".join(list(payload.keys())[:5])
        return (
            f"Structured data generated ({keys}). "
            f"Use /raw to inspect exact JSON."
        )

    if isinstance(payload, list):
        return (
            f"Structured list generated ({len(payload)} item(s)). "
            f"Use /raw to inspect exact JSON."
        )

    return "Structured data generated."


def _sanitize_agent_text(text: str, raw_output: bool) -> str:
    cleaned = (text or "").replace("\r\n", "\n").strip()
    if not cleaned:
        return "Done."
    if raw_output:
        return cleaned

    def _replace_fence(match: re.Match) -> str:
        block = match.group(1).strip()
        if _looks_like_json_blob(block):
            return _summarize_json_blob(block)
        return block

    cleaned = _JSON_FENCE_RE.sub(_replace_fence, cleaned)

    if _looks_like_json_blob(cleaned):
        return _summarize_json_blob(cleaned)

    # Catch common Python-dict style blobs that are not strict JSON.
    if (
        len(cleaned) > 180
        and ("'nodes'" in cleaned or '"nodes"' in cleaned)
        and ("'edges'" in cleaned or '"edges"' in cleaned)
        and "{" in cleaned
        and "}" in cleaned
    ):
        return "Topology data generated. Use /raw to inspect exact structure."

    return re.sub(r"\n{3,}", "\n\n", cleaned)


def _is_auth_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 401:
        return True

    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 401:
        return True

    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return (
        "authenticationerror" in name
        or "invalid api key" in msg
        or "user not found" in msg
        or "unauthorized" in msg
    )


def _format_turn_error(exc: Exception) -> str:
    if _is_auth_error(exc):
        return (
            "Authentication failed (401). OpenRouter rejected the current API key. "
            "Set OPENROUTER_API_KEY to a valid key and try again."
        )

    name = type(exc).__name__
    return f"Request failed ({name}): {exc}"


def _close_memory_client(daap_memory) -> None:
    """Best-effort close to avoid noisy client destructor errors at shutdown."""
    if daap_memory is None:
        return

    mem_client = getattr(daap_memory, "mem", None)
    if mem_client is None:
        return

    close_fn = getattr(mem_client, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass


def _print_agent(text: str, raw_output: bool):
    final_text = _sanitize_agent_text(text, raw_output=raw_output)
    print(f"\n{CYAN}{BOLD}daap{RESET}")
    _print_wrapped(final_text, indent=2)
    print("")

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
    flow = " -> ".join(nodes) if nodes else "(no nodes)"
    print(f"\n{YELLOW}{BOLD}plan{RESET}")
    _print_wrapped(f"Flow     : {flow}", indent=2)
    _print_wrapped(f"Est cost : ${cost:.4f}", indent=2)
    _print_wrapped(f"Est time : {lat:.0f}s", indent=2)
    if usage:
        models = ", ".join(usage.get("models_used", []))
        _print_wrapped(f"Planning : {usage.get('total_tokens', 0)} tokens ({models})", indent=2)
    print(f"\n{YELLOW}Next: /approve | /cheaper | /cancel{RESET}\n")

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
    parts: list[str] = []

    def _walk(value):
        if value is None:
            return
        if isinstance(value, str):
            if value.strip():
                parts.append(value)
            return
        if isinstance(value, dict):
            if value.get("type") == "text" and isinstance(value.get("text"), str):
                txt = value.get("text", "")
                if txt.strip():
                    parts.append(txt)
                return
            if "content" in value:
                _walk(value.get("content"))
                return
            if isinstance(value.get("text"), str):
                txt = value.get("text", "")
                if txt.strip():
                    parts.append(txt)
                return
            for nested in value.values():
                _walk(nested)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                _walk(item)
            return

        text_attr = getattr(value, "text", None)
        if isinstance(text_attr, str) and text_attr.strip():
            parts.append(text_attr)
            return

        content_attr = getattr(value, "content", None)
        if content_attr is not None:
            _walk(content_attr)
            return

    _walk(content)
    if parts:
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
    args = _parse_args()
    raw_output = bool(args.raw_output)
    _print_header(raw_output)

    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        _print_warn("OPENROUTER_API_KEY is not set. API requests will fail until you set it.")

    # Init session
    session_mgr = SessionManager()
    session = session_mgr.create_session()
    topology_store = TopologyStore()
    session.token_tracker = TokenTracker()

    # Optional memory — disabled gracefully if credentials missing
    daap_memory = None
    user_context = None
    try:
        from daap.memory.client import DaapMemory
        from daap.memory.reader import load_user_context_for_master
        daap_memory = DaapMemory(mode="ephemeral")
        user_context = load_user_context_for_master(daap_memory, session.user_id)
    except Exception:
        pass  # memory is optional — never block startup

    rl_optimizer = None
    try:
        from daap.rl.optimizer import TopologyOptimizer
        rl_optimizer = TopologyOptimizer()
    except Exception:
        pass

    toolkit = create_session_scoped_toolkit(
        session,
        topology_store=topology_store,
        daap_memory=daap_memory,
        rl_optimizer=rl_optimizer,
    )
    session.master_agent = create_master_agent_with_toolkit(
        toolkit,
        user_context=user_context,
        tracker=session.token_tracker,
    )
    from daap.spec.resolver import MODEL_REGISTRY
    _print_system(f"Session {session.session_id} ready | Master: {MODEL_REGISTRY['powerful']}")

    try:
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

            # ------------------------------------------------------------------
            # Built-in commands
            if cmd in ("quit", "exit", "bye"):
                print("Bye!")
                break

            if cmd == "help":
                _print_system("Commands: /help /approve /cheaper /cancel /raw /clean /quit")
                continue

            if cmd == "raw":
                raw_output = True
                _print_system("Raw output enabled.")
                continue

            if cmd == "clean":
                raw_output = False
                _print_system("Clean output enabled.")
                continue

            if cmd == "approve":
                user_input = "Proceed with this topology and continue."

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
            try:
                response_text, usage = await _run_agent_turn(session)
            except Exception as exc:
                _print_error(_format_turn_error(exc))
                if _is_auth_error(exc):
                    _print_system("Tip: verify OPENROUTER_API_KEY in your shell or .env and restart if needed.")
                clear_last_topology_result()
                continue

            session.conversation.append({"role": "assistant", "content": response_text})

            # Check if topology was generated this turn
            topo_result = get_last_topology_result()
            if topo_result.get("topology") is not None:
                session.pending_topology = topo_result["topology"]
                session.pending_estimate = topo_result["estimate"]
                clear_last_topology_result()
                _print_agent(response_text, raw_output=raw_output)
                _print_plan(session.pending_topology, session.pending_estimate, usage)
            else:
                _print_agent(response_text, raw_output=raw_output)
                _print_usage(usage)
    finally:
        _close_memory_client(daap_memory)


if __name__ == "__main__":
    asyncio.run(main())
