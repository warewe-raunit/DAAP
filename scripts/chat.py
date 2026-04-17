"""
DAAP CLI Chat — real-time terminal interface to the master agent.

Usage:
    python scripts/chat.py [--raw-output] [--session ID]
    python scripts/chat.py --api-url http://107.174.35.26:8000   # remote server

Commands during chat:
    /help        — show command help
    /approve     — approve latest topology plan
    /cheaper     — ask agent to make topology cheaper
    /cancel      — cancel pending topology
    /sessions    — list all saved sessions (resume with --session <id>)
    /history     — list saved topologies
    /topology    — list topologies; /topology load <id> to reload one
    /memory      — list profile facts; search/delete subcommands
    /clear       — clear conversation history (start fresh)
    /profile     — show user profile + optimizer summary
    /mcp         — show connected MCP servers and tools
    /skills      — list registered skills
    /skill       — manage skills (add|remove|create)
    /raw         — show raw model output (including JSON)
    /clean       — hide raw JSON/code blocks (default)
    /quit        — end session

Requires: OPENROUTER_API_KEY in .env or environment.
"""

import argparse
import asyncio
import json
import pathlib
import re
import shlex
import shutil
import sys
import os
import textwrap
import time

# Force UTF-8 output — prevents cp1252 crash on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from daap.env import load_project_env

load_project_env()

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.patch_stdout import patch_stdout

from agentscope.message import Msg
from daap.api.sessions import Session, SessionManager, create_session_scoped_toolkit
from daap.identity import load_local_user, resolve_cli_user
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

if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    RESET = BOLD = CYAN = GREEN = YELLOW = RED = DIM = ""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAAP terminal chat")
    parser.add_argument(
        "--raw-output",
        action="store_true",
        help="Show raw model output (including JSON blobs and fenced blocks).",
    )
    parser.add_argument(
        "--session", "-s",
        metavar="SESSION_ID",
        default=None,
        help="Resume a previous session by ID (e.g. python chat.py --session abc123).",
    )
    parser.add_argument(
        "--api-url",
        metavar="URL",
        default=None,
        help="Connect to a remote DAAP server instead of running locally. "
             "Example: --api-url http://107.174.35.26:8000",
    )
    return parser.parse_args()


def _terminal_width() -> int:
    cols = shutil.get_terminal_size((100, 24)).columns
    return max(72, min(cols, 140))


def _print_header(raw_output: bool):
    mode = "RAW" if raw_output else "CLEAN"
    print(f"\n{BOLD}DAAP CLI{RESET} {DIM}[{mode} mode]{RESET}")
    print(f"{DIM}{'-' * min(_terminal_width(), 100)}{RESET}")
    print(f"{DIM}Commands: /help /approve /cheaper /cancel /history /memory /topology /mcp /skills /skill /profile /raw /clean /quit{RESET}\n")


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


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

def _sessions_dir(user_id: str) -> pathlib.Path:
    p = pathlib.Path.home() / ".daap" / "sessions" / user_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_session(user_id: str, session_id: str, conversation: list[dict]) -> None:
    try:
        d = _sessions_dir(user_id)
        payload = json.dumps(
            {"session_id": session_id, "saved_at": time.time(), "conversation": conversation},
            ensure_ascii=False,
        )
        # Write named file + update "last" pointer
        (d / f"{session_id}.json").write_text(payload, encoding="utf-8")
        (d / "last.json").write_text(payload, encoding="utf-8")
    except Exception:
        pass


def _load_session(user_id: str, session_id: str | None = None) -> tuple[str | None, list[dict]]:
    """Return (session_id, conversation). session_id=None if nothing found."""
    try:
        d = _sessions_dir(user_id)
        if session_id:
            # Try exact match first, then prefix match
            exact = d / f"{session_id}.json"
            if exact.exists():
                f = exact
            else:
                matches = sorted(d.glob(f"{session_id}*.json"))
                matches = [m for m in matches if m.name != "last.json"]
                if not matches:
                    return None, []
                f = matches[0]
        else:
            f = d / "last.json"
            if not f.exists():
                return None, []
        data = json.loads(f.read_text(encoding="utf-8"))
        return data.get("session_id"), data.get("conversation", [])
    except Exception:
        return None, []


def _list_sessions(user_id: str) -> list[dict]:
    """List all saved sessions for a user, newest first."""
    try:
        d = _sessions_dir(user_id)
        sessions = []
        for f in d.glob("*.json"):
            if f.name == "last.json":
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id": data.get("session_id", f.stem),
                    "saved_at": data.get("saved_at", 0),
                    "messages": len(data.get("conversation", [])),
                })
            except Exception:
                pass
        return sorted(sessions, key=lambda s: s["saved_at"], reverse=True)
    except Exception:
        return []


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
        deduped: list[str] = []
        for part in parts:
            text = str(part or "").strip()
            if not text:
                continue
            normalized = re.sub(r"\s+", " ", text)
            prev_normalized = (
                re.sub(r"\s+", " ", deduped[-1].strip())
                if deduped
                else None
            )
            if normalized == prev_normalized:
                continue
            deduped.append(text)

        if deduped:
            return "\n".join(deduped)
        return ""
    return str(content)


def _suppress_agentscope_stdout(agent) -> None:
    """
    Disable AgentScope's built-in terminal printing.

    The CLI already renders responses via _print_agent; without this, output
    appears twice (AgentScope `DAAP: ...` + CLI rendering).
    """
    async def _quiet_print(*_args, **_kwargs):
        return None

    try:
        agent.print = _quiet_print
    except Exception:
        pass


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

        raw = await asyncio.get_running_loop().run_in_executor(None, input, "")
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
# Identity helpers
# ---------------------------------------------------------------------------

def _build_welcome_line(user_id: str, topology_store, profile_facts: list[str] | None = None) -> str:
    """Assemble 'Welcome back, X — N runs · optimizer active · M memory facts'."""
    display = user_id.replace("-", " ").title()
    segments: list[str] = []

    try:
        run_count = topology_store.count_runs(user_id)
        if run_count > 0:
            segments.append(f"{run_count} run{'s' if run_count != 1 else ''}")
    except Exception:
        pass

    try:
        from daap.optimizer.store import BanditStore
        summary = BanditStore().get_profile_summary(user_id)
        segments.append("optimizer active" if summary else "optimizer learning")
    except Exception:
        pass

    if profile_facts:
        segments.append(f"{len(profile_facts)} memory fact{'s' if len(profile_facts) != 1 else ''}")

    suffix = " · ".join(segments)
    return f"Welcome back, {display} — {suffix}" if suffix else f"Welcome back, {display}"


def _run_skill_create_wizard(print_fn) -> None:
    """Interactive wizard to create a new skill directory and register it."""
    from pathlib import Path

    from daap.skills.manager import SkillValidationError, get_skill_manager

    print_fn("Skill creator — press Ctrl+C to cancel")

    try:
        while True:
            name = input("Name: ").strip()
            if not name:
                print_fn("Name is required.")
                continue
            name = re.sub(r"[^a-zA-Z0-9-]", "-", name).strip("-").lower()
            if name:
                break
            print_fn("Invalid name. Use letters, numbers, and hyphens.")

        while True:
            description = input("Description (one line): ").strip()
            if description:
                break
            print_fn("Description is required.")

        targets_input = input("Targets [all/master/subagent, default=all]: ").strip().lower()
        targets = targets_input if targets_input in {"all", "master", "subagent"} else "all"

        default_dir = str(Path.home() / ".daap" / "skills" / name)
        dir_input = input(f"Save to dir [{default_dir}]: ").strip()
        save_dir = Path(dir_input if dir_input else default_dir).expanduser()

        print_fn("Skill body (enter '.' on its own line to finish):")
        body_lines: list[str] = []
        while True:
            line = input("> ")
            if line.strip() == ".":
                break
            body_lines.append(line)
        body = "\n".join(body_lines).strip()

        skill_md = f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"
        print_fn(f"Preview -> {save_dir}\\SKILL.md")
        print(skill_md)

        confirm = input("Write? [y/N]: ").strip().lower()
        if confirm != "y":
            print_fn("Cancelled.")
            return

        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

        manager = get_skill_manager()
        registered_name, _ = manager.add_skill(str(save_dir), targets=targets, persist=True)
        print_fn(f"Skill '{registered_name}' created and registered.")
    except KeyboardInterrupt:
        print()
        print_fn("Skill creation cancelled.")
    except SkillValidationError as exc:
        print_fn(f"Skill error: {exc}")
    except Exception as exc:
        print_fn(f"Skill creation failed: {exc}")


def _handle_skill_command(raw_input: str, print_fn) -> None:
    """Dispatch /skill subcommands: add, remove, create."""
    from daap.skills.manager import SkillValidationError, get_skill_manager

    command = raw_input.lstrip("/")
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()

    if len(parts) < 2:
        print_fn("Usage: /skill add <path> [master|subagent|all] | /skill remove <path> | /skill create")
        return

    sub = parts[1].lower()
    manager = get_skill_manager()

    if sub == "add":
        if len(parts) < 3:
            print_fn("Usage: /skill add <path> [master|subagent|all]")
            return
        directory = parts[2]
        targets = parts[3].lower() if len(parts) >= 4 else "all"
        if targets not in {"all", "master", "subagent"}:
            print_fn("Usage: /skill add <path> [master|subagent|all]")
            return
        try:
            name, was_new = manager.add_skill(directory, targets=targets, persist=True)
            if not was_new:
                print_fn(f"Skill '{name}' already registered.")
                return
            target_display = targets if targets != "all" else "master, subagent"
            print_fn(f"Skill '{name}' registered [{target_display}].")
        except SkillValidationError as exc:
            print_fn(f"Skill error: {exc}")
        except Exception as exc:
            print_fn(f"Skill error: {exc}")
        return

    if sub == "remove":
        if len(parts) < 3:
            print_fn("Usage: /skill remove <path>")
            return
        directory = parts[2]
        try:
            name = manager.remove_skill(directory, persist=True)
            print_fn(f"Skill '{name}' removed.")
        except KeyError:
            print_fn(f"Skill not found: {directory}")
        except Exception as exc:
            print_fn(f"Skill error: {exc}")
        return

    if sub == "create":
        _run_skill_create_wizard(print_fn)
        return

    print_fn(f"Unknown /skill subcommand: '{sub}'. Try: add, remove, create")


# ---------------------------------------------------------------------------
# CLI progress callback (replaces WebSocket progress for terminal)
# ---------------------------------------------------------------------------

def _make_cli_progress_send():
    """Return an async callable that prints node progress to the terminal."""
    async def _send(event: dict) -> None:
        etype = event.get("type")
        if etype == "executing":
            total = event.get("total_nodes", "?")
            print(f"{DIM}  [executing {total} nodes]{RESET}", flush=True)
        elif etype == "progress":
            ev = event.get("event", "")
            node_id = event.get("node_id", "?")
            done = event.get("completed_nodes", 0)
            total = event.get("total_nodes", 0)
            bar_width = 20
            filled = int((done / total) * bar_width) if total else 0
            bar = "█" * filled + "░" * (bar_width - filled)
            if ev == "node_start":
                print(f"\r{DIM}  [{bar}] {done}/{total}  running: {node_id}{RESET}    ", end="", flush=True)
            elif ev == "node_complete":
                done_now = done
                filled_now = int((done_now / total) * bar_width) if total else 0
                bar_now = "█" * filled_now + "░" * (bar_width - filled_now)
                if done_now >= total:
                    print(f"\r{DIM}  [{bar_now}] {done_now}/{total}  done{RESET}          ", flush=True)
                else:
                    print(f"\r{DIM}  [{bar_now}] {done_now}/{total}  done: {node_id}{RESET}    ", end="", flush=True)
    return _send


# ---------------------------------------------------------------------------
# Remote mode — WebSocket client against a DAAP server
# ---------------------------------------------------------------------------

def _ws_url(api_url: str, path: str) -> str:
    """Convert http(s):// base URL to ws(s):// WebSocket URL."""
    base = api_url.rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://"):] + path
    return "ws://" + base.removeprefix("http://") + path


def _remote_session_file(user_id: str, api_url: str) -> pathlib.Path:
    host = api_url.rstrip("/").split("//")[-1].replace(":", "_").replace("/", "_")
    p = pathlib.Path.home() / ".daap" / "sessions" / user_id / "remote"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{host}.json"


def _save_remote_session(user_id: str, api_url: str, session_id: str) -> None:
    try:
        _remote_session_file(user_id, api_url).write_text(
            json.dumps({"session_id": session_id, "api_url": api_url, "saved_at": time.time()}),
            encoding="utf-8",
        )
    except Exception:
        pass


def _load_remote_session(user_id: str, api_url: str) -> str | None:
    try:
        data = json.loads(_remote_session_file(user_id, api_url).read_text(encoding="utf-8"))
        return data.get("session_id")
    except Exception:
        return None


async def _remote_main(args: argparse.Namespace) -> None:
    import httpx
    import websockets

    api_url = args.api_url.rstrip("/")
    raw_output = bool(args.raw_output)
    _print_header(raw_output)

    # Resolve user identity
    is_new_user = load_local_user() is None
    user_id = await asyncio.get_running_loop().run_in_executor(None, resolve_cli_user)

    _print_system(f"Remote mode → {api_url}")

    # Try to reuse an existing server session, else create one
    existing_sid = args.session or _load_remote_session(user_id, api_url)
    session_id: str | None = None

    async with httpx.AsyncClient(timeout=10) as http:
        if existing_sid:
            try:
                r = await http.get(f"{api_url}/session/{existing_sid}/config")
                if r.status_code == 200:
                    session_id = existing_sid
                    _print_system(f"Session resumed: {session_id}")
            except Exception:
                pass

        if session_id is None:
            try:
                r = await http.post(f"{api_url}/session", params={"user_id": user_id})
                r.raise_for_status()
                session_id = r.json()["session_id"]
                _print_system(f"Session {session_id} ready")
                _print_system(f"Resume later: python scripts/chat.py --api-url {api_url} --session {session_id}")
            except Exception as exc:
                _print_error(f"Could not create server session: {exc}")
                _print_error(f"Is the server running at {api_url}?")
                return

    _save_remote_session(user_id, api_url, session_id)

    _commands = [
        "/help", "/approve", "/cheaper", "/cancel",
        "/raw", "/clean", "/quit",
    ]
    _completer = WordCompleter(_commands, sentence=True)
    _prompt_session: PromptSession = PromptSession(completer=_completer, complete_while_typing=True)

    ws_endpoint = _ws_url(api_url, f"/ws/{session_id}")
    _print_system(f"Connecting to {ws_endpoint} ...")

    try:
        async with websockets.connect(ws_endpoint, ping_interval=30) as ws:
            _print_system("Connected. Type your message or /help.")

            async def _recv_loop():
                """Background task: receive server events and render them."""
                nonlocal raw_output
                pending_plan: dict | None = None

                async for raw in ws:
                    event = json.loads(raw)
                    etype = event.get("type")

                    if etype == "response":
                        content = event.get("content", "")
                        # Server may send content as str(list) e.g. "[{'type':'text','text':'...'}]"
                        # Try ast.literal_eval to recover the structure, then extract text.
                        if isinstance(content, str) and content.startswith("[{"):
                            try:
                                import ast
                                content = ast.literal_eval(content)
                            except Exception:
                                pass
                        _print_agent(_extract_text(content), raw_output=raw_output)
                        usage = event.get("usage", {})
                        _print_usage(usage)

                    elif etype == "plan":
                        topo_stub = {"nodes": []}  # server holds the real topo
                        est = {
                            "total_cost_usd": event.get("cost_usd", 0),
                            "total_latency_seconds": event.get("latency_seconds", 0),
                        }
                        print(f"\n{YELLOW}{BOLD}plan{RESET}")
                        _print_wrapped(f"Est cost : ${est['total_cost_usd']:.4f}", indent=2)
                        _print_wrapped(f"Est time : {est['total_latency_seconds']:.0f}s", indent=2)
                        print(f"\n{YELLOW}Next: /approve | /cheaper | /cancel{RESET}\n")
                        pending_plan = event

                    elif etype == "executing":
                        total = event.get("total_nodes", "?")
                        print(f"{DIM}  [executing {total} nodes]{RESET}", flush=True)

                    elif etype == "progress":
                        ev = event.get("event", "")
                        node_id = event.get("node_id", "?")
                        done = event.get("completed_nodes", 0)
                        total = event.get("total_nodes", 0)
                        bar_width = 20
                        filled = int((done / total) * bar_width) if total else 0
                        bar = "█" * filled + "░" * (bar_width - filled)
                        if ev == "node_start":
                            print(f"\r{DIM}  [{bar}] {done}/{total}  running: {node_id}{RESET}    ", end="", flush=True)
                        elif ev == "node_complete":
                            filled2 = int((done / total) * bar_width) if total else 0
                            bar2 = "█" * filled2 + "░" * (bar_width - filled2)
                            suffix = "done" if done >= total else f"done: {node_id}"
                            end = "\n" if done >= total else ""
                            print(f"\r{DIM}  [{bar2}] {done}/{total}  {suffix}{RESET}    ", end=end, flush=True)

                    elif etype == "result":
                        output = event.get("output", "")
                        latency = event.get("latency_seconds", 0)
                        models = event.get("models", [])
                        usage = event.get("usage", {})
                        _print_result(output, latency, models, usage)

                    elif etype == "questions":
                        questions = event.get("questions", [])
                        print(f"\n{YELLOW}{BOLD}DAAP needs more info:{RESET}")
                        answers = []
                        for i, q in enumerate(questions, 1):
                            print(f"\n  {BOLD}Q{i}:{RESET} {q.get('question', '')}")
                            options = q.get("options", [])
                            if options:
                                for j, opt in enumerate(options):
                                    label = opt.get("label", "")
                                    desc = opt.get("description", "")
                                    print(f"    {j+1}. {label}" + (f" — {desc}" if desc else ""))
                                print(f"    {DIM}Enter number or type your answer:{RESET} ", end="", flush=True)
                            else:
                                print(f"    {DIM}Your answer:{RESET} ", end="", flush=True)
                            raw_ans = await asyncio.get_running_loop().run_in_executor(None, input, "")
                            raw_ans = raw_ans.strip()
                            if options and raw_ans.isdigit():
                                idx = int(raw_ans) - 1
                                if 0 <= idx < len(options):
                                    raw_ans = options[idx]["label"]
                            answers.append(raw_ans)
                        await ws.send(json.dumps({"type": "answer", "answers": answers}))

                    elif etype == "error":
                        _print_error(f"Server error: {event.get('message', '')}")

            recv_task = asyncio.create_task(_recv_loop())

            try:
                while True:
                    try:
                        with patch_stdout():
                            user_input = await _prompt_session.prompt_async("You: ")
                        user_input = user_input.strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\nBye!")
                        break

                    if not user_input:
                        continue

                    cmd = user_input.lower().strip().lstrip("/")

                    if cmd in ("quit", "exit", "bye"):
                        print("Bye!")
                        break

                    if cmd == "help":
                        _print_system("Commands: /help /approve /cheaper /cancel /sessions /history /memory /topology /profile /mcp /skills /skill /clear /raw /clean /quit")
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
                        # fall through to send as message

                    elif cmd == "cheaper":
                        await ws.send(json.dumps({"type": "make_cheaper"}))
                        continue

                    elif cmd == "cancel":
                        await ws.send(json.dumps({"type": "cancel"}))
                        continue

                    elif cmd == "clear":
                        _save_session(user_id, session_id, [])
                        _print_system("Session history cleared on client.")
                        continue

                    elif cmd == "sessions":
                        saved = _list_sessions(user_id)
                        if not saved:
                            _print_system("No saved sessions.")
                        else:
                            import datetime
                            _print_system(f"Saved sessions ({len(saved)}):")
                            for s in saved[:10]:
                                ts = datetime.datetime.fromtimestamp(s["saved_at"]).strftime("%Y-%m-%d %H:%M")
                                active = " ← current" if s["session_id"] == session_id else ""
                                print(f"  {s['session_id']}  {ts}  ({s['messages']} messages){active}")
                            _print_system(f"Resume: python scripts/chat.py --api-url {api_url} --session <id>")
                        continue

                    elif cmd == "profile":
                        display = user_id.replace("-", " ").title()
                        _print_system(f"User: {display} ({user_id})")
                        _print_system(f"Server: {api_url}")
                        _print_system(f"Session: {session_id}")
                        try:
                            async with httpx.AsyncClient(timeout=5) as hc:
                                r = await hc.get(f"{api_url}/api/v1/memory/{user_id}/profile")
                                if r.status_code == 200:
                                    facts = r.json().get("profile", [])
                                    if facts:
                                        _print_system(f"Memory ({len(facts)} facts):")
                                        for f in facts[:5]:
                                            print(f"  - {f}")
                                    else:
                                        _print_system("Memory: no facts stored yet")
                                else:
                                    _print_system("Memory: unavailable on server")
                        except Exception:
                            _print_system("Memory: could not reach server")
                        continue

                    elif cmd == "history" or cmd == "topology":
                        _print_system("Topology history lives on the server.")
                        _print_system(f"View via: curl {api_url}/topology/{session_id}")
                        continue

                    elif cmd.startswith("topology load"):
                        _print_system("Topology load not supported in remote mode — ask the agent to re-run a previous task.")
                        continue

                    elif cmd.startswith("memory"):
                        parts = cmd.split(None, 2)
                        sub = parts[1] if len(parts) > 1 else ""
                        try:
                            async with httpx.AsyncClient(timeout=5) as hc:
                                if sub == "search" and len(parts) > 2:
                                    r = await hc.get(f"{api_url}/api/v1/memory/{user_id}/history", params={"q": parts[2]})
                                    items = r.json().get("history", []) if r.status_code == 200 else []
                                    if items:
                                        _print_system(f"Memory search '{parts[2]}' ({len(items)} results):")
                                        for i, m in enumerate(items, 1):
                                            print(f"  {i}. {m}")
                                    else:
                                        _print_system("No matching facts found.")
                                elif sub == "clear":
                                    confirm = await asyncio.get_running_loop().run_in_executor(None, input, "Delete ALL memory on server? [y/N]: ")
                                    if confirm.strip().lower() == "y":
                                        r = await hc.delete(f"{api_url}/api/v1/memory/{user_id}")
                                        _print_system("All memory deleted." if r.status_code == 200 else f"Failed: {r.text}")
                                    else:
                                        _print_system("Cancelled.")
                                else:
                                    r = await hc.get(f"{api_url}/api/v1/memory/{user_id}/profile")
                                    items = r.json().get("profile", []) if r.status_code == 200 else []
                                    if items:
                                        _print_system(f"Memory facts ({len(items)}):")
                                        for i, m in enumerate(items, 1):
                                            print(f"  {i}. {m}")
                                        _print_system("Usage: /memory search <query> | /memory clear")
                                    else:
                                        _print_system("No memory facts stored.")
                        except Exception as exc:
                            _print_system(f"Memory error: {exc}")
                        continue

                    elif cmd == "mcp":
                        _print_system("MCP servers run on the server — not visible from remote CLI.")
                        _print_system(f"Check server logs: docker compose logs daap")
                        continue

                    elif cmd == "skills" or cmd.startswith("skill"):
                        _print_system("Skills are loaded on the server — not configurable from remote CLI.")
                        continue

                    await ws.send(json.dumps({"type": "message", "content": user_input}))

            finally:
                recv_task.cancel()

    except Exception as exc:
        _print_error(f"WebSocket error: {exc}")


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace | None = None):
    if args is None:
        args = _parse_args()
    raw_output = bool(args.raw_output)
    _print_header(raw_output)

    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        _print_warn("OPENROUTER_API_KEY is not set. API requests will fail until you set it.")

    # Resolve user identity (first-run prompt or load saved)
    is_new_user = load_local_user() is None
    user_id = await asyncio.get_running_loop().run_in_executor(None, resolve_cli_user)

    # Init session
    session_mgr = SessionManager()
    topology_store = TopologyStore()
    session = session_mgr.create_session(user_id=user_id)
    session.token_tracker = TokenTracker()

    # Restore previous conversation if --session given, else auto-load last
    resume_id = args.session
    restored_id, prior_conversation = _load_session(user_id, resume_id)
    if prior_conversation:
        session.conversation = prior_conversation
        _print_system(f"Session resumed: {restored_id}  ({len(prior_conversation)} messages)")
    elif resume_id:
        _print_warn(f"Session '{resume_id}' not found. Starting fresh.")

    # CLI progress — mirrors WebSocket progress events in terminal
    session._ws_send = _make_cli_progress_send()

    if not is_new_user:
        profile_facts: list[str] = []
        try:
            from daap.memory.config import check_memory_available
            from daap.memory.reader import load_user_profile
            ok, _ = check_memory_available()
            if ok:
                profile_facts = load_user_profile(user_id)
        except Exception:
            pass
        _print_system(_build_welcome_line(user_id, topology_store, profile_facts))
        for fact in profile_facts:
            _print_system(f"  · {fact}")

    # MCP servers — start all configured, non-blocking on failure
    mcp_manager = None
    try:
        from daap.mcpx.manager import get_mcp_manager
        mcp_manager = get_mcp_manager()
        await mcp_manager.start_all()
        connected = mcp_manager.list_connected()
        if connected:
            _print_system(f"MCP servers: {', '.join(connected)}")
    except Exception:
        pass  # MCP is optional

    # Optional memory — disabled gracefully if credentials missing
    daap_memory = None
    user_context = None
    try:
        from daap.memory.palace import DaapMemory
        daap_memory = DaapMemory()
        if daap_memory.available:
            user_context = daap_memory.format_for_master_prompt(session.user_id, "")
        else:
            daap_memory = None
    except Exception:
        pass  # memory is optional — never block startup

    rl_optimizer = None
    try:
        from daap.optimizer.integration import get_tier_recommendations  # noqa: F401
        rl_optimizer = True  # sentinel: LinTS optimizer available
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
    _suppress_agentscope_stdout(session.master_agent)
    try:
        from daap.skills.manager import get_skill_manager

        get_skill_manager().bind_toolkit(toolkit, target="master")
    except Exception:
        pass
    from daap.spec.resolver import MODEL_REGISTRY
    _print_system(f"Session {session.session_id} ready | Master: {MODEL_REGISTRY['powerful']}")
    _print_system(f"Resume later: python scripts/chat.py --session {session.session_id}")
    try:
        from daap.skills.manager import get_skill_manager

        skill_manager = get_skill_manager()
        all_dirs = sorted(
            set(skill_manager.list_skill_dirs("master"))
            | set(skill_manager.list_skill_dirs("subagent"))
        )
        if not all_dirs:
            _print_system("No skills found. Use /skill add <path> or drop skills in ~/.daap/skills/")
        else:
            _print_system(f"Skills loaded: {len(all_dirs)}")
    except Exception:
        pass

    _commands = [
        "/help", "/approve", "/cheaper", "/cancel",
        "/sessions", "/history",
        "/memory", "/memory search", "/memory delete", "/memory clear",
        "/topology", "/topology load",
        "/profile", "/mcp", "/skills",
        "/skill", "/skill add", "/skill remove", "/skill create",
        "/clear", "/raw", "/clean", "/quit",
    ]
    _completer = WordCompleter(_commands, sentence=True)
    _prompt_session: PromptSession = PromptSession(completer=_completer, complete_while_typing=True)

    try:
        while True:
            # Get user input
            try:
                with patch_stdout():
                    user_input = await _prompt_session.prompt_async(f"You: ")
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
                _print_system("Commands: /help /approve /cheaper /cancel /history /memory /topology /mcp /skills /skill /profile /raw /clean /quit")
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

            if cmd == "clear":
                session.conversation = []
                _save_session(user_id, session.session_id, [])
                _print_system("Conversation history cleared.")
                continue

            if cmd == "mcp":
                if mcp_manager:
                    connected = mcp_manager.list_connected()
                    tools = await mcp_manager.list_all_tools()
                    _print_system(f"Connected MCP servers: {', '.join(connected) or 'none'}")
                    _print_system(f"Available tools ({len(tools)} total):")
                    for t in tools:
                        desc = (t.get("description") or "").splitlines()[0][:60]
                        print(f"  {t['name']}: {desc}")
                else:
                    _print_system("No MCP servers connected. Configure servers in ~/.daap/mcpx.json")
                continue

            if cmd.startswith("skill ") or cmd == "skill":
                _handle_skill_command(user_input.strip(), _print_system)
                continue

            if cmd == "skills":
                try:
                    from daap.skills.manager import get_skill_manager
                    skill_manager = get_skill_manager()
                    master_dirs = skill_manager.list_skill_dirs("master")
                    sub_dirs = skill_manager.list_skill_dirs("subagent")
                    all_dirs = sorted(set(master_dirs) | set(sub_dirs))
                    if all_dirs:
                        _print_system(f"Configured skills ({len(all_dirs)} total):")
                        for d in all_dirs:
                            targets = []
                            if d in master_dirs:
                                targets.append("master")
                            if d in sub_dirs:
                                targets.append("subagent")
                            print(f"  {d}  [{', '.join(targets)}]")
                    else:
                        _print_system("No skills configured. Use /skill add <path> or /skill create")
                except Exception as exc:
                    _print_system(f"Skills unavailable: {exc}")
                continue

            if cmd == "profile":
                display = user_id.replace("-", " ").title()
                _print_system(f"User: {display} ({user_id})")
                try:
                    run_count = topology_store.count_runs(user_id)
                    _print_system(f"Runs: {run_count}")
                except Exception:
                    _print_system("Runs: unavailable")
                try:
                    from daap.optimizer.store import BanditStore
                    summary = BanditStore().get_profile_summary(user_id)
                    if summary:
                        _print_system("Optimizer:")
                        for entry in summary:
                            print(f"  {entry['role']:<16} -> {entry['best_arm']}  ({entry['n_pulls']} obs)")
                    else:
                        _print_system("Optimizer: learning (no runs yet)")
                except Exception:
                    _print_system("Optimizer: unavailable")
                try:
                    from daap.memory.reader import load_user_profile
                    facts = load_user_profile(user_id)
                    if facts:
                        _print_system(f"Memory ({len(facts)} facts):")
                        for f in facts[:3]:
                            print(f"  - {f}")
                        if len(facts) > 3:
                            print(f"  [+ {len(facts) - 3} more]")
                    else:
                        _print_system("Memory: no facts stored yet")
                except Exception:
                    _print_system("Memory: unavailable")
                continue

            if cmd == "sessions":
                saved = _list_sessions(user_id)
                if not saved:
                    _print_system("No saved sessions.")
                else:
                    import datetime
                    _print_system(f"Saved sessions ({len(saved)}):")
                    for s in saved[:10]:
                        ts = datetime.datetime.fromtimestamp(s["saved_at"]).strftime("%Y-%m-%d %H:%M")
                        active = " ← current" if s["session_id"] == session.session_id else ""
                        print(f"  {s['session_id']}  {ts}  ({s['messages']} messages){active}")
                    _print_system("Resume: python scripts/chat.py --session <id>")
                continue

            if cmd == "history":
                topos = topology_store.list_topologies(user_id)
                if not topos:
                    _print_system("No topologies saved yet.")
                else:
                    _print_system(f"Saved topologies ({len(topos)}):")
                    for t in topos[:10]:
                        import datetime
                        ts = datetime.datetime.fromtimestamp(t.updated_at).strftime("%Y-%m-%d %H:%M")
                        runs = topology_store.get_runs(t.topology_id, limit=1)
                        run_label = f"{len(runs)} run{'s' if len(runs) != 1 else ''}" if runs else "no runs"
                        print(f"  {t.topology_id[:12]}  {ts}  {t.name or '(unnamed)'}  [{run_label}]")
                    _print_system("Use /topology load <id> to reload one.")
                continue

            if cmd.startswith("memory"):
                parts = cmd.split(None, 2)
                sub = parts[1] if len(parts) > 1 else ""
                try:
                    from daap.memory.config import check_memory_available, get_memory_client
                    from daap.memory.scopes import profile_scope
                    ok, reason = check_memory_available()
                    if not ok:
                        _print_system(f"Memory unavailable: {reason}")
                    elif sub == "search" and len(parts) > 2:
                        from daap.memory.reader import search_user_profile
                        results = search_user_profile(user_id, parts[2])
                        if results:
                            _print_system(f"Memory search '{parts[2]}' ({len(results)} results):")
                            for i, f in enumerate(results, 1):
                                print(f"  {i}. {f}")
                        else:
                            _print_system("No matching facts found.")
                    elif sub == "delete" and len(parts) > 2:
                        idx_str = parts[2].strip()
                        if not idx_str.isdigit():
                            _print_system("Usage: /memory delete <number>  (use /memory to list)")
                        else:
                            idx = int(idx_str) - 1
                            client = get_memory_client()
                            result = client.get_all(**profile_scope(user_id), limit=50)
                            items = result.get("results", [])
                            if idx < 0 or idx >= len(items):
                                _print_system(f"No fact #{idx + 1}. Use /memory to list.")
                            else:
                                mem_id = items[idx]["id"]
                                client.delete(mem_id)
                                _print_system(f"Deleted: {items[idx]['memory']}")
                    elif sub == "clear":
                        confirm = await asyncio.get_running_loop().run_in_executor(None, input, "Delete ALL memory facts? [y/N]: ")
                        if confirm.strip().lower() == "y":
                            client = get_memory_client()
                            result = client.get_all(**profile_scope(user_id), limit=200)
                            for m in result.get("results", []):
                                try:
                                    client.delete(m["id"])
                                except Exception:
                                    pass
                            _print_system("All profile facts deleted.")
                        else:
                            _print_system("Cancelled.")
                    else:
                        # Default: list all
                        client = get_memory_client()
                        result = client.get_all(**profile_scope(user_id), limit=50)
                        items = result.get("results", [])
                        if not items:
                            _print_system("No memory facts stored.")
                            _print_system("Usage: /memory search <query> | /memory delete <n> | /memory clear")
                        else:
                            _print_system(f"Memory facts ({len(items)}):")
                            for i, m in enumerate(items, 1):
                                print(f"  {i}. {m['memory']}")
                            _print_system("Usage: /memory search <query> | /memory delete <n> | /memory clear")
                except Exception as exc:
                    _print_system(f"Memory error: {exc}")
                continue

            if cmd.startswith("topology"):
                parts = cmd.split(None, 2)
                sub = parts[1] if len(parts) > 1 else ""
                if sub == "load" and len(parts) > 2:
                    topo_id_prefix = parts[2].strip()
                    topos = topology_store.list_topologies(user_id)
                    match = next((t for t in topos if t.topology_id.startswith(topo_id_prefix)), None)
                    if not match:
                        _print_system(f"Topology not found: {topo_id_prefix}")
                    else:
                        session.pending_topology = match.spec
                        session.pending_estimate = None
                        _print_system(f"Loaded: {match.name or match.topology_id}")
                        _print_system("Use /approve to execute, or chat to modify.")
                else:
                    topos = topology_store.list_topologies(user_id)
                    if not topos:
                        _print_system("No topologies saved yet.")
                    else:
                        _print_system(f"Saved topologies ({len(topos)}):")
                        for t in topos[:10]:
                            import datetime
                            ts = datetime.datetime.fromtimestamp(t.updated_at).strftime("%Y-%m-%d %H:%M")
                            print(f"  {t.topology_id[:12]}  {ts}  {t.name or '(unnamed)'}")
                        _print_system("Use /topology load <id-prefix> to reload.")
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
        _save_session(user_id, session.session_id, session.conversation)
        _close_memory_client(daap_memory)
        if mcp_manager:
            try:
                await mcp_manager.stop_all()
            except Exception:
                pass


if __name__ == "__main__":
    _args = _parse_args()
    if _args.api_url:
        asyncio.run(_remote_main(_args))
    else:
        asyncio.run(main(_args))
