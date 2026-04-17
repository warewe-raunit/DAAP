"""
DAAP CLI Chat — real-time terminal interface to the master agent.

Usage:
    python scripts/chat.py [--raw-output]

Commands during chat:
    raw          — show raw model output (including JSON)
    clean        — hide raw JSON/code blocks (default)
    help         — show command help
    approve      — approve latest topology plan
    cheaper      — ask agent to make topology cheaper
    cancel       — cancel pending topology
    skill        — manage skills (/skill add|remove|create)
    quit / exit  — end session

Requires: OPENROUTER_API_KEY in .env or environment.
"""

import argparse
import asyncio
import json
import re
import shlex
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

from daap.env import load_project_env

load_project_env()

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
    print(f"{DIM}Commands: /help /approve /cheaper /cancel /mcp /skills /skill /profile /raw /clean /quit{RESET}\n")


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
# Main chat loop
# ---------------------------------------------------------------------------

async def main():
    args = _parse_args()
    raw_output = bool(args.raw_output)
    _print_header(raw_output)

    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        _print_warn("OPENROUTER_API_KEY is not set. API requests will fail until you set it.")

    # Resolve user identity (first-run prompt or load saved)
    is_new_user = load_local_user() is None
    user_id = await asyncio.get_event_loop().run_in_executor(None, resolve_cli_user)

    # Init session
    session_mgr = SessionManager()
    topology_store = TopologyStore()
    session = session_mgr.create_session(user_id=user_id)
    session.token_tracker = TokenTracker()

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
                _print_system("Commands: /help /approve /cheaper /cancel /mcp /skills /skill /profile /raw /clean /quit")
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
        if mcp_manager:
            try:
                await mcp_manager.stop_all()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
