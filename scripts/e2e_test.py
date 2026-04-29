"""
Step 4: End-to-End Integration Test — full loop without a browser.

Simulates: create session → WebSocket conversation → answer questions →
           topology approval → execution → rating.

Run with:
  python scripts/e2e_test.py

Requires: OPENROUTER_API_KEY in environment.
Uses real DuckDuckGo search + web_fetch (no extra API keys needed).
"""

import asyncio
import json
import sys
import os

# Force UTF-8 stdout/stderr — prevents cp1252 crash on Windows when
# AgentScope prints tool results containing unicode arrows etc.
if sys.stdout.encoding != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Load .env before anything else
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
except ImportError:
    pass

from agentscope.message import Msg
from daap.api.sessions import Session, SessionManager, create_session_scoped_toolkit
from daap.master.agent import create_master_agent_with_toolkit
from daap.feedback.store import FeedbackStore
from daap.feedback.collector import collect_run_feedback
from daap.spec.schema import TopologySpec
from daap.spec.resolver import resolve_topology
from daap.executor.engine import execute_topology


USER_PROMPT = (
    "Find B2B leads for my project management SaaS targeting mid-size "
    "construction companies and draft personalized cold emails."
)

# Fallback answers when Gemini asks via text instead of ask_user tool
CONTEXT_FOLLOWUP = (
    "Here are my answers to your questions: "
    "Target roles: Project Managers and VPs of Operations. "
    "Product: project management SaaS with Gantt scheduling, subcontractor coordination, RFI tracking. "
    "Email tone: casual and direct, under 150 words. "
    "Please generate the topology now."
)

# Canned answers for ask_user tool calls (auto-resolved in polling loop)
AUTO_ANSWERS = [
    "Project Managers and VPs of Operations",
    "Gantt scheduling, subcontractor coordination, RFI tracking",
    "Casual & Friendly (Recommended)",
    "yes",
    "yes",
]
_answer_idx = 0


def _next_answer(questions: list) -> list[str]:
    global _answer_idx
    answers = []
    for q in questions:
        if _answer_idx < len(AUTO_ANSWERS):
            answers.append(AUTO_ANSWERS[_answer_idx])
            _answer_idx += 1
        else:
            answers.append(q["options"][0]["label"] if q.get("options") else "yes")
    return answers


async def _run_agent_turn(session: Session, user_text: str) -> str:
    """Send one message to the master agent, auto-resolve any ask_user pauses."""
    msg = Msg(name="user", content=user_text, role="user")

    agent_task = asyncio.create_task(session.master_agent(msg))

    while not agent_task.done():
        if session.pending_questions is not None:
            qs = session.pending_questions
            answers = _next_answer(qs)
            print(f"  [ask_user] {len(qs)} question(s) -> answering: {answers}")
            session._resolve_answers(answers)
        await asyncio.sleep(0.05)

    response_msg = agent_task.result()
    return response_msg.content if isinstance(response_msg.content, str) else str(response_msg.content)


def _banner(title: str):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


async def main():
    _banner("Step 1 — Create session")
    session_mgr = SessionManager()
    session = session_mgr.create_session()
    session.user_id = "e2e-test-user"

    toolkit = create_session_scoped_toolkit(session)
    session.master_agent = create_master_agent_with_toolkit(toolkit)
    print(f"Session: {session.session_id}")

    # ------------------------------------------------------------------
    _banner("Step 2 — Send user prompt")
    print(f"Prompt: {USER_PROMPT}")
    response = await _run_agent_turn(session, USER_PROMPT)
    print(f"\nAgent response (preview):\n{response[:400]}")
    session.conversation.append({"role": "user", "content": USER_PROMPT})
    session.conversation.append({"role": "assistant", "content": response})

    # Keep running turns until we get a topology
    max_turns = 6
    for turn in range(max_turns):
        topo_result = toolkit.get_last_topology_result()
        if topo_result.get("topology") is not None:
            session.pending_topology = topo_result["topology"]
            session.pending_estimate = topo_result["estimate"]
            toolkit.clear_last_topology_result()
            break
        if turn == max_turns - 1:
            print("\n[FAIL] Master agent did not generate a topology after max turns.")
            sys.exit(1)
        # Continue conversation — send context answers in case Gemini asked via text
        print(f"\n  [Turn {turn+2}] Continuing conversation...")
        followup = CONTEXT_FOLLOWUP if turn == 0 else "Please generate the topology now."
        response = await _run_agent_turn(session, followup)
        print(f"  Agent: {response[:200]}")

    assert session.pending_topology is not None, "No topology generated"
    nodes = [n.get("node_id") for n in session.pending_topology.get("nodes", [])]
    est = session.pending_estimate or {}
    print(f"\nTopology: {nodes}")
    print(f"Estimated cost: ${est.get('total_cost_usd', 0):.3f}")
    print(f"Estimated latency: {est.get('total_latency_seconds', 0):.0f}s")
    print(f"Min viable cost: ${est.get('min_viable_cost_usd', 0):.3f}")

    # ------------------------------------------------------------------
    _banner("Step 3 — Execute topology")
    topo_id = session.pending_topology.get("topology_id", "unknown")
    user_prompt_for_exec = session.pending_topology.get("user_prompt", USER_PROMPT)
    print(f"Executing {topo_id} ...")

    topology_spec = TopologySpec.model_validate(session.pending_topology)
    resolved = resolve_topology(topology_spec)
    if isinstance(resolved, list):
        print(f"[FAIL] Resolution errors: {[e.message for e in resolved]}")
        sys.exit(1)

    session.is_executing = True
    result = await execute_topology(resolved=resolved, user_prompt=user_prompt_for_exec)
    session.is_executing = False

    print(f"Success: {result.success}")
    print(f"Latency: {result.total_latency_seconds:.1f}s")
    if result.error:
        print(f"Error: {result.error}")
    if result.final_output:
        print(f"\nOutput preview:\n{result.final_output[:600]}")

    session.execution_result = {
        "topology_id": result.topology_id,
        "final_output": result.final_output,
        "success": result.success,
        "error": result.error,
        "latency_seconds": result.total_latency_seconds,
    }

    # ------------------------------------------------------------------
    _banner("Step 4 — Store feedback")
    fb_store = FeedbackStore("e2e_test_feedback.db")
    collect_run_feedback(fb_store, session.session_id, session.pending_topology, result)
    fb_store.store_rating(session.session_id, rating=4, comment="E2E test run")
    runs = fb_store.get_runs_for_session(session.session_id)
    print(f"Feedback stored: {len(runs)} run(s), rating={runs[0]['rating']}")

    # ------------------------------------------------------------------
    _banner("RESULT")
    if result.success:
        print("E2E test PASSED ✓")
        print(f"  Nodes executed: {len(result.node_results)}")
        for nr in result.node_results:
            print(f"    {nr.node_id}: {nr.latency_seconds:.1f}s — {nr.output_text[:80]}")
    else:
        print(f"E2E test FAILED ✗ — {result.error}")
        sys.exit(1)

    # Clean up test DB
    try:
        os.remove("e2e_test_feedback.db")
    except OSError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
