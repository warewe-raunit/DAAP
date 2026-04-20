"""
Step 2: Master Agent Smoke Test — real OpenRouter API, no execution.

Tests routing behavior against 3 prompt categories:
  direct   → agent responds without calling any tool
  ask_user → agent calls ask_user tool with structured questions
  topology → agent calls generate_topology tool

Run with:
  python scripts/smoke_test_master.py

Requires: OPENROUTER_API_KEY in environment.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
except ImportError:
    pass

from agentscope.message import Msg
from daap.master.agent import create_master_agent
from daap.master.tools import (
    clear_last_topology_result,
    get_last_topology_result,
)

PROMPTS = [
    {
        "id": "simple_email",
        "prompt": "Write me a cold email for a CRM product targeting law firms.",
        "expected": "direct",
    },
    {
        "id": "vague_leads",
        "prompt": "I need leads.",
        "expected": "ask_user",
    },
    {
        "id": "complex_pipeline",
        "prompt": (
            "Find B2B leads for my project management SaaS targeting mid-size "
            "construction companies and draft personalized cold emails."
        ),
        "expected": "topology",
    },
]

AUTO_RESOLVE_ANSWERS = [
    "SaaS product",
    "Mid-market (50-500 employees)",
    "Construction",
]


async def _auto_resolve_pending_questions(
    agent_task: asyncio.Task,
    toolkit,
) -> None:
    """Resolve ask_user prompts as soon as they appear while the agent is running."""
    while not agent_task.done():
        if toolkit.get_pending_questions() is not None:
            print("  → ask_user called — auto-resolving questions")
            toolkit.resolve_pending_questions(AUTO_RESOLVE_ANSWERS)
            return
        await asyncio.sleep(0.1)


async def run_one(prompt_spec: dict) -> dict:
    """Run a single prompt through the master agent and classify the result."""
    print(f"\n{'='*60}")
    print(f"[{prompt_spec['id']}] Expected: {prompt_spec['expected']}")
    print(f"Prompt: {prompt_spec['prompt']}")
    print("-" * 60)

    clear_last_topology_result()
    master = create_master_agent()
    toolkit = master._daap_toolkit

    msg = Msg(name="user", content=prompt_spec["prompt"], role="user")
    agent_task = asyncio.create_task(master(msg))
    resolver_task = asyncio.create_task(_auto_resolve_pending_questions(agent_task, toolkit))

    response_msg = await agent_task
    await resolver_task

    response_text = (
        response_msg.content
        if isinstance(response_msg.content, str)
        else str(response_msg.content)
    )

    # Classify what happened
    topo = get_last_topology_result()
    pending_qs = toolkit.get_pending_questions()

    if topo.get("topology") is not None:
        actual = "topology"
    elif pending_qs is not None:
        actual = "ask_user"
    else:
        actual = "direct"

    passed = actual == prompt_spec["expected"]
    status = "PASS ✓" if passed else f"FAIL ✗  (got '{actual}', expected '{prompt_spec['expected']}')"

    print(f"Response preview: {response_text[:200]}")
    print(f"Result: {status}")

    if topo.get("topology"):
        nodes = [n.get("node_id") for n in topo["topology"].get("nodes", [])]
        est = topo.get("estimate", {})
        print(f"  Topology nodes: {nodes}")
        print(f"  Estimated cost: ${est.get('total_cost_usd', 0):.3f}")
        print(f"  Estimated latency: {est.get('total_latency_seconds', 0):.0f}s")

    clear_last_topology_result()
    return {"id": prompt_spec["id"], "passed": passed, "actual": actual, "expected": prompt_spec["expected"]}


async def main():
    results = []
    for spec in PROMPTS:
        result = await run_one(spec)
        results.append(result)

    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r["passed"])
    for r in results:
        icon = "✓" if r["passed"] else "✗"
        print(f"  {icon} [{r['id']}] expected={r['expected']} actual={r['actual']}")
    print(f"\n{passed}/{len(results)} passed")

    if passed < len(results):
        print("\nFAILED PROMPTS → tune master/prompts.py system prompt")
        sys.exit(1)
    else:
        print("\nAll routing decisions correct ✓")


if __name__ == "__main__":
    asyncio.run(main())
