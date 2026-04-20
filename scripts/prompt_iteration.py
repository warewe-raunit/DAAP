"""
Step 5: Prompt Iteration — tests master agent routing across 20 prompts.

Runs each prompt, classifies actual behavior (direct / ask_user / topology),
compares to expected, and prints a failure analysis so you know exactly what
to fix in master/prompts.py.

Run with:
  python scripts/prompt_iteration.py

Requires: OPENROUTER_API_KEY in environment.
"""

import asyncio
import sys
import os
import time

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

# ---------------------------------------------------------------------------
# Test suite — 20 prompts across verticals + complexity levels
# ---------------------------------------------------------------------------

PROMPTS = [
    # --- Direct (simple, self-contained) ---
    {
        "id": "cold_email_crm_law",
        "prompt": "Write me a cold email for a CRM product targeting law firms.",
        "expected": "direct",
        "rationale": "specific product + audience + clear deliverable",
    },
    {
        "id": "subject_line_advice",
        "prompt": "What's a good subject line for a cold email to CTOs?",
        "expected": "direct",
        "rationale": "pure advice, single response",
    },
    {
        "id": "value_props_invoicing",
        "prompt": "Give me 5 value propositions for a B2B invoicing SaaS targeting freelancers.",
        "expected": "direct",
        "rationale": "brainstorm, self-contained",
    },
    {
        "id": "follow_up_email",
        "prompt": "Write a follow-up email for a SaaS demo that went well with an HR director.",
        "expected": "direct",
        "rationale": "specific enough to handle directly",
    },
    {
        "id": "icp_advice",
        "prompt": "What industries are best for a document automation SaaS to target first?",
        "expected": "direct",
        "rationale": "strategy question, direct answer",
    },

    # --- Ask User (vague, missing critical info) ---
    {
        "id": "vague_leads",
        "prompt": "I need leads.",
        "expected": "ask_user",
        "rationale": "no product, no ICP, no industry",
    },
    {
        "id": "vague_outreach",
        "prompt": "Help me with outreach.",
        "expected": "ask_user",
        "rationale": "no product, no audience, no channel",
    },
    {
        "id": "vague_startup",
        "prompt": "I'm a startup and I need customers.",
        "expected": "ask_user",
        "rationale": "no product info, no ICP",
    },
    {
        "id": "vague_emails",
        "prompt": "Write me some emails.",
        "expected": "ask_user",
        "rationale": "emails for what? to whom?",
    },
    {
        "id": "vague_pipeline",
        "prompt": "I want to automate my sales process.",
        "expected": "ask_user",
        "rationale": "no product, no ICP, no current state",
    },

    # --- Topology (complex, multi-agent) ---
    {
        "id": "full_outreach_construction",
        "prompt": (
            "Find B2B leads for my project management SaaS targeting mid-size "
            "construction companies and draft personalized cold emails."
        ),
        "expected": "topology",
        "rationale": "research + evaluate + personalize + write",
    },
    {
        "id": "market_research",
        "prompt": (
            "Research the competitive landscape for project management tools in "
            "construction, identify the top 10 competitors, and summarize their "
            "key differentiators."
        ),
        "expected": "topology",
        "rationale": "research + synthesis — multi-agent",
    },
    {
        "id": "lead_enrichment",
        "prompt": (
            "Take this list of 50 companies and enrich each one with recent news, "
            "key decision makers, and LinkedIn presence. Then score them by ICP fit."
        ),
        "expected": "topology",
        "rationale": "enrichment + scoring — parallel research then evaluation",
    },
    {
        "id": "full_outreach_healthcare",
        "prompt": (
            "Find decision makers at mid-size healthcare clinics who might need "
            "scheduling software and write personalized outreach emails."
        ),
        "expected": "topology",
        "rationale": "research + write — different vertical",
    },
    {
        "id": "sequence_generation",
        "prompt": (
            "Research 20 fintech companies, qualify them against my ICP "
            "(Series A-B, 50-200 employees, payments focus), and create a "
            "3-step email sequence for each qualified lead."
        ),
        "expected": "topology",
        "rationale": "research + qualify + multi-step write",
    },

    # --- Edge cases (boundary prompts) ---
    {
        "id": "semi_specific_email",
        "prompt": "Write a cold email for my SaaS to construction companies.",
        "expected": "direct",
        "rationale": "enough info to write one email directly",
    },
    {
        "id": "partial_context_leads",
        "prompt": "Find leads for my HR software.",
        "expected": "ask_user",
        "rationale": "product known but ICP, volume, and output format missing",
    },
    {
        "id": "explicit_pipeline_request",
        "prompt": (
            "Build me a full lead generation and outreach pipeline for "
            "my legal tech SaaS targeting boutique law firms."
        ),
        "expected": "topology",
        "rationale": "user explicitly asks for pipeline — multi-agent required",
    },
    {
        "id": "single_company_research",
        "prompt": (
            "Research Procore Technologies — their recent news, key decision makers, "
            "and write a personalized cold email pitching my PM SaaS as a complement."
        ),
        "expected": "topology",
        "rationale": "research one company then write — still 2-phase",
    },
    {
        "id": "pricing_strategy",
        "prompt": "What pricing model works best for a B2B SaaS targeting SMBs?",
        "expected": "direct",
        "rationale": "pure strategy advice, no data gathering needed",
    },
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

AUTO_RESOLVE_ANSWERS = ["SaaS product", "Mid-market (50-500)", "Construction", "Proceed"]


async def run_prompt(spec: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        clear_last_topology_result()
        master = create_master_agent()
        toolkit = master._daap_toolkit

        # Auto-resolve ask_user if triggered (so agent doesn't block forever)
        async def auto_resolve(agent_task: asyncio.Task):
            while not agent_task.done():
                if toolkit.get_pending_questions() is not None:
                    toolkit.resolve_pending_questions(AUTO_RESOLVE_ANSWERS)
                    return
                await asyncio.sleep(0.1)

        t0 = time.time()
        msg = Msg(name="user", content=spec["prompt"], role="user")
        master_task = asyncio.create_task(master(msg))
        resolver_task = asyncio.create_task(auto_resolve(master_task))
        try:
            response_msg = await asyncio.wait_for(master_task, timeout=60)
            elapsed = time.time() - t0
        except asyncio.TimeoutError:
            resolver_task.cancel()
            await asyncio.gather(resolver_task, return_exceptions=True)
            return {**spec, "actual": "timeout", "passed": False,
                    "response": "", "elapsed": 60.0}
        else:
            await resolver_task

        response_text = (
            response_msg.content
            if isinstance(response_msg.content, str)
            else str(response_msg.content)
        )

        topo = get_last_topology_result()
        pending_qs = toolkit.get_pending_questions()

        if topo.get("topology") is not None:
            actual = "topology"
        elif pending_qs is not None:
            actual = "ask_user"
        else:
            actual = "direct"

        clear_last_topology_result()

        return {
            **spec,
            "actual": actual,
            "passed": actual == spec["expected"],
            "response": response_text[:150],
            "elapsed": round(time.time() - t0, 1),
        }


async def main():
    print("DAAP Prompt Iteration Test")
    print(f"Running {len(PROMPTS)} prompts (concurrency=2)...\n")

    # Run with limited concurrency to avoid rate limits
    semaphore = asyncio.Semaphore(2)
    tasks = [run_prompt(spec, semaphore) for spec in PROMPTS]
    results = await asyncio.gather(*tasks)

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]

    print(f"\n{'='*70}")
    print(f"RESULTS  {len(passed)}/{len(results)} passed")
    print("=" * 70)

    # Group by category
    categories = {"direct": [], "ask_user": [], "topology": []}
    for r in results:
        categories.get(r["expected"], []).append(r)

    for cat, items in categories.items():
        cat_passed = sum(1 for r in items if r["passed"])
        print(f"\n  {cat.upper()} ({cat_passed}/{len(items)})")
        for r in items:
            icon = "✓" if r["passed"] else "✗"
            mismatch = f" → got '{r['actual']}'" if not r["passed"] else ""
            print(f"    {icon} [{r['id']}]{mismatch}  ({r['elapsed']}s)")
            if not r["passed"]:
                print(f"       prompt:   {r['prompt'][:80]}")
                print(f"       response: {r['response'][:100]}")
                print(f"       why:      {r['rationale']}")

    # ---------------------------------------------------------------------------
    # Failure analysis → prompt fix suggestions
    # ---------------------------------------------------------------------------
    if failed:
        print(f"\n{'='*70}")
        print("PROMPT FIX SUGGESTIONS")
        print("=" * 70)

        wrong_topology = [r for r in failed if r["expected"] != "topology" and r["actual"] == "topology"]
        wrong_direct   = [r for r in failed if r["expected"] != "direct" and r["actual"] == "direct"]
        wrong_ask      = [r for r in failed if r["expected"] != "ask_user" and r["actual"] == "ask_user"]

        if wrong_topology:
            print("\n[Over-engineering] Called generate_topology when should respond directly:")
            for r in wrong_topology:
                print(f"  • [{r['id']}] {r['prompt'][:70]}")
            print("  FIX: Strengthen 'DON'T use generate_topology' list in prompts.py.")
            print("       Add: 'single email', 'one summary', 'advice question'.")

        if wrong_direct:
            print("\n[Under-acting] Responded directly when should ask or build topology:")
            for r in wrong_direct:
                print(f"  • [{r['id']}] expected={r['expected']} — {r['prompt'][:70]}")
            print("  FIX: Check 'ACT IMMEDIATELY' criteria — may be too broad.")
            print("       Add examples of vague prompts that REQUIRE ask_user.")

        if wrong_ask:
            print("\n[Over-cautious] Asked questions when should act immediately:")
            for r in wrong_ask:
                print(f"  • [{r['id']}] {r['prompt'][:70]}")
            print("  FIX: Add to 'ACT IMMEDIATELY' section: prompts with specific")
            print("       product + audience + deliverable need no clarification.")

        timeouts = [r for r in failed if r["actual"] == "timeout"]
        if timeouts:
            print("\n[Timeout] These prompts exceeded 60s:")
            for r in timeouts:
                print(f"  • [{r['id']}]")
            print("  FIX: Check AgentScope max_iters — may be looping.")

        print(f"\nEdit: daap/master/prompts.py → get_master_system_prompt()")
        sys.exit(1)
    else:
        print("\nAll routing decisions correct ✓")
        print("Prompt quality verified across all verticals and complexity levels.")


if __name__ == "__main__":
    asyncio.run(main())
