"""
Topology Architect — single-shot structured topology generation.

Replaces the ReAct self-correction loop with one LLM call constrained by
`response_format: json_schema`. Pydantic validates the result; domain
validation runs once. No iterative validate→fix→retry round-trips.

Why single-shot: ReAct loops on JSON synthesis amplify cost without
improving output quality. Constrained decoding pushes parse failures
to ~0 on the first attempt, so the loop is dead weight.
"""

import json
import logging
import os
from datetime import datetime, timezone

from agentscope.formatter import OpenAIChatFormatter

from daap.executor.tracked_model import TrackedOpenAIChatModel
from daap.spec.schema import get_topology_json_schema
from daap.tools.registry import get_available_tool_names, get_tool_descriptions

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_ARCHITECT_MODEL = "qwen/qwen3.6-plus"

# ---------------------------------------------------------------------------
# Topology Architect system prompt — all domain knowledge lives here
# ---------------------------------------------------------------------------

_TOPOLOGY_ARCHITECT_SYSTEM_PROMPT = """\
The Topology Architect designs multi-agent execution topologies as strict JSON.

## Output

Output is constrained to a TopologySpec JSON object via response_format:json_schema — structural validity is guaranteed by the runtime. Focus on substance:

- Selecting the right tools per node
- Picking the correct model_tier
- Writing tight, complete node system_prompts (4 sections, ≤250 words)
- Connecting nodes via well-typed edges

Output ONLY the JSON object. No commentary, no markdown fences.

## Runtime Context (ground every choice in this)

{runtime_block}

## User Context

{user_context_block}

## Tools Available to Nodes (semantic catalog — name, behaviour, I/O)

Pick tools strictly from this catalog. Never invent tool names. If a needed capability is missing here, design around it instead of hallucinating a tool.

{tool_catalog}

### Compact Allow-List

The following abstract tool names are permitted in node `tools[].name`. Anything else fails validation:

{tool_names}

## Model Tiers — pick by reasoning load, not task type

- `fast` — mechanical I/O: the node calls a tool and passes the result on. No judgment required.\
 Default for every tool node (search, fetch, parse, extract, route, deduplicate by exact match).
- `smart` — non-trivial judgment: the node must make a decision a human would find non-obvious\
 (score candidates, compare conflicting sources, draft persuasive prose, synthesize into an original argument).\
 Used ONLY when `fast` would produce visibly wrong output.
- `powerful` — rare. Only when `smart` has empirically failed on this specific synthesis task.

Default: `fast`. Upgraded to `smart` only when the node makes a judgment call requiring reasoning.\
 `powerful` is not "safer" — it is slower and more expensive.

## Structural Constraints

- 2–4 nodes typical; 6+ is over-engineered. One node = one responsibility.
- DAG only. No cycles. First node has no edge inputs. Last node = final deliverable.
- Every node MUST be connected: every non-first node must have at least one incoming edge, every non-last node must have at least one outgoing edge. Orphan nodes (no edges) are invalid.
- `parallel_instances` 2–4 max; consolidation node required when >1.
- `react` mode requires ≥1 tool. `single` for deterministic write/format steps.
- `max_react_iterations`: search/fetch nodes = **4** (2-3 tool calls + final answer turn). Consolidation/eval = **2**. Never exceed 6 — higher values cause runaway loops that repeat output and waste tokens.

## Reddit Research Pattern

For Reddit post discovery and content reading, use only the tools that exist at runtime:

1. **RedditSearch** — PRIMARY tool for Reddit URL discovery. Search by product category, pain point,
   target subreddit, and buyer language. Use `time_filter` for recency.
2. **RedditFetch** — PRIMARY tool for reading a specific Reddit post URL. It returns the post title,
   body, metadata, and top comments. Use it after RedditSearch or WebSearch finds candidate post URLs.
3. **BatchRedditFetch** — PRIMARY tool when the upstream search node returns multiple candidate URLs.
   It reads every candidate Reddit URL in one tool call and labels each URL ACTIVE, REJECTED, or
   SKIPPED_DUE_TO_SIZE. Promotion-fit topologies MUST prefer BatchRedditFetch over many serial RedditFetch calls.
4. **WebSearch** — fallback only. Search with `site:reddit.com/r/<subreddit>/comments` or
   `site:reddit.com "<pain keyword>" "<product category>"`.

Do NOT use WebFetch for reddit.com URLs. Do NOT invent PullpushSearch, Pushshift, or
pullpush.io tools. If RedditSearch or WebSearch finds candidate Reddit URLs, every promotion-fit topology MUST include
a BatchRedditFetch node (or RedditFetch only for a single explicit URL) before any scoring/evaluation node.

### Query and call budgets (non-negotiable)

The reddit_search node system_prompt MUST embed these hard limits verbatim:

- **Maximum 3 RedditSearch calls per node, total.** After 3 calls, call generate_response immediately with the URLs collected so far.
- If a RedditSearch result contains the marker `[STOP_SEARCHING_SIGNAL]`, stop searching on the NEXT iteration and call generate_response.
- If a returned URL is tagged `[ALREADY_SEEN_IN_PRIOR_SEARCH]`, do NOT re-issue the same or near-identical query.
- Do NOT call RedditSearch with more than one variation per pain-point dimension (one for pain phrasing, one for product category, one for subreddit-scoped). Three queries cover the space.

The fetch_posts node system_prompt MUST embed:

- Read the BatchRedditFetch output as a structured manifest with three lists: ACTIVE, REJECTED, SKIPPED_DUE_TO_SIZE.
- Forward ONLY the ACTIVE post bodies to downstream nodes. Mark REJECTED and SKIPPED_DUE_TO_SIZE URLs with status='POST_NOT_READ' so downstream evaluation does not treat them as content.
- NEVER fabricate post bodies for URLs in REJECTED or SKIPPED_DUE_TO_SIZE.

The evaluate_posts node system_prompt MUST embed:

- If a post body matches a known bot/automod template (contains "I am a bot" or "I am a moderator of this subreddit" or "automatically performed"), mark it status='BOT_TEMPLATE' and exclude it from the promote list. Do NOT score bot bodies as if they were real posts.
- If an upstream item has status='POST_NOT_READ', do NOT invent a body. Output the URL with score=0 and a note "POST_NOT_READ — body not available, cannot evaluate."

Recommended topology for "find Reddit posts where I can promote X":
`reddit_search` (RedditSearch, react, **smart**, max_react_iterations=4) → `fetch_posts` (BatchRedditFetch, react, fast, max_react_iterations=2) →
`evaluate_posts` (single, smart) → `final_report` (single, smart).
IMPORTANT: reddit_search MUST be `smart` tier when the search plan involves structured, multi-category queries with
different `sort`/`time_filter` combinations. The `fast` model cannot reliably follow multi-step parameterized tool call
instructions — it fires generic queries without passing `subreddit`, `sort`, or `time_filter` parameters.
The reddit_search system_prompt MUST explicitly list each RedditSearch call with its exact parameters so the model
executes them verbatim (e.g. "Call 1: RedditSearch(query='email verification', sort='new', time_filter='week', subreddit='SaaS')").
Do not use parallel_instances to mean "one URL per instance"; DAAP parallel instances receive the same input.

5. **KeywordsEverywhere** — keyword search volume, CPC, and competition data via Keywords Everywhere API.
   USE FOR: ranking which product keywords have real monthly search demand before driving RedditSearch queries.
   Requires KEYWORDS_EVERYWHERE_API_KEY env var. Max 100 keywords per call. Each keyword costs 1 credit.
   Returns: keyword | monthly_vol | cpc | competition, sorted by volume. Use commercial_score = vol * (1 + cpc).

6. **KeywordsEverywhereTraffic** — estimates monthly organic Google traffic for a list of URLs.
   USE FOR: identifying which Reddit post URLs already rank on Google (google_monthly_traffic > 0 = Google-indexed).
   Higher traffic = post reaches audiences beyond Reddit = higher-value outreach target.
   Requires KEYWORDS_EVERYWHERE_API_KEY. Max 20 URLs per call. Each URL costs credits.
   Returns: url | total_keywords_ranked | monthly_organic_traffic | traffic_value.

Recommended topology for high-quality Reddit post discovery with SEO scoring:
`keyword_ranker` (KeywordsEverywhere, react, fast) → `reddit_search` (RedditSearch, react, **smart**) →
`fetch_posts` (BatchRedditFetch, react, fast) → `google_rank_check` (KeywordsEverywhereTraffic, react, fast) →
`evaluate_posts` (single, smart) → `final_report` (single, smart).

## Temporal Rule

Time-scoped tasks → every WebSearch call MUST include `date_from`.\
 RedditSearch only supports `time_filter` buckets (`day`, `week`, `month`, `year`, `all`), so for "last 2 months" use `time_filter="year"` in discovery and require the fetch/evaluation node to reject posts with `created_utc` older than today minus 2 months.\
 The date filter is never omitted when recency is required.

## Node system_prompt — 4 Required Sections (in order)

Every node system_prompt contains exactly these 4 sections. Missing any degrades output quality.

1. **ROLE** — one specific role, no compound roles. "You are a [specific role]."
2. **SCOPE** — what it receives + explicit NOT-list. "You receive: [input]. You must NOT: [out-of-scope items]."
3. **OUTPUT CONTRACT** — format + hard length cap + field names + "Begin immediately with X. No preamble." + **"Output your result EXACTLY ONCE. Do not repeat it. Your turn ends after you output the result."**
4. **EFFORT** (react nodes) — max iterations · stop condition · one alternative query on empty result.
   **STOP RULE**: Once output contract is satisfied, output your result and STOP. Do not call more tools. Do not restate the output.
   **CRITICAL FALLBACK**: If a tool fails repeatedly or you cannot fulfill the scope, DO NOT hallucinate. Output EXACTLY `EXECUTION_FAILED: <reason>` to abort.
   **TEMPORAL** (time-scoped nodes) — compute date_from from today minus window; discard out-of-range results.

Max 250 words per node system_prompt.

## Required JSON Structure

```json
{{
  "topology_id": "topo-a1b2c3d4",
  "version": 1,
  "created_at": "2026-04-23T12:00:00+00:00",
  "user_prompt": "original user request",
  "nodes": [
    {{
      "node_id": "search",
      "role": "Web Researcher",
      "model_tier": "fast",
      "agent_mode": "react",
      "system_prompt": "ROLE / SCOPE / OUTPUT CONTRACT / EFFORT sections",
      "tools": [{{"name": "WebSearch"}}],
      "inputs": [],
      "outputs": [{{"data_key": "results", "data_type": "string", "description": "..."}}],
      "instance_config": {{"parallel_instances": 1, "consolidation": null}},
      "max_react_iterations": 4
    }},
    {{
      "node_id": "write",
      "role": "Report Writer",
      "model_tier": "smart",
      "agent_mode": "single",
      "system_prompt": "ROLE / SCOPE / OUTPUT CONTRACT sections",
      "tools": [],
      "inputs": [{{"data_key": "results", "data_type": "string", "description": "..."}}],
      "outputs": [{{"data_key": "report", "data_type": "string", "description": "..."}}],
      "instance_config": {{"parallel_instances": 1, "consolidation": null}},
      "max_react_iterations": 1
    }}
  ],
  "edges": [
    {{"source_node_id": "search", "target_node_id": "write", "data_key": "results", "description": ""}}
  ]
}}
```

Critical field rules:
- `outputs` non-empty on every node
- `instance_config` always present — {{"parallel_instances": 1, "consolidation": null}} for single instances
- `react` mode requires tools; `single` requires no tools
- `node_id` lowercase, matches `^[a-z][a-z0-9_]*$`
- edge `data_key` must match the `data_key` in the source node's `outputs`
- `topology_id`: "topo-" + exactly 8 lowercase hex chars
- `created_at`: ISO 8601 timestamp with timezone offset
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tool_catalog(available_tools: set[str]) -> str:
    """Return semantic catalog (name + docstring) for builtin tools the architect may use.

    Only emits builtins (mcp:// entries excluded — MCP tools surface in the
    runtime block as `connected_mcp_servers`). Filters by `available_tools`
    so disabled tools never appear in the prompt.
    """
    raw = get_tool_descriptions()
    if not available_tools:
        return raw
    blocks: list[str] = []
    for block in raw.split("\n\n"):
        first = block.split("\n", 1)[0].strip()
        # First line shape: "**ToolName**"
        name = first.strip("* ").strip()
        if name in available_tools:
            blocks.append(block)
    return "\n\n".join(blocks) if blocks else raw


def _build_runtime_block(runtime_context: dict | None) -> str:
    """Compact runtime grounding block. Always includes today's date.

    Pulled from the master runtime snapshot so the architect sees the same
    capability picture the master agent does (MCP servers, known gaps,
    feature flags). Empty / None inputs degrade gracefully.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [f"- today: {today} (UTC)"]

    rc = runtime_context if isinstance(runtime_context, dict) else {}

    mcp_servers = rc.get("connected_mcp_servers") or []
    if isinstance(mcp_servers, list) and mcp_servers:
        lines.append(f"- connected_mcp_servers: {', '.join(map(str, mcp_servers))}")
    else:
        lines.append("- connected_mcp_servers: none")

    gaps = rc.get("known_gaps") or []
    if isinstance(gaps, list) and gaps:
        formatted = []
        for g in gaps:
            if isinstance(g, dict):
                cap = g.get("capability") or g.get("name") or "?"
                reason = g.get("reason") or g.get("note") or ""
                formatted.append(f"{cap} ({reason})" if reason else str(cap))
            else:
                formatted.append(str(g))
        lines.append("- known_gaps: " + "; ".join(formatted))

    caps = rc.get("functional_capabilities") or []
    if isinstance(caps, list) and caps:
        names = [c.get("capability", c) if isinstance(c, dict) else c for c in caps]
        lines.append("- functional_capabilities: " + ", ".join(map(str, names)))

    flags = rc.get("feature_flags") or {}
    if isinstance(flags, dict) and flags:
        flag_str = ", ".join(f"{k}={v}" for k, v in flags.items())
        lines.append(f"- feature_flags: {flag_str}")

    mode = rc.get("execution_mode")
    if mode:
        lines.append(f"- execution_mode: {mode}")

    return "\n".join(lines)


def _build_user_context_block(user_context: object) -> str:
    """One-paragraph user context, capped. Returns 'none' if absent.

    Accepts dict (memory snapshot) or pre-formatted string (master prompt
    formatter output). Never raises.
    """
    if user_context is None:
        return "none"
    if isinstance(user_context, str):
        text = user_context.strip()
        return text[:1500] if text else "none"
    if isinstance(user_context, dict):
        try:
            return json.dumps(user_context, indent=2, default=str)[:1500]
        except Exception:
            return str(user_context)[:1500]
    return str(user_context)[:1500]


def _content_to_text(content: object) -> str:
    """Extract plain text from AgentScope/OpenAI message content."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "content" in block:
                    parts.append(str(block.get("content", "")))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


# ---------------------------------------------------------------------------
# Single-shot generation
# ---------------------------------------------------------------------------

async def generate_topology(
    user_msg: str,
    available_tools: set[str] | None = None,
    operator_config: dict | None = None,
    *,
    user_context: object = None,
    runtime_context: dict | None = None,
) -> tuple[dict | None, str]:
    """
    Generate a topology in a single LLM call constrained by JSON schema.

    Args:
        user_msg: Composed architect input — current topology + user feedback.
        available_tools: Tool names the architect may reference (defaults to all).
        operator_config: Optional provider override for the architect call.
        user_context: User memory snapshot (dict) or pre-formatted string. The
            architect uses it to ground node prompts in the user's domain
            (e.g. their product, audience, prior preferences).
        runtime_context: Master runtime snapshot — MCP servers, known_gaps,
            feature_flags, execution_mode. The architect uses it to avoid
            designing topologies that depend on unavailable infrastructure.

    Returns:
        (parsed_dict, raw_text). parsed_dict is None if JSON could not be parsed.
        Pydantic validation runs at the call site (delegate_to_architect).
    """
    if available_tools is None:
        available_tools = get_available_tool_names()

    builtin_tools = {t for t in available_tools if not t.startswith("mcp://")}
    tool_names = ", ".join(sorted(builtin_tools))
    tool_catalog = _build_tool_catalog(builtin_tools)
    runtime_block = _build_runtime_block(runtime_context)
    user_context_block = _build_user_context_block(user_context)

    sys_prompt = _TOPOLOGY_ARCHITECT_SYSTEM_PROMPT.format(
        tool_catalog=tool_catalog,
        tool_names=tool_names,
        runtime_block=runtime_block,
        user_context_block=user_context_block,
    )

    if operator_config:
        base_url = operator_config.get("base_url") or OPENROUTER_BASE_URL
        api_key_env = operator_config.get("api_key_env", "OPENROUTER_API_KEY")
        model_id = operator_config.get("model_map", {}).get(
            "powerful", OPENROUTER_ARCHITECT_MODEL
        )
    else:
        base_url = OPENROUTER_BASE_URL
        api_key_env = "OPENROUTER_API_KEY"
        model_id = OPENROUTER_ARCHITECT_MODEL

    schema = get_topology_json_schema()

    model = TrackedOpenAIChatModel(
        model_name=model_id,
        api_key=os.environ.get(api_key_env, ""),
        client_kwargs={"base_url": base_url},
        stream=False,
        generate_kwargs={
            "temperature": 0.2,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "TopologySpec",
                    "strict": False,
                    "schema": schema,
                },
            },
        },
    )
    formatter = OpenAIChatFormatter()

    from agentscope.message import Msg
    msgs = [
        Msg(name="system", role="system", content=sys_prompt),
        Msg(name="user", role="user", content=user_msg),
    ]
    formatted = await formatter.format(msgs)

    try:
        response = await model(formatted)
    except Exception:
        logger.exception("Topology Architect generation call failed")
        return None, ""

    raw = _content_to_text(getattr(response, "content", None) or str(response))

    # Constrained decoding produces JSON content directly. Fast path:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed, raw
    except json.JSONDecodeError:
        pass

    # Lightweight rescue if a provider wraps output despite json_schema mode.
    try:
        from json_repair import repair_json
        repaired = repair_json(raw, return_objects=True)
        if isinstance(repaired, dict) and "nodes" in repaired:
            return repaired, raw
    except Exception:
        pass

    return None, raw
