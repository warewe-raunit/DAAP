"""
DAAP Node Builder — converts a ResolvedNode into a live AgentScope agent.

Supports OpenRouter (primary) and any OpenAI-compatible provider
(OpenCode, Ollama, etc.) via provider-aware model selection.

Swap boundary: only this file knows about AgentScope model classes.
engine.py and patterns.py work with BuiltNode and never import AgentScope models.
"""

import asyncio
import functools
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Type

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg, TextBlock
from agentscope.tool import Toolkit, ToolResponse
from pydantic import BaseModel, Field


class NodeResult(BaseModel):
    """Structured output model for react nodes. Call generate_response(result=...) when done."""
    result: str = Field(description="Your complete final output, exactly as specified in your output contract.")

try:
    from agentscope.mcp import MCPToolFunction
except Exception:  # pragma: no cover - optional import guard
    MCPToolFunction = None

from daap.executor.context_manager import BoundedMemory, get_max_input_tokens
from daap.executor.tracked_model import TrackedOpenAIChatModel
from daap.skills.manager import apply_configured_skills
from daap.spec.resolver import ResolvedNode
from daap.tools.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Hard ceiling regardless of what the topology architect generates.
# Prevents runaway loops from burning API credits.
_MAX_REACT_ITERS_HARD_CAP = 10


def _make_dedup_guard(tool_func, tool_name: str):
    """
    Wrap a tool function so that calling it twice with identical arguments
    returns an error instead of executing again.

    Uses functools.wraps so AgentScope's toolkit introspection (inspect.signature,
    __name__, __doc__) sees the original function's schema unchanged.
    """
    seen: set[str] = set()

    def _call_key(args, kwargs) -> str:
        try:
            return json.dumps({"a": args, "k": kwargs}, sort_keys=True, default=str)
        except Exception:
            return repr(args) + repr(kwargs)

    def _dupe_response() -> ToolResponse:
        return ToolResponse(content=[TextBlock(
            type="text",
            text=(
                f"DUPLICATE_CALL_BLOCKED: {tool_name} was already called with these exact "
                f"arguments and returned results. You already have the data. "
                f"Do NOT call any tool again. Call generate_response NOW with your complete output."
            ),
        )])

    if asyncio.iscoroutinefunction(tool_func):
        @functools.wraps(tool_func)
        async def _guarded(*args, **kwargs):
            key = _call_key(args, kwargs)
            if key in seen:
                logger.warning("dedup_guard: blocked duplicate %s call", tool_name)
                return _dupe_response()
            seen.add(key)
            return await tool_func(*args, **kwargs)
    else:
        @functools.wraps(tool_func)
        def _guarded(*args, **kwargs):
            key = _call_key(args, kwargs)
            if key in seen:
                logger.warning("dedup_guard: blocked duplicate %s call", tool_name)
                return _dupe_response()
            seen.add(key)
            return tool_func(*args, **kwargs)

    return _guarded


class TerminatingReActAgent(ReActAgent):
    """
    ReActAgent variant that terminates the reasoning loop on the first successful
    `generate_response` call.

    Why: AgentScope's stock loop continues iterating after a successful structured
    finalization if the reasoning message lacked a text block, and even allows the
    LLM to call `generate_response` repeatedly with stale duplicate payloads when
    structured output validation fails on the first attempt. Both modes burn the
    per-node token budget without producing new information.

    Behavior:
    - Tracks `_finalizer_done` and `_cached_structured_output` per reply.
    - First successful `generate_response` records cached output and flips the flag.
    - Subsequent `generate_response` calls in the same reply return DUPLICATE_FINALIZER_BLOCKED
      while preserving the cached structured payload (so AgentScope's success path still fires).
    - `_reasoning` short-circuits with an empty text Msg once the finalizer fires,
      driving the parent loop into its no-tool-use exit branch on the next iteration.
    - State resets at the start of each `reply` so parallel-fanout reuse is safe.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._finalizer_done: bool = False
        self._finalizer_call_count: int = 0
        self._cached_structured_output: dict | None = None

    def generate_response(self, **kwargs: Any) -> ToolResponse:
        self._finalizer_call_count += 1

        if self._finalizer_done:
            logger.warning(
                "dedup_guard: blocked duplicate generate_response on '%s' (call #%d)",
                self.name,
                self._finalizer_call_count,
            )
            return ToolResponse(
                content=[TextBlock(
                    type="text",
                    text=(
                        "DUPLICATE_FINALIZER_BLOCKED: generate_response was already called "
                        "successfully. The loop is ending now. Do not emit any further output."
                    ),
                )],
                metadata={
                    "success": True,
                    "structured_output": self._cached_structured_output,
                },
                is_last=True,
            )

        result = super().generate_response(**kwargs)
        if result.metadata and result.metadata.get("success"):
            self._finalizer_done = True
            self._cached_structured_output = result.metadata.get("structured_output")
        return result

    async def _reasoning(
        self,
        tool_choice: Literal["auto", "none", "required"] | None = None,
    ) -> Msg:
        if self._finalizer_done:
            # Short-circuit further reasoning. Empty text content + no tool_use
            # makes the parent loop exit via its `not has_content_blocks("tool_use")`
            # branch, attaching the cached structured_output as metadata.
            msg = Msg(
                name=self.name,
                content=[TextBlock(type="text", text="")],
                role="assistant",
            )
            await self.memory.add(msg)
            return msg
        return await super()._reasoning(tool_choice)

    async def reply(
        self,
        msg: Msg | list[Msg] | None = None,
        structured_model: Type[BaseModel] | None = None,
    ) -> Msg:
        # Reset finalizer state per reply so re-used agents (sequential reuse)
        # behave identically to fresh ones. Parallel fan-out already uses
        # agent_factory() so each instance is fresh.
        self._finalizer_done = False
        self._finalizer_call_count = 0
        self._cached_structured_output = None
        return await super().reply(msg, structured_model)


def _tool_id_to_function_name(tool_id: str) -> str:
    """Convert topology tool ID into a stable AgentScope function name."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", tool_id).strip("_")
    if not name:
        name = "tool"
    if name[0].isdigit():
        name = f"tool_{name}"
    return name[:80]


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class BuiltNode:
    """A fully constructed agent ready to execute."""
    node_id: str
    agent: ReActAgent               # live AgentScope agent
    parallel_instances: int         # how many copies to run concurrently
    consolidation_func: str | None  # function path for merging parallel outputs
    consolidation_strategy: str | None  # "merge" | "deduplicate" | "rank" | "vote"
    agent_mode: str                 # "react" or "single"
    # Operator info forwarded for consolidation model selection in patterns.py
    operator_provider: str
    operator_base_url: str | None
    operator_api_key_env: str
    # Optional factory for creating fresh, isolated agent instances.
    # Used by parallel fan-out to avoid shared agent memory/state.
    agent_factory: object | None = None
    # Token tracker — forwarded to consolidation so those LLM calls are counted
    tracker: TokenTracker | None = None
    # Concrete model ID — used for source attribution in node output XML tags
    model_id: str = ""
    # Safe input token budget for this node's model — used in patterns.py for truncation
    max_input_tokens: int = 0


# ---------------------------------------------------------------------------
# Provider-aware model factory
# ---------------------------------------------------------------------------

def _create_model_and_formatter(
    resolved_node: ResolvedNode,
    tracker: TokenTracker | None = None,
):
    """
    Select TrackedOpenAIChatModel + formatter for the given node.
    Routes all calls through OpenRouter (default) or operator-provided base_url.
    """
    model_id = resolved_node.concrete_model_id
    base_url = resolved_node.operator_base_url or OPENROUTER_BASE_URL
    api_key_env = resolved_node.operator_api_key_env or "OPENROUTER_API_KEY"

    if "/" not in model_id:
        model_id = f"anthropic/{model_id}"

    api_key = os.environ.get(api_key_env, "")
    client_kwargs = {"base_url": base_url} if base_url else None

    model = TrackedOpenAIChatModel(
        model_name=model_id,
        api_key=api_key,
        client_kwargs=client_kwargs,
        stream=False,
        generate_kwargs={"temperature": 0, "seed": 42},
        tracker=tracker,
    )
    return model, OpenAIChatFormatter()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def build_node(
    resolved_node: ResolvedNode,
    tool_registry: dict,            # resolved_tool_id → async function
    daap_memory=None,               # optional DaapMemory for system prompt enrichment
    tracker: TokenTracker | None = None,
    user_id: str | None = None,
    today: str | None = None,       # frozen date for the entire execution run
) -> BuiltNode:
    """
    Build a live AgentScope agent from a ResolvedNode.

    Steps:
    1. Optionally enrich system prompt via agent diary (memory)
    2. Build Toolkit with built-in tools via toolkit.register_tool_function()
    3. Create ReActAgent (react or single-shot mode)
    4. Return BuiltNode
    """
    # 1. Use frozen execution date so all nodes in a run see the same date.
    run_date = today or datetime.now().strftime("%Y-%m-%d")
    system_prompt = f"Today's date: {run_date}\n\n{resolved_node.system_prompt}"

    # Optionally enrich system prompt with agent diary
    from daap.memory.reader import load_agent_context_for_node
    if daap_memory and user_id:
        try:
            # client.DaapMemory interface (load_agent_context_for_node)
            if hasattr(daap_memory, "get_agent_learnings"):
                diary_context = load_agent_context_for_node(
                    daap_memory,
                    resolved_node.role,
                    resolved_node.system_prompt[:200],
                )
            else:
                # palace.DaapMemory interface (older path)
                diary_context = daap_memory.format_for_node_prompt(
                    user_id=user_id,
                    role=resolved_node.role,
                    task=resolved_node.system_prompt[:200],
                )
            if diary_context:
                system_prompt = system_prompt + "\n" + diary_context
        except Exception as exc:
            logger.warning("Memory enrichment failed (non-fatal): %s", exc, exc_info=True)

    # 1b. Inject hard system constraints — these are non-negotiable and appended
    #     after the LLM-authored system_prompt so they cannot be overridden by it.
    #     Implements Anthropic/OpenAI best practice: embed effort scaling rules and
    #     output contracts at the harness level, not just in the authored prompt.
    if resolved_node.agent_mode == "react":
        _allowed_tools = [tid.split(".")[-1] for tid in resolved_node.concrete_tool_ids]
        _allowed_str = ", ".join(_allowed_tools) if _allowed_tools else "none"
        system_prompt += (
            f"\n\n[SYSTEM CONSTRAINTS — non-negotiable]"
            f"\n- You have at most {resolved_node.max_react_iterations} tool call rounds total."
            f"\n- When your output contract is satisfied, call `generate_response` with your complete deliverable in the `result` field. This is the ONLY way to submit your final answer and terminate execution."
            f"\n- Do NOT output your result as free text and then call generate_response separately — call generate_response ONCE with the full result. The loop ends immediately when generate_response is called."
            f"\n- Do not make redundant tool calls once you have sufficient data. If a tool returns DUPLICATE_CALL_BLOCKED, you already have that data — call generate_response NOW."
            f"\n- If a search-style tool returns the marker `[STOP_SEARCHING_SIGNAL]` or items tagged `[ALREADY_SEEN_IN_PRIOR_SEARCH]` / `[ALREADY_FETCHED]`, stop issuing more search queries and proceed to call generate_response with the URLs / data you already have."
            f"\n- RedditSearch budget: at most 3 calls per node. After the 3rd call, call generate_response immediately."
            f"\n- Do not explain your search plan. Execute it silently."
            f"\n- If a tool call returns an error (wrong name, missing param, etc.), immediately retry with the correct tool name or fixed arguments. NEVER apologize or give up after a single failure."
            f"\n- You have ONLY these tools available: {_allowed_str}. Do NOT call any other tool (e.g. WebSearch, WebFetch) — they are not registered and will error."
            f"\n- For Reddit URL discovery, use RedditSearch. For one Reddit post use RedditFetch; for multiple candidate URLs use BatchRedditFetch. Do not use WebFetch for reddit.com URLs."
            f"\n- BatchRedditFetch returns a manifest with ACTIVE, REJECTED, and SKIPPED_DUE_TO_SIZE sections. Treat REJECTED and SKIPPED_DUE_TO_SIZE URLs as 'POST_NOT_READ'. NEVER fabricate body text for them."
            f"\n- Treat any post body containing 'I am a bot', 'I am a moderator of this subreddit', 'automatically performed', or 'action was performed automatically' as a bot/automod template. Mark it status='BOT_TEMPLATE' and exclude it from the result list."
            f"\n- If a critical tool fails permanently (same error after 2 retries with corrected args) and you cannot produce your required output, call generate_response with result starting exactly with: EXECUTION_FAILED: <reason>. This signals the pipeline to abort immediately."
            f"\n- NEVER hallucinate or fabricate URLs, content, or data. If a search tool returns zero results and you have NO valid data after your allowed attempts, DO NOT make up results to satisfy your output contract. Instead, call generate_response with result exactly: EXECUTION_FAILED: No valid results found."
        )
    else:
        system_prompt += (
            "\n\n[SYSTEM CONSTRAINTS — non-negotiable]"
            "\n- You have exactly one response turn. There is no follow-up."
            "\n- Begin your response immediately with the deliverable. No preamble, no plan description, no summary of what you are about to do."
            "\n- Do not exceed the length cap specified in your output contract. Trim ruthlessly."
            "\n- If ALL upstream input items are marked 'FAILED_TO_LOAD', 'POST_REMOVED', 'POST_NOT_READ', or 'SKIPPED_DUE_TO_SIZE', output EXACTLY: EXECUTION_FAILED: All upstream content failed to load — cannot produce grounded output."
            "\n- NEVER fabricate, invent, or hallucinate content summaries. If you cannot read the actual content, say so explicitly and emit EXECUTION_FAILED."
            "\n- Treat any item containing 'I am a bot', 'I am a moderator of this subreddit', 'automatically performed', or 'action was performed automatically' as a bot/automod template — exclude it from results. Do NOT score, summarize, or promote bot bodies."
            "\n- For URLs with status 'POST_NOT_READ' / 'SKIPPED_DUE_TO_SIZE' / 'REJECTED', record them with score=0 and reason 'POST_NOT_READ' — never invent body text."
        )

    # 2. Collect built-in tool functions
    builtin_tool_funcs: list[tuple[str, object]] = []

    if resolved_node.agent_mode == "react":
        for tool_id in resolved_node.concrete_tool_ids:
            if tool_id not in tool_registry:
                if tool_id.startswith("mcp://"):
                    logger.warning(
                        "MCP tool '%s' not found in registry for node '%s'; skipping.",
                        tool_id,
                        resolved_node.node_id,
                    )
                    continue
                raise ValueError(
                    f"Tool '{tool_id}' not found in registry for node '{resolved_node.node_id}'. "
                    f"Available: {list(tool_registry.keys())}"
                )
            builtin_tool_funcs.append((tool_id, tool_registry[tool_id]))

    # 3. Agent factory — builds a FRESH toolkit (and therefore fresh
    #    dedup_guard `seen` sets) per spawn. Critical for parallel fan-out:
    #    sharing primary_toolkit across instances would also share dedup
    #    state, causing one instance's tool calls to silently block another's.
    def _spawn_agent() -> ReActAgent:
        model, formatter = _create_model_and_formatter(resolved_node, tracker)

        if resolved_node.agent_mode == "react":
            primary_toolkit = Toolkit()
            apply_configured_skills(primary_toolkit, target="subagent")
            for tool_id, tool_func in builtin_tool_funcs:
                if MCPToolFunction is not None and isinstance(tool_func, MCPToolFunction):
                    primary_toolkit.register_tool_function(
                        tool_func,
                        func_name=_tool_id_to_function_name(tool_id),
                        namesake_strategy="rename",
                    )
                else:
                    # Abstract name (last segment of dotted ID) so the LLM sees
                    # "WebSearch" not "web_search".
                    func_name = tool_id.split(".")[-1]
                    guarded_func = _make_dedup_guard(tool_func, func_name)
                    primary_toolkit.register_tool_function(guarded_func, func_name=func_name)

            capped_iters = min(resolved_node.max_react_iterations, _MAX_REACT_ITERS_HARD_CAP)
            if capped_iters < resolved_node.max_react_iterations:
                logger.warning(
                    "node '%s': max_react_iterations=%d capped to %d",
                    resolved_node.node_id,
                    resolved_node.max_react_iterations,
                    capped_iters,
                )
            return TerminatingReActAgent(
                name=resolved_node.node_id,
                sys_prompt=system_prompt,
                model=model,
                formatter=formatter,
                memory=BoundedMemory(resolved_node.concrete_model_id),
                toolkit=primary_toolkit,
                max_iters=capped_iters,
                parallel_tool_calls=False,
            )

        single_toolkit = Toolkit()
        apply_configured_skills(single_toolkit, target="subagent")
        return TerminatingReActAgent(
            name=resolved_node.node_id,
            sys_prompt=system_prompt,
            model=model,
            formatter=formatter,
            memory=BoundedMemory(resolved_node.concrete_model_id),
            toolkit=single_toolkit,
            max_iters=1,
        )

    agent = _spawn_agent()

    return BuiltNode(
        node_id=resolved_node.node_id,
        agent=agent,
        parallel_instances=resolved_node.parallel_instances,
        consolidation_func=resolved_node.consolidation_func,
        consolidation_strategy=resolved_node.consolidation_strategy,
        agent_mode=resolved_node.agent_mode,
        operator_provider=resolved_node.operator_provider,
        operator_base_url=resolved_node.operator_base_url,
        operator_api_key_env=resolved_node.operator_api_key_env,
        agent_factory=_spawn_agent,
        tracker=tracker,
        model_id=resolved_node.concrete_model_id,
        max_input_tokens=get_max_input_tokens(resolved_node.concrete_model_id),
    )
