"""
DAAP Node Builder — converts a ResolvedNode into a live AgentScope agent.

Supports OpenRouter (primary) and any OpenAI-compatible provider
(OpenCode, Ollama, etc.) via provider-aware model selection.

Swap boundary: only this file knows about AgentScope model classes.
engine.py and patterns.py work with BuiltNode and never import AgentScope models.
"""

import os
from dataclasses import dataclass

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

from daap.executor.tracked_model import TrackedOpenAIChatModel
from daap.spec.resolver import ResolvedNode
from daap.tools.token_tracker import TokenTracker


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


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
    agent_mode: str                 # "react" or "single"
    # Operator info forwarded for consolidation model selection in patterns.py
    operator_provider: str
    operator_base_url: str | None
    operator_api_key_env: str
    # Optional factory for creating fresh, isolated agent instances.
    # Used by parallel fan-out to avoid shared agent memory/state.
    agent_factory: object | None = None


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
    tracker: optional TokenTracker — records input/output tokens per call.
    """
    model_id = resolved_node.concrete_model_id
    base_url = resolved_node.operator_base_url or OPENROUTER_BASE_URL
    api_key_env = resolved_node.operator_api_key_env or "OPENROUTER_API_KEY"

    # Prefix bare Claude model IDs with provider namespace for OpenRouter.
    if "/" not in model_id:
        model_id = f"anthropic/{model_id}"

    api_key = os.environ.get(api_key_env, "")
    client_args = {"base_url": base_url} if base_url else None

    model = TrackedOpenAIChatModel(
        model_name=model_id,
        api_key=api_key,
        client_args=client_args,
        stream=False,
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
    tracker: TokenTracker | None = None,  # optional TokenTracker for usage recording
) -> BuiltNode:
    """
    Build a live AgentScope agent from a ResolvedNode.

    Steps:
    1. Create model + formatter based on operator provider
    2. Build Toolkit from resolved tool IDs (skips MCP tools with warning in Phase 1)
    3. Create ReActAgent:
       - react mode: full toolkit, max_iters from spec
       - single mode: empty toolkit, max_iters=1 (one LLM call, no tool loop)
    4. Return BuiltNode wrapping agent + metadata
    """
    # Optionally enrich system prompt with agent diary learnings
    system_prompt = resolved_node.system_prompt
    if daap_memory:
        try:
            from daap.memory.reader import load_agent_context_for_node
            learnings = load_agent_context_for_node(
                daap_memory,
                agent_role=resolved_node.role.lower(),
                task_description=resolved_node.system_prompt[:200],
            )
            if learnings:
                system_prompt = f"{learnings}\n\n{system_prompt}"
        except Exception:
            pass  # memory enrichment is optional — never break execution

    # Pre-resolve tool functions once so fresh agents can be spawned cheaply.
    resolved_tool_funcs: list[object] = []
    if resolved_node.agent_mode == "react":
        for tool_id in resolved_node.concrete_tool_ids:
            if tool_id.startswith("mcp://"):
                # MCP tools resolved at runtime — skip Phase 1, warn
                # TODO: Phase 2 — HttpStatelessClient MCP integration
                print(
                    f"[DAAP] Warning: MCP tool '{tool_id}' skipped in Phase 1. "
                    f"Add MCP client support in Phase 2."
                )
                continue

            if tool_id not in tool_registry:
                raise ValueError(
                    f"Tool '{tool_id}' not found in registry for node '{resolved_node.node_id}'. "
                    f"Available: {list(tool_registry.keys())}"
                )
            resolved_tool_funcs.append(tool_registry[tool_id])

    def _build_toolkit_instance() -> Toolkit:
        toolkit = Toolkit()
        if resolved_node.agent_mode == "react":
            for tool_func in resolved_tool_funcs:
                toolkit.register_tool_function(tool_func)
        return toolkit

    def _spawn_agent() -> ReActAgent:
        # Fresh model + memory per instance prevents crosstalk in parallel fan-out.
        model, formatter = _create_model_and_formatter(resolved_node, tracker)

        if resolved_node.agent_mode == "react":
            return ReActAgent(
                name=resolved_node.node_id,
                sys_prompt=system_prompt,
                model=model,
                formatter=formatter,
                memory=InMemoryMemory(),
                toolkit=_build_toolkit_instance(),
                max_iters=resolved_node.max_react_iterations,
                parallel_tool_calls=False,  # Gemini strict: tool call/result count must match
            )

        # Single-shot: ReActAgent with max_iters=1, empty toolkit.
        # One LLM call → return. No tool loop.
        return ReActAgent(
            name=resolved_node.node_id,
            sys_prompt=system_prompt,
            model=model,
            formatter=formatter,
            memory=InMemoryMemory(),
            toolkit=Toolkit(),
            max_iters=1,
        )

    # Primary agent instance (single-instance nodes use this directly).
    agent = _spawn_agent()

    return BuiltNode(
        node_id=resolved_node.node_id,
        agent=agent,
        parallel_instances=resolved_node.parallel_instances,
        consolidation_func=resolved_node.consolidation_func,
        agent_mode=resolved_node.agent_mode,
        operator_provider=resolved_node.operator_provider,
        operator_base_url=resolved_node.operator_base_url,
        operator_api_key_env=resolved_node.operator_api_key_env,
        agent_factory=_spawn_agent,
    )
