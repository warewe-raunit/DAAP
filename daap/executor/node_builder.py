"""
DAAP Node Builder — converts a ResolvedNode into a live AgentScope agent.

Supports OpenRouter (primary) and any OpenAI-compatible provider
(OpenCode, Ollama, etc.) via provider-aware model selection.

Swap boundary: only this file knows about AgentScope model classes.
engine.py and patterns.py work with BuiltNode and never import AgentScope models.
"""

import logging
import os
import re
from dataclasses import dataclass

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

try:
    from agentscope.mcp import MCPToolFunction
except Exception:  # pragma: no cover - optional import guard
    MCPToolFunction = None

from daap.executor.tracked_model import TrackedOpenAIChatModel
from daap.skills.manager import apply_configured_skills
from daap.spec.resolver import ResolvedNode
from daap.tools.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


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
) -> BuiltNode:
    """
    Build a live AgentScope agent from a ResolvedNode.

    Steps:
    1. Optionally enrich system prompt via agent diary (memory)
    2. Build Toolkit with built-in tools via toolkit.register_tool_function()
    3. Create ReActAgent (react or single-shot mode)
    4. Return BuiltNode
    """
    # 1. Optionally enrich system prompt with agent diary
    from daap.memory.reader import load_agent_context_for_node

    system_prompt = resolved_node.system_prompt
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

    # 3. Build the primary toolkit
    primary_toolkit = Toolkit()
    apply_configured_skills(primary_toolkit, target="subagent")
    if resolved_node.agent_mode == "react":
        for tool_id, tool_func in builtin_tool_funcs:
            if MCPToolFunction is not None and isinstance(tool_func, MCPToolFunction):
                primary_toolkit.register_tool_function(
                    tool_func,
                    func_name=_tool_id_to_function_name(tool_id),
                    namesake_strategy="rename",
                )
            else:
                primary_toolkit.register_tool_function(tool_func)

    single_toolkit = Toolkit()
    apply_configured_skills(single_toolkit, target="subagent")

    # 4. Agent factory — sync, reuses same toolkit for parallel fan-out.
    def _spawn_agent() -> ReActAgent:
        model, formatter = _create_model_and_formatter(resolved_node, tracker)

        if resolved_node.agent_mode == "react":
            return ReActAgent(
                name=resolved_node.node_id,
                sys_prompt=system_prompt,
                model=model,
                formatter=formatter,
                memory=InMemoryMemory(),
                toolkit=primary_toolkit,
                max_iters=resolved_node.max_react_iterations,
                parallel_tool_calls=False,
            )

        return ReActAgent(
            name=resolved_node.node_id,
            sys_prompt=system_prompt,
            model=model,
            formatter=formatter,
            memory=InMemoryMemory(),
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
    )
