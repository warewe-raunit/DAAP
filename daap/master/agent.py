"""
DAAP Master Agent — creates and runs the master ReActAgent.

Default provider: OpenRouter (OpenAI-compatible endpoint).
Override via operator_config for custom providers.
"""

import os
from dataclasses import dataclass, field

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel

from daap.executor.tracked_model import TrackedOpenAIChatModel
from daap.master.runtime import build_master_runtime_snapshot
from daap.tools.token_tracker import TokenTracker

from daap.master.prompts import get_master_system_prompt
from daap.master.tools import (
    clear_last_topology_result,
    create_master_toolkit,
    get_last_topology_result,
)
from daap.tools.registry import get_available_tool_names

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MASTER_MODEL = "google/gemini-2.5-flash"  # powerful tier — thinking mode, strong topology generation


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class MasterAgentTurnResult:
    """Result from a single turn of the master agent conversation."""
    response_text: str
    has_topology: bool = False
    topology_dict: dict | None = None
    estimate_data: dict | None = None
    is_asking_questions: bool = False
    pending_questions: list | None = None
    needs_user_input: bool = True


# ---------------------------------------------------------------------------
# Shared model factory
# ---------------------------------------------------------------------------

def _build_model_and_formatter(
    operator_config: dict | None,
    tracker: TokenTracker | None = None,
):
    """
    Return (model, formatter) for the given operator_config.

    Default (no operator_config or provider=="openrouter"):
        OpenRouter via TrackedOpenAIChatModel — uses OPENROUTER_API_KEY.

    Custom override (operator_config with different provider):
        TrackedOpenAIChatModel pointed at that provider's base_url and api_key_env.

    tracker: optional TokenTracker — if provided, each model call records usage.
    """
    if operator_config:
        base_url = operator_config.get("base_url")
        api_key_env = operator_config.get("api_key_env", "OPENROUTER_API_KEY")
        model_id = operator_config.get("model_map", {}).get(
            "powerful", OPENROUTER_MASTER_MODEL
        )

        base_url = base_url or OPENROUTER_BASE_URL
        if "/" not in model_id:
            model_id = f"anthropic/{model_id}"

        client_kwargs = {"base_url": base_url} if base_url else {}
    else:
        # Default: OpenRouter
        api_key_env = "OPENROUTER_API_KEY"
        model_id = OPENROUTER_MASTER_MODEL
        client_kwargs = {"base_url": OPENROUTER_BASE_URL}

    model = TrackedOpenAIChatModel(
        model_name=model_id,
        api_key=os.environ.get(api_key_env, ""),
        client_kwargs=client_kwargs,
        stream=False,
        tracker=tracker,
    )
    return model, OpenAIChatFormatter()


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

def create_master_agent(
    user_context: dict | None = None,
    operator_config: dict | None = None,
    tracker: TokenTracker | None = None,
    runtime_context: dict | None = None,
) -> ReActAgent:
    """
    Create the DAAP master agent with its default module-level toolkit.

    Default: OpenRouter (OPENROUTER_API_KEY).
    Custom: pass operator_config with provider/base_url/api_key_env/model_map.
    tracker: optional TokenTracker for recording token usage.
    """
    toolkit = create_master_toolkit()
    model, formatter = _build_model_and_formatter(operator_config, tracker)
    snapshot = build_master_runtime_snapshot(
        toolkit,
        execution_mode="script",
        extra=runtime_context,
    )

    agent = ReActAgent(
        name="DAAP",
        sys_prompt=get_master_system_prompt(
            available_tools=get_available_tool_names(include_mcp_placeholders=False),
            user_context=user_context,
            runtime_context=snapshot,
        ),
        model=model,
        formatter=formatter,
        memory=InMemoryMemory(),
        toolkit=toolkit,
        max_iters=15,
    )
    # Attach toolkit so parse_turn_result can check per-instance ask_user state
    agent._daap_toolkit = toolkit
    agent._daap_runtime_context = snapshot
    return agent


def create_master_agent_with_toolkit(
    toolkit,
    user_context: dict | None = None,
    operator_config: dict | None = None,
    tracker: TokenTracker | None = None,
    runtime_context: dict | None = None,
) -> ReActAgent:
    """
    Create master agent with an externally provided (session-scoped) toolkit.

    Used by the API layer to inject a session-scoped ask_user closure.
    tracker: optional TokenTracker for recording token usage.
    """
    model, formatter = _build_model_and_formatter(operator_config, tracker)
    snapshot = build_master_runtime_snapshot(
        toolkit,
        execution_mode="api-session",
        extra=runtime_context,
    )

    agent = ReActAgent(
        name="DAAP",
        sys_prompt=get_master_system_prompt(
            available_tools=get_available_tool_names(include_mcp_placeholders=False),
            user_context=user_context,
            runtime_context=snapshot,
        ),
        model=model,
        formatter=formatter,
        memory=InMemoryMemory(),
        toolkit=toolkit,
        max_iters=15,
    )
    agent._daap_toolkit = toolkit
    agent._daap_runtime_context = snapshot
    return agent


# ---------------------------------------------------------------------------
# Turn parsing
# ---------------------------------------------------------------------------

def parse_turn_result(response_msg: Msg, master: ReActAgent) -> MasterAgentTurnResult:
    """
    Parse the master agent's response into a structured turn result.

    Detection is TOOL-BASED, not heuristic:
    - ask_user called   → is_asking_questions=True
    - generate_topology called and succeeded → has_topology=True
    - Neither           → direct response
    """
    text = (
        response_msg.content
        if isinstance(response_msg.content, str)
        else str(response_msg.content)
    )

    toolkit = getattr(master, "_daap_toolkit", None)
    get_pending = getattr(toolkit, "get_pending_questions", None)
    pending_qs = get_pending() if callable(get_pending) else None
    if isinstance(pending_qs, list) and pending_qs:
        return MasterAgentTurnResult(
            response_text=text,
            has_topology=False,
            is_asking_questions=True,
            pending_questions=pending_qs,
            needs_user_input=True,
        )

    topo_result = get_last_topology_result()
    if topo_result.get("topology") is not None:
        result = MasterAgentTurnResult(
            response_text=text,
            has_topology=True,
            topology_dict=topo_result["topology"],
            estimate_data=topo_result["estimate"],
            is_asking_questions=False,
            needs_user_input=True,
        )
        clear_last_topology_result()
        return result

    return MasterAgentTurnResult(
        response_text=text,
        has_topology=False,
        is_asking_questions=False,
        needs_user_input=True,
    )


# ---------------------------------------------------------------------------
# Conversation API
# ---------------------------------------------------------------------------

async def run_master_conversation(
    user_prompt: str,
    user_context: dict | None = None,
    operator_config: dict | None = None,
) -> dict:
    """Start a conversation with the master agent."""
    clear_last_topology_result()
    master = create_master_agent(user_context, operator_config)

    msg = Msg(name="user", content=user_prompt, role="user")
    conversation = [{"role": "user", "content": user_prompt}]

    response_msg = await master(msg)
    turn_result = parse_turn_result(response_msg, master)

    conversation.append({"role": "assistant", "content": turn_result.response_text})

    return {
        "master_agent": master,
        "turn_result": turn_result,
        "conversation": conversation,
    }


async def continue_conversation(master: ReActAgent, user_message: str) -> dict:
    """Continue an ongoing conversation with the master agent."""
    clear_last_topology_result()
    msg = Msg(name="user", content=user_message, role="user")
    response_msg = await master(msg)
    turn_result = parse_turn_result(response_msg, master)

    return {"master_agent": master, "turn_result": turn_result}
