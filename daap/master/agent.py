"""
DAAP Master Agent — creates and runs the master ReActAgent.

Default provider: OpenRouter (OpenAI-compatible endpoint).
Override via operator_config for custom providers.
"""

import os
from dataclasses import dataclass, field

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel

from daap.executor.context_manager import BoundedMemory
from daap.executor.tracked_model import TrackedOpenAIChatModel
from daap.master.runtime import build_master_runtime_snapshot
from daap.tools.token_tracker import TokenTracker

from daap.master.planner import plan_turn
from daap.master.prompts import get_master_system_prompt
from daap.master.tools import create_master_toolkit
from daap.tools.registry import get_available_tool_names

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MASTER_MODEL = "qwen/qwen3.6-plus"  # powerful tier — thinking mode, strong topology generation


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
        generate_kwargs={"temperature": 0.2},  # seed removed — identical inputs repeat same broken JSON on retry
        tracker=tracker,
    )
    return model, OpenAIChatFormatter(), model_id


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
    model, formatter, model_id = _build_model_and_formatter(operator_config, tracker)
    snapshot = build_master_runtime_snapshot(
        toolkit,
        execution_mode="script",
        extra=runtime_context,
    )

    # Inject grounding context into the architect — see topology_agent.py.
    set_ctx = getattr(toolkit, "set_architect_context", None)
    if callable(set_ctx):
        set_ctx(
            user_context=user_context,
            runtime_context=snapshot,
            operator_config=operator_config,
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
        memory=BoundedMemory(model_id),
        toolkit=toolkit,
        max_iters=12,
    )
    # Attach toolkit so parse_turn_result can check per-instance ask_user state
    agent._daap_toolkit = toolkit
    agent._daap_runtime_context = snapshot
    agent._daap_base_sys_prompt = agent._sys_prompt  # frozen base for per-turn hint injection
    # Store model/formatter for plan_turn reuse (no extra credentials needed)
    agent._daap_model = model
    agent._daap_formatter = formatter
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
    model, formatter, model_id = _build_model_and_formatter(operator_config, tracker)
    snapshot = build_master_runtime_snapshot(
        toolkit,
        execution_mode="api-session",
        extra=runtime_context,
    )

    # Inject grounding context into the architect — see topology_agent.py.
    set_ctx = getattr(toolkit, "set_architect_context", None)
    if callable(set_ctx):
        set_ctx(
            user_context=user_context,
            runtime_context=snapshot,
            operator_config=operator_config,
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
        memory=BoundedMemory(model_id),
        toolkit=toolkit,
        max_iters=12,
    )
    agent._daap_toolkit = toolkit
    agent._daap_runtime_context = snapshot
    agent._daap_base_sys_prompt = agent._sys_prompt  # frozen base for per-turn hint injection
    agent._daap_model = model
    agent._daap_formatter = formatter
    return agent


# ---------------------------------------------------------------------------
# Plan injection
# ---------------------------------------------------------------------------

def _inject_plan_hint(agent: ReActAgent, plan) -> None:
    """
    Write the plan hint into the system prompt for this turn.

    Injecting into sys_prompt (not the user message) keeps the user turn
    clean conversational input. The model attends to system-prompt guidance
    during reasoning, not as part of the user's request.
    """
    base = getattr(agent, "_daap_base_sys_prompt", agent._sys_prompt)
    agent._daap_base_sys_prompt = base  # idempotent — ensure stored
    agent._sys_prompt = base + plan.hint()  # property has no setter; write the backing attr
    agent.max_iters = plan.max_iters


# ---------------------------------------------------------------------------
# Turn parsing
# ---------------------------------------------------------------------------

def parse_turn_result(response_msg: Msg, master: ReActAgent) -> MasterAgentTurnResult:
    """
    Parse the master agent's response into a structured turn result.

    Detection is TOOL-BASED, not heuristic:
    - ask_user called             → is_asking_questions=True
    - delegate_to_architect succeeded → has_topology=True
    - Neither                     → direct response
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

    get_topo = getattr(toolkit, "get_last_topology_result", None)
    topo_result = get_topo() if callable(get_topo) else {"topology": None, "estimate": None}
    if topo_result.get("topology") is not None:
        result = MasterAgentTurnResult(
            response_text=text,
            has_topology=True,
            topology_dict=topo_result["topology"],
            estimate_data=topo_result["estimate"],
            is_asking_questions=False,
            needs_user_input=True,
        )
        clear_topo = getattr(toolkit, "clear_last_topology_result", None)
        if callable(clear_topo):
            clear_topo()
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
    model, formatter, _ = _build_model_and_formatter(operator_config)
    master = create_master_agent(user_context, operator_config)
    clear_topo = getattr(master._daap_toolkit, "clear_last_topology_result", None)
    if callable(clear_topo):
        clear_topo()

    # Plan-and-execute: classify intent before the ReAct loop so the executor
    # arrives with a concrete plan instead of deliberating from scratch.
    plan = await plan_turn(user_prompt, model, formatter)
    _inject_plan_hint(master, plan)  # hint → sys_prompt; also sets max_iters

    msg = Msg(name="user", content=user_prompt, role="user")
    conversation = [{"role": "user", "content": user_prompt}]

    response_msg = await master(msg)
    turn_result = parse_turn_result(response_msg, master)

    conversation.append({"role": "assistant", "content": turn_result.response_text})

    return {
        "master_agent": master,
        "turn_result": turn_result,
        "conversation": conversation,
        "plan": plan,
    }


async def continue_conversation(master: ReActAgent, user_message: str) -> dict:
    """Continue an ongoing conversation with the master agent."""
    clear_topo = getattr(master._daap_toolkit, "clear_last_topology_result", None)
    if callable(clear_topo):
        clear_topo()

    # Reuse model/formatter stored on agent — no extra credentials needed.
    model = getattr(master, "_daap_model", None)
    formatter = getattr(master, "_daap_formatter", None)
    if model is not None and formatter is not None:
        plan = await plan_turn(user_message, model, formatter)
        _inject_plan_hint(master, plan)  # hint → sys_prompt; also sets max_iters
    else:
        plan = None

    msg = Msg(name="user", content=user_message, role="user")
    response_msg = await master(msg)
    turn_result = parse_turn_result(response_msg, master)

    return {"master_agent": master, "turn_result": turn_result, "plan": plan}
