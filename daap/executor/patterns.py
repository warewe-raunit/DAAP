"""
DAAP Execution Patterns — parallel fan-out, consolidation, and step execution.

Implements the three patterns the engine uses:
  1. run_parallel_instances  — fan-out: N copies of same agent on same input
  2. consolidate_outputs     — fan-in: merge N outputs into one
  3. run_execution_step      — one step of the execution_order DAG walk

engine.py is framework-agnostic above this level.
This file is the lowest layer that knows about Msg objects.
"""

import asyncio
from collections import Counter

from agentscope.message import Msg

from daap.executor.node_builder import BuiltNode


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# 1. Parallel fan-out
# ---------------------------------------------------------------------------

async def run_parallel_instances(
    built_node: BuiltNode,
    input_msg: Msg,
) -> list[Msg]:
    """
    Run N parallel instances of the same agent on the same input.

    If parallel_instances == 1: runs once, returns [result].
    If parallel_instances > 1: asyncio.gather for true concurrency.
    Each call is independent — same agent object, separate async tasks.
    """
    n = built_node.parallel_instances

    def _clone_input(msg: Msg) -> Msg:
        return Msg(name=msg.name, content=msg.content, role=msg.role)

    if n == 1:
        result = await built_node.agent(_clone_input(input_msg))
        return [result]

    tasks = []
    for _ in range(n):
        # Prefer fresh agents for each parallel instance to prevent
        # cross-instance memory/state contamination.
        if built_node.agent_factory is not None:
            agent_instance = built_node.agent_factory()
            tasks.append(agent_instance(_clone_input(input_msg)))
        else:
            # Backward-compatible fallback for tests/mocks that do not set a factory.
            tasks.append(built_node.agent(_clone_input(input_msg)))

    results = await asyncio.gather(*tasks, return_exceptions=False)
    return list(results)


# ---------------------------------------------------------------------------
# 2. Consolidation
# ---------------------------------------------------------------------------

async def consolidate_outputs(
    outputs: list[Msg],
    strategy: str,
    consolidation_model_id: str | None = None,
    operator_provider: str = "openrouter",
    operator_base_url: str | None = None,
    operator_api_key_env: str = "OPENROUTER_API_KEY",
) -> Msg:
    """
    Merge N parallel outputs into one Msg using the specified strategy.

    merge       — simple concatenation, no LLM call
    vote        — majority vote on first word/line, no LLM call
    deduplicate — LLM call (fast model): remove duplicates, keep unique items
    rank        — LLM call (smart model): rank by quality, best first

    Args:
        outputs:               list of Msg from parallel instances
        strategy:              "merge" | "deduplicate" | "rank" | "vote"
        consolidation_model_id: concrete model ID for LLM-based strategies
        operator_provider:     "openrouter" or openai-compatible provider
        operator_base_url:     base URL override (for OpenRouter etc.)
        operator_api_key_env:  env var name for API key
    """
    if not outputs:
        return Msg(name="consolidator", content="", role="assistant")

    if len(outputs) == 1:
        return outputs[0]

    texts = [o.content if isinstance(o.content, str) else str(o.content)
             for o in outputs]

    if strategy == "merge":
        return _merge(texts)

    if strategy == "vote":
        return _vote(texts)

    if strategy in ("deduplicate", "rank"):
        return await _llm_consolidate(
            texts, strategy,
            consolidation_model_id,
            operator_provider, operator_base_url, operator_api_key_env,
        )

    # Unknown strategy — fall back to merge
    return _merge(texts)


def _merge(texts: list[str]) -> Msg:
    combined = "\n---\n".join(texts)
    return Msg(name="consolidator", content=combined, role="assistant")


def _vote(texts: list[str]) -> Msg:
    """Majority vote: most common first line/word wins."""
    votes = [t.strip().split("\n")[0].strip() for t in texts]
    winner = Counter(votes).most_common(1)[0][0]
    return Msg(name="consolidator", content=winner, role="assistant")


async def _llm_consolidate(
    texts: list[str],
    strategy: str,
    model_id: str | None,
    provider: str,
    base_url: str | None,
    api_key_env: str,
) -> Msg:
    """
    LLM-based consolidation (deduplicate or rank).
    Uses a lightweight model call — provider-aware.
    Phase 1: if model_id is None, falls back to merge.
    """
    if model_id is None:
        return _merge(texts)

    import os
    combined = "\n---\n".join(texts)

    if strategy == "deduplicate":
        prompt = (
            "Remove duplicate information from these results. "
            "Keep unique items only. Preserve all details from unique items.\n\n"
            f"{combined}"
        )
    else:  # rank
        prompt = (
            "Rank these results by quality and relevance. "
            "Return the ranked list with the best results first. "
            "Remove low-quality items.\n\n"
            f"{combined}"
        )

    try:
        from agentscope.model import OpenAIChatModel
        from agentscope.formatter import OpenAIChatFormatter

        base_url = base_url or OPENROUTER_BASE_URL
        api_key_env = api_key_env or "OPENROUTER_API_KEY"

        # Prefix bare Claude model IDs with provider namespace for OpenRouter.
        if "/" not in model_id:
            model_id = f"anthropic/{model_id}"

        client_kwargs = {"base_url": base_url} if base_url else None
        model = OpenAIChatModel(
            model_name=model_id,
            api_key=os.environ.get(api_key_env, ""),
            client_kwargs=client_kwargs,
            stream=False,
        )
        formatter = OpenAIChatFormatter()

        msgs = [{"role": "user", "content": prompt}]
        formatted = formatter.format_messages(msgs)
        response = await model(formatted)
        result_text = getattr(response, "content", None) or getattr(response, "text", None)
        if result_text is None:
            result_text = str(response)
        return Msg(name="consolidator", content=result_text, role="assistant")

    except Exception as e:
        # Consolidation LLM call failed — fall back to merge
        fallback = _merge(texts)
        fallback.content = f"[Consolidation failed: {e}]\n\n" + str(fallback.content)
        return fallback


# ---------------------------------------------------------------------------
# 3. Step execution
# ---------------------------------------------------------------------------

async def run_execution_step(
    built_nodes: list[BuiltNode],
    input_data: dict[str, Msg],
    edges: list,
    initial_msg: Msg,
) -> dict[str, Msg]:
    """
    Execute one step of the DAG (all nodes in this step run concurrently).

    For each node:
    1. Build input Msg from incoming edges in input_data
       (if no incoming edges → use initial_msg, the user's prompt)
    2. run_parallel_instances → list of outputs
    3. consolidate if parallel_instances > 1
    4. Store result keyed by the node's first output data_key

    All nodes in the step run concurrently via asyncio.gather.
    Returns updated input_data dict (accumulates all outputs for future steps).
    """
    async def _run_one(built_node: BuiltNode) -> tuple[str, Msg]:
        # Find incoming edges for this node
        incoming = [e for e in edges if e.target_node_id == built_node.node_id]

        if not incoming:
            # First node — receives user prompt directly
            node_input = initial_msg
        else:
            # Gather all inputs from prior step outputs
            parts = []
            for edge in incoming:
                if edge.data_key in input_data:
                    content = input_data[edge.data_key].content
                    parts.append(content if isinstance(content, str) else str(content))
            combined = "\n\n".join(parts) if parts else initial_msg.content
            node_input = Msg(name="user", content=combined, role="user")

        # Fan-out
        outputs = await run_parallel_instances(built_node, node_input)

        # Consolidate if needed
        if len(outputs) > 1 and built_node.consolidation_func:
            strategy = built_node.consolidation_strategy or "merge"
            # Phase 1: use a fixed model per strategy, same provider as node
            from daap.spec.resolver import MODEL_REGISTRY
            if strategy == "deduplicate":
                cons_model = MODEL_REGISTRY.get("fast")
            else:
                cons_model = MODEL_REGISTRY.get("smart")

            result = await consolidate_outputs(
                outputs, strategy, cons_model,
                built_node.operator_provider,
                built_node.operator_base_url,
                built_node.operator_api_key_env,
            )
        else:
            result = outputs[0]

        # Output key = first declared output of this node
        # (We don't have resolved_node here, but BuiltNode doesn't carry outputs.
        #  The engine passes output_key separately — see engine.py.)
        return built_node.node_id, result

    # Run all nodes in this step concurrently
    tasks = [_run_one(bn) for bn in built_nodes]
    step_results = await asyncio.gather(*tasks)

    # Merge into input_data — caller maps node_id → output_key
    updated = dict(input_data)
    for node_id, msg in step_results:
        # Key by node_id for now; engine.py remaps to data_key
        updated[node_id] = msg

    return updated
