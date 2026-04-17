"""
DAAP Spec Schema — Pydantic v2 models for the DAAP spec language.

This is the STABLE CONTRACT between the master agent (which generates topologies)
and the execution engine (which runs them). All other modules import from here.

Provider-agnostic by design: ModelTier is abstract, OperatorConfig maps tiers
to concrete model IDs for any LLM operator (Anthropic, OpenRouter, OpenCode, etc.).
"""

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelTier(str, Enum):
    """Abstract model tier. Resolver maps these to concrete model IDs per operator."""
    FAST = "fast"           # search, extract, format tasks
    SMART = "smart"         # reason, evaluate, synthesize tasks
    POWERFUL = "powerful"   # complex planning, master agent


class HandoffMode(str, Enum):
    """Controls how nodes delegate to each other. From Oracle Agent Spec."""
    NEVER = "never"         # Agent-as-Tool: call → get result → retain control
    OPTIONAL = "optional"   # Hybrid: decide at runtime
    ALWAYS = "always"       # Coordinator: transfer full context, caller exits


class ConsolidationStrategy(str, Enum):
    """How to merge outputs when a node has multiple parallel instances."""
    MERGE = "merge"             # concatenate all instance outputs
    DEDUPLICATE = "deduplicate" # LLM-based dedup across outputs
    RANK = "rank"               # LLM-based ranking of outputs
    VOTE = "vote"               # majority-vote (for classification tasks)


class AgentMode(str, Enum):
    """
    Controls whether a node's agent uses ReAct (reason-act-observe loop)
    or single-shot execution.

    WHY THIS EXISTS: AgentScope's ReActAgent runs an iterative loop —
    it reasons, calls a tool, observes the result, and repeats until done.
    This is essential for research/exploration nodes that need to search,
    read results, refine their query, and search again. But it's wasteful
    for simple nodes like an email drafter that just takes input and
    produces output in a single pass.

    The master agent should decide this per-node based on task requirements.
    No other framework exposes this as a per-node config.

    Maps to AgentScope:
      REACT  → agentscope.agent.ReActAgent (iterative tool-calling loop)
      SINGLE → Simple single-pass agent (one LLM call, no tool loop)
    """
    REACT = "react"     # iterative reason-act-observe loop with tools
    SINGLE = "single"   # single-pass: one LLM call, return output


# ---------------------------------------------------------------------------
# Operator / Provider config
# ---------------------------------------------------------------------------

class OperatorConfig(BaseModel):
    """
    LLM operator configuration. Allows DAAP to run against any OpenAI-compatible
    or Anthropic-compatible provider — Anthropic, OpenRouter, OpenCode, etc.

    The resolver reads this to map ModelTier → concrete model ID for this operator.

    Fields:
        provider:     Human-readable provider name (e.g. "anthropic", "openrouter", "opencode").
        base_url:     API base URL. None = use provider SDK default.
        api_key_env:  Name of the environment variable holding the API key.
        model_map:    Maps ModelTier values to concrete model IDs for this operator.
                      Example (OpenRouter):
                        {"fast": "meta-llama/llama-3-8b-instruct",
                         "smart": "anthropic/claude-3-5-sonnet",
                         "powerful": "anthropic/claude-opus-4"}
    """
    provider: str                           # e.g. "anthropic", "openrouter", "opencode"
    base_url: str | None = None             # None = SDK default for that provider
    api_key_env: str = "OPENROUTER_API_KEY" # env var name for the API key
    model_map: dict[str, str] = {}          # ModelTier value → concrete model ID

    @field_validator("provider")
    @classmethod
    def provider_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("provider must not be empty")
        return v.strip().lower()


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

class ToolBinding(BaseModel):
    """Abstract tool reference. Resolver maps to concrete tool instance."""
    name: str  # e.g. "WebSearch", "WebFetch", "mcp://linkedin/search_people"


class IOSchema(BaseModel):
    """Typed input/output definition for a node. Used for edge compatibility checks."""
    data_key: str       # e.g. "raw_leads", "qualified_leads"
    data_type: str      # e.g. "list[Lead]", "string", "list[Email]"
    description: str    # human-readable description of what this data is


class InstanceConfig(BaseModel):
    """
    Scaling configuration for a node. Separate from NodeSpec by design.

    WHY SEPARATE: Phase 2 RL needs to tune instance counts independently
    without touching the node's role/prompt/tools. If merged, changing
    instance count = changing node definition = messy version history.
    This is DAAP's original contribution — no other framework does this.
    """
    parallel_instances: int = 1
    consolidation: ConsolidationStrategy | None = None  # required when parallel_instances > 1

    @model_validator(mode="after")
    def consolidation_required_for_parallel(self) -> "InstanceConfig":
        if self.parallel_instances > 1 and self.consolidation is None:
            raise ValueError("consolidation strategy required when parallel_instances > 1")
        return self


class NodeSpec(BaseModel):
    """
    One logical agent role in the topology.

    Defines WHAT an agent does — role, model, prompt, tools, I/O.
    Does NOT define HOW MANY instances run (that's InstanceConfig).
    """
    node_id: str                                            # unique identifier within topology
    role: str                                               # human-readable role name
    model_tier: ModelTier                                   # abstract tier (resolver → concrete ID)
    system_prompt: str                                      # the agent's system prompt
    tools: list[ToolBinding] = []                           # tools this node can use
    inputs: list[IOSchema] = []                             # data this node consumes
    outputs: list[IOSchema]                                 # data this node produces (≥1)
    instance_config: InstanceConfig                         # parallel scaling config
    handoff_mode: HandoffMode = HandoffMode.NEVER           # default: Agent-as-Tool
    operator_override: OperatorConfig | None = None         # per-node provider override
    agent_mode: AgentMode = AgentMode.REACT                 # default: ReAct (most nodes use tools)
    max_react_iterations: int = 10                          # safety cap for ReAct loop

    @field_validator("node_id")
    @classmethod
    def node_id_valid_identifier(cls, v: str) -> str:
        if not re.match(r"^[a-z][a-z0-9_]*$", v):
            raise ValueError(
                f"node_id '{v}' invalid — must match ^[a-z][a-z0-9_]*$ "
                "(lowercase, start with letter, alphanumeric + underscore only)"
            )
        return v

    @field_validator("outputs")
    @classmethod
    def outputs_nonempty(cls, v: list[IOSchema]) -> list[IOSchema]:
        if not v:
            raise ValueError("every node must produce at least one output")
        return v

    @model_validator(mode="after")
    def react_requires_tools(self) -> "NodeSpec":
        if self.agent_mode == AgentMode.REACT and not self.tools:
            raise ValueError(
                f"agent_mode is 'react' but no tools provided. Use 'single' instead."
            )
        return self


class EdgeSpec(BaseModel):
    """
    Data flow connection between two nodes.

    Explicit data flow. Validator uses edges to check:
    - Source node exists
    - Target node exists
    - Source output type compatible with target input type (Section 2)
    """
    source_node_id: str     # which node produces the data
    target_node_id: str     # which node consumes the data
    data_key: str           # output key from source → input key on target
    description: str = ""   # optional human-readable description


class ConstraintSpec(BaseModel):
    """
    Execution boundaries. Prevents master agent from generating $100/run topologies.
    Sensible defaults for non-technical users.
    """
    max_cost_usd: float = 1.00
    max_latency_seconds: float = 120.0
    max_nodes: int = 10
    max_total_instances: int = 20
    max_retries_per_node: int = 2
    max_tokens_per_node: int = 50_000   # per-node token budget (input + output)


class TopologySpec(BaseModel):
    """
    The complete multi-agent system for one task.

    Single serializable JSON unit. Master agent generates it.
    Spec block validates it. Execution engine consumes it.
    This is the STABLE CONTRACT between intelligence and execution.

    Includes metadata for versioning (Phase 2 needs this for
    per-user spec evolution tracking).
    """
    topology_id: str                                        # unique ID (UUID at creation)
    version: int = 1                                        # spec version (Phase 2 increments)
    created_at: str                                         # ISO 8601 timestamp
    user_prompt: str                                        # original natural language input
    nodes: list[NodeSpec]                                   # all agent nodes
    edges: list[EdgeSpec]                                   # all data flow connections
    constraints: ConstraintSpec = ConstraintSpec()          # execution boundaries
    operator_config: OperatorConfig | None = None           # default operator for all nodes
    metadata: dict[str, Any] = {}                           # extensible for future use

    @model_validator(mode="after")
    def validate_topology(self) -> "TopologySpec":
        # 1. nodes must not be empty
        if not self.nodes:
            raise ValueError("topology must have at least one node")

        node_ids = [n.node_id for n in self.nodes]

        # 2. node_ids must be unique
        seen: set[str] = set()
        duplicates: list[str] = []
        for nid in node_ids:
            if nid in seen:
                duplicates.append(nid)
            seen.add(nid)
        if duplicates:
            raise ValueError(f"duplicate node_ids found: {duplicates}")

        # 3. edge references must point to existing nodes
        node_id_set = set(node_ids)
        broken: list[str] = []
        for edge in self.edges:
            if edge.source_node_id not in node_id_set:
                broken.append(f"source '{edge.source_node_id}'")
            if edge.target_node_id not in node_id_set:
                broken.append(f"target '{edge.target_node_id}'")
        if broken:
            raise ValueError(f"edges reference non-existent nodes: {broken}")

        return self


# ---------------------------------------------------------------------------
# Schema export
# ---------------------------------------------------------------------------

def get_topology_json_schema() -> dict:
    """
    Returns the JSON schema for TopologySpec.

    This schema is injected into the master agent's system prompt
    so the LLM knows exactly what valid JSON looks like.
    """
    return TopologySpec.model_json_schema()
