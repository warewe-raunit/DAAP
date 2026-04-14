# DAAP — Project Context

**DAAP** (Dynamic Agent Automation Platform) is a B2B sales automation system where a **Master Agent** designs and executes multi-agent pipelines (called **topologies**) based on natural language user requests. Think of it as: user describes a sales task → master agent designs a DAG of specialized AI agents → user approves → engine executes.

---

## Quick Reference

| Thing | Value |
|-------|-------|
| Language | Python 3.11+ |
| Agent framework | AgentScope (`agentscope`) |
| LLM provider | OpenRouter (`https://openrouter.ai/api/v1`) |
| Default model | `google/gemini-2.0-flash-001` (all tiers) |
| API key env var | `OPENROUTER_API_KEY` |
| Web server | FastAPI + uvicorn, port 8000 |
| Entrypoint (API) | `uvicorn daap.main:app --reload --port 8000` |
| Entrypoint (CLI) | `python scripts/chat.py` |
| E2E test | `python scripts/e2e_test.py` |
| Unit tests | `pytest daap/tests/` |

---

## Architecture Overview

```
User
 │
 ▼
Master Agent (ReActAgent)
 │  tools: generate_topology, ask_user, get_execution_status
 │
 ├── ask_user → waits for user to answer questions
 ├── generate_topology → validates + estimates → stores pending topology
 │
 ▼
User approves ("approve" / "run it" / "go ahead" etc.)
 │
 ▼
Execution Engine (execute_topology)
 │
 ├── build_node() → ResolvedNode → BuiltNode (live AgentScope agent)
 │
 └── Walk execution_order (topological sort, parallel groups)
      For each step:
        run_execution_step()
          └── run_parallel_instances() → consolidate_outputs() → Msg
 │
 ▼
ExecutionResult (final_output, node_results, tokens, latency)
```

---

## Directory Structure

```
DAAP/
├── daap/
│   ├── main.py                    # FastAPI app entrypoint
│   ├── api/
│   │   ├── routes.py              # REST endpoints + WebSocket mount
│   │   ├── ws_handler.py          # WebSocket conversation handler
│   │   └── sessions.py            # Session dataclass + SessionManager + session-scoped toolkit
│   ├── master/
│   │   ├── agent.py               # Master ReActAgent factory
│   │   ├── prompts.py             # Master agent system prompt (injected with schema + tools)
│   │   └── tools.py               # generate_topology, ask_user tools + module-level state
│   ├── spec/
│   │   ├── schema.py              # Pydantic models: TopologySpec, NodeSpec, EdgeSpec, etc.
│   │   ├── resolver.py            # Abstract → concrete IDs, topological sort (execution_order)
│   │   ├── validator.py           # Business rule validation (DAG check, tool availability, etc.)
│   │   └── estimator.py           # Cost + latency estimation per topology
│   ├── executor/
│   │   ├── engine.py              # Top-level execute_topology() orchestrator
│   │   ├── node_builder.py        # ResolvedNode → BuiltNode (live AgentScope agent)
│   │   ├── patterns.py            # run_parallel_instances, consolidate_outputs, run_execution_step
│   │   └── tracked_model.py       # TrackedOpenAIChatModel — subclass that records token usage
│   ├── tools/
│   │   ├── registry.py            # Tool implementations: web_search, web_fetch, read_file, etc.
│   │   └── token_tracker.py       # TokenTracker — per-session input/output token accumulator
│   ├── feedback/
│   │   ├── store.py               # SQLite FeedbackStore
│   │   └── collector.py           # collect_run_feedback()
│   ├── memory/
│   │   ├── palace.py              # DaapMemory (mem0ai wrapper)
│   │   ├── reader.py              # load_agent_context_for_node()
│   │   └── writer.py              # write_agent_diary_entry()
│   └── tests/                     # pytest test suite (141 tests)
├── scripts/
│   ├── chat.py                    # CLI chat interface (primary user interface)
│   └── e2e_test.py                # End-to-end integration test
└── requirements.txt
```

---

## Core Data Flow

### 1. Topology Spec (the contract)

The master agent generates a `TopologySpec` JSON. This is the stable contract between intelligence and execution.

```python
TopologySpec
├── topology_id: str          # "topo-" + 8 hex chars
├── version: int              # 1
├── created_at: str           # ISO 8601
├── user_prompt: str          # original user request
├── nodes: list[NodeSpec]
│   ├── node_id: str          # unique, snake_case
│   ├── role: str             # human-readable
│   ├── model_tier: "fast" | "smart" | "powerful"
│   ├── system_prompt: str
│   ├── tools: [{"name": "WebSearch"}]
│   ├── inputs: [{"data_key": "...", "data_type": "...", "description": "..."}]
│   ├── outputs: [{"data_key": "...", ...}]  # must have ≥1
│   ├── instance_config: {"parallel_instances": 1, "consolidation": null}
│   ├── agent_mode: "react" | "single"
│   └── max_react_iterations: int
├── edges: [{"source_node_id", "target_node_id", "data_key"}]
├── constraints: {"max_cost_usd": 1.0, "max_latency_seconds": 120, ...}
└── operator_config: null  # or {"provider": "openrouter", "model_map": {...}}
```

### 2. Resolution

`resolve_topology(spec)` → `ResolvedTopology`

- `model_tier` → concrete model ID via MODEL_REGISTRY (or operator override)
- Tool names → concrete tool identifiers
- Computes `execution_order: list[list[str]]` via Kahn's topological sort
- Parallel groups: nodes in same list run concurrently

### 3. Execution

`execute_topology(resolved, user_prompt, tracker, on_node_start, on_node_complete)` → `ExecutionResult`

- Builds all nodes: `ResolvedNode → BuiltNode` (live `ReActAgent`)
- Walks `execution_order` step by step
- Each step: `run_execution_step()` → concurrent `asyncio.gather`
- Per node: fan-out parallel instances → consolidate → store output `Msg`
- Data flows via `data_store: dict[str, Msg]` (keyed by `data_key`)
- First node gets `initial_msg` (user prompt). Subsequent nodes get prior node outputs via edges.

---

## Key Classes & Functions

### `TopologySpec` (`daap/spec/schema.py`)
Pydantic model. The JSON contract. Validators: unique node_ids, edges reference existing nodes, react nodes must have tools, consolidation required when parallel_instances > 1.

### `ResolvedTopology` / `ResolvedNode` (`daap/spec/resolver.py`)
Dataclasses with concrete model IDs, tool identifiers, execution_order.

Model resolution chain: node.operator_override → topology.operator_config → MODEL_REGISTRY

### `BuiltNode` (`daap/executor/node_builder.py`)
Live AgentScope `ReActAgent` + metadata. Created by `build_node(resolved_node, tool_registry, tracker)`.

- `react` mode: full toolkit, max_iters from spec, `parallel_tool_calls=False` (Gemini strict)
- `single` mode: empty toolkit, max_iters=1 (one LLM call, no tool loop)

### `TrackedOpenAIChatModel` (`daap/executor/tracked_model.py`)
Subclass of `OpenAIChatModel`. Overrides `__call__` to capture `ChatUsage` and record to `TokenTracker`.

### `TokenTracker` (`daap/tools/token_tracker.py`)
Per-session accumulator. `add(model_id, input_tokens, output_tokens)`, `reset()`, `to_dict()`. Tracks `total_input`, `total_output`, `models_used`.

### `Session` (`daap/api/sessions.py`)
```python
Session:
  session_id, created_at, user_id
  master_agent           # live ReActAgent
  conversation           # list[{"role", "content"}]
  pending_topology       # raw dict after generate_topology
  pending_estimate       # cost/latency estimate dict
  is_executing           # bool — True during execute_topology
  execution_result       # dict after completion
  execution_progress     # dict with progress info (optional)
  token_tracker          # TokenTracker instance
  pending_questions      # list | None — set by ask_user tool
  _resolve_answers       # callable — WebSocket/CLI calls to deliver answers
```

### Master Agent Tools (`daap/master/tools.py` + `daap/api/sessions.py`)

| Tool | What it does |
|------|-------------|
| `generate_topology` | Parses JSON → validates → resolves → estimates → stores result |
| `ask_user` | Blocks agent, stores questions on session, waits for answers via asyncio.Event |
| `get_execution_status` | Reads `session.is_executing` + `session.execution_result` — agent calls this to check run status |

---

## Agent Modes

| Mode | When to use | AgentScope impl |
|------|-------------|----------------|
| `react` | Needs tools (search, fetch, iterate) | `ReActAgent`, iterative tool loop, max_iters |
| `single` | Pure LLM: write, format, summarize | `ReActAgent` with empty toolkit, max_iters=1 |

**Critical prompt rules:**
- `single` mode nodes: prompt must say "Output X now. Start immediately." — NOT "You will write..."
- `react` mode nodes: prompt must specify exact search strategy (Step 1: search X, Step 2: find Y) — NOT just the goal

---

## Available Tools for Subagents

| Abstract name | Implementation | Notes |
|--------------|----------------|-------|
| `WebSearch` | DuckDuckGo (`ddgs`) | No API key needed |
| `WebFetch` | httpx + BeautifulSoup | Strips HTML to text, 8000 char limit |
| `ReadFile` | Local file read | UTF-8 |
| `WriteFile` | Local file write | UTF-8 |
| `CodeExecution` | `subprocess`, Python only | 10s timeout |
| `mcp://linkedin` | MCP (Phase 2) | Not yet implemented |
| `mcp://crunchbase` | MCP (Phase 2) | Not yet implemented |

---

## Model Tiers → Concrete Models

All tiers currently map to same model (cost-effective for testing):

```python
MODEL_REGISTRY = {
    "fast":     "google/gemini-2.0-flash-001",
    "smart":    "google/gemini-2.0-flash-001",
    "powerful": "google/gemini-2.0-flash-001",
}
```

To use a different model, pass `operator_config` with a `model_map`.

---

## CLI Chat (`scripts/chat.py`)

Primary user interface. Commands:
- `approve` (+ variations: "run it", "go ahead", "yes execute", etc.) — execute pending topology
- `cheaper` — ask agent to reduce cost
- `cancel` — discard pending topology
- `quit` / `exit` — end session

On approve:
1. Validates + resolves topology
2. Fires `on_node_start` callback → prints `[1/2] Running lead_researcher (...)`
3. Fires `on_node_complete` callback → prints `[done] lead_researcher — 31.2s | 4821 tokens`
4. Injects execution result into `session.conversation` so agent can discuss it

---

## API Endpoints (FastAPI)

| Method | Path | What |
|--------|------|------|
| POST | `/sessions` | Create session → returns `session_id` |
| GET | `/sessions/{id}` | Get session state |
| DELETE | `/sessions/{id}` | Delete session |
| GET | `/sessions` | List all sessions |
| POST | `/sessions/{id}/message` | Send message (non-WebSocket) |
| GET | `/models` | List available model presets |
| WS | `/ws/{session_id}` | WebSocket conversation |

WebSocket message types sent by server: `response`, `plan`, `executing`, `result`, `error`, `ask_user`

---

## Execution Progress Callbacks

`execute_topology` signature:
```python
async def execute_topology(
    resolved: ResolvedTopology,
    user_prompt: str,
    tracker: TokenTracker | None = None,
    on_node_start: callable | None = None,   # (node_id, model_id, step_num, total_steps)
    on_node_complete: callable | None = None, # (NodeResult)
) -> ExecutionResult
```

`NodeResult` fields: `node_id`, `output_text` (truncated 500 chars), `latency_seconds`, `model_id`, `input_tokens`, `output_tokens`

`ExecutionResult` fields: `topology_id`, `final_output`, `node_results`, `total_latency_seconds`, `success`, `error`, `total_input_tokens`, `total_output_tokens`, `models_used`

---

## Feedback System

`FeedbackStore` (SQLite) stores per-run results + user ratings.
`collect_run_feedback(store, session_id, topology, result)` — called after execution.
`store.store_rating(session_id, rating=1-5, comment="...")` — user rates the run.

---

## Memory System (Phase 2 — optional)

`DaapMemory` wraps `mem0ai`. Used to enrich node system prompts with past learnings.
`load_agent_context_for_node(memory, agent_role, task_description)` → prepended to system_prompt.
If memory fails, execution continues (never breaks the pipeline).

---

## Known Constraints & Design Decisions

| Decision | Why |
|----------|-----|
| `parallel_tool_calls=False` | Gemini strict: tool call + result count must match exactly |
| All models → Gemini Flash | Cost-effective for dev/test; change via `MODEL_REGISTRY` or `operator_config` |
| OpenRouter as sole provider | Single API key, access to 100+ models |
| `sys.stdout` UTF-8 wrap on Windows | AgentScope prints `→` (U+2192); Windows cp1252 crashes without this |
| Session-scoped `ask_user` closure | Fixes concurrency bug: module-level state causes cross-session answer delivery |
| `agent_mode=single` uses max_iters=1 | ReActAgent with empty toolkit + 1 iteration = single LLM call |
| DAG only, no cycles | Validator enforces via topological sort (Kahn's algorithm) |
| First node gets user prompt | No incoming edges → receives `initial_msg` directly |

---

## Environment Setup

```bash
# .env file
OPENROUTER_API_KEY=sk-or-v1-...

# Install
pip install -r requirements.txt

# Run CLI
python scripts/chat.py

# Run API server
uvicorn daap.main:app --reload --port 8000

# Run E2E test
python scripts/e2e_test.py

# Run unit tests
pytest daap/tests/ -v
```

---

## What's Implemented (Phase 1 Complete)

- Master agent with `generate_topology`, `ask_user`, `get_execution_status` tools
- Topology spec schema + validation + resolution + estimation
- Execution engine: DAG walk, parallel steps, fan-out, consolidation
- Token tracking per session (input/output/models used)
- Real-time progress callbacks (`on_node_start`, `on_node_complete`)
- CLI chat with fuzzy approve detection + live progress display
- WebSocket API for browser clients
- Feedback store (SQLite)
- Memory system (mem0ai, optional enrichment)
- 141 unit tests

## What's Planned (Phase 2+)

- MCP tool integration (`mcp://linkedin`, `mcp://crunchbase`)
- RL-based instance count tuning
- Redis-backed sessions
- Per-user memory personalization
- Smarter retry with partial results + fallback nodes
