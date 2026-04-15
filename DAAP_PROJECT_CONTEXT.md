# DAAP — Project Context

**DAAP** (Dynamic Agent Automation Platform) is a B2B sales automation system where a **Master Agent** designs and executes multi-agent pipelines (called **topologies**) based on natural language user requests. User describes a sales task → master agent designs a DAG of specialized AI agents → user approves → engine executes.

---

## Quick Reference

| Thing | Value |
|-------|-------|
| Language | Python 3.11+ |
| Agent framework | AgentScope (`agentscope`) |
| LLM provider | OpenRouter (`https://openrouter.ai/api/v1`) |
| API key env var | `OPENROUTER_API_KEY` |
| Web server | FastAPI + uvicorn, port 8000 |
| Entrypoint (API) | `uvicorn daap.main:app --reload --port 8000` |
| Entrypoint (CLI) | `python scripts/chat.py` |
| E2E test | `python scripts/e2e_test.py` |
| Unit tests | `pytest daap/tests/` (121+ tests pass without agentscope) |
| Git remote | `https://github.com/warewe-raunit/DAAP.git` (branch: `master`) |

---

## Model Tiers (P0 — current)

```python
# daap/spec/resolver.py
MODEL_REGISTRY = {
    "fast":     "google/gemini-2.5-flash-lite",   # $0.10/$0.40 per 1M — search, extract, format
    "smart":    "deepseek/deepseek-v3.2",          # $0.26/$0.38 per 1M — evaluate, score, write
    "powerful": "google/gemini-2.5-flash",         # $0.30/$2.50 per 1M — master agent, planning
}
```

- Master agent uses `google/gemini-2.5-flash` (powerful tier).
- All model pricing in `MODEL_PRICING` dict in `resolver.py`.
- Helper: `get_model_pricing(model_id)` → returns pricing or `DEFAULT_PRICING`.
- Legacy `google/gemini-2.0-flash-001` in `MODEL_PRICING` for backward compat with saved topologies (deprecated June 1 2026 — do not add back to registry).

---

## Architecture Overview

```
User
 │
 ▼
Master Agent (ReActAgent — google/gemini-2.5-flash)
 │  session-scoped tools: generate_topology, ask_user, get_execution_status,
 │                         execute_pending_topology, load_topology,
 │                         persist_topology, rerun_topology
 │
 ├── ask_user → waits for user answers (asyncio.Event)
 ├── generate_topology → validates + estimates → stores as session.pending_topology
 ├── execute_pending_topology → agent calls this after user approves via ask_user
 │
 ▼
execute_pending_topology tool (daap/api/sessions.py)
 │  streams progress via session._ws_send (WebSocket) or None (CLI)
 │  writes run + learnings to DaapMemory post-execution
 │  auto-saves topology + run to TopologyStore
 │
 ▼
Execution Engine (execute_topology in daap/executor/engine.py)
 │  accepts daap_memory= — passes to build_node for prompt enrichment
 │
 ├── build_node() → enriches system_prompt with agent diary learnings
 │                → ResolvedNode → BuiltNode (live AgentScope ReActAgent)
 │
 └── Walk execution_order (topological sort, parallel groups)
      For each step: run_execution_step()
        └── run_parallel_instances() → consolidate_outputs() → Msg
 │
 ▼
ExecutionResult (final_output, node_results, tokens, latency)
 │
 └── Memory writes: write_run_to_memory + write_agent_learnings_from_run
```

---

## Directory Structure

```
DAAP/
├── daap/
│   ├── main.py                    # FastAPI app entrypoint
│   ├── api/
│   │   ├── routes.py              # REST endpoints + WebSocket mount + memory init
│   │   ├── ws_handler.py          # WebSocket conversation handler (no approve message type)
│   │   ├── sessions.py            # Session dataclass + SessionManager + session-scoped toolkit
│   │   └── topology_routes.py     # 11 REST endpoints for topology persistence
│   ├── master/
│   │   ├── agent.py               # Master ReActAgent factory (uses OPENROUTER_MASTER_MODEL)
│   │   ├── prompts.py             # Master agent system prompt (real tier costs injected)
│   │   └── tools.py               # generate_topology, ask_user tools + module-level state
│   ├── spec/
│   │   ├── schema.py              # Pydantic models: TopologySpec, NodeSpec, EdgeSpec, etc.
│   │   ├── resolver.py            # MODEL_REGISTRY, MODEL_PRICING, get_model_pricing(), topological sort
│   │   ├── validator.py           # Business rule validation (DAG check, tool availability, etc.)
│   │   └── estimator.py           # Cost + latency estimation (uses resolver.get_model_pricing)
│   ├── executor/
│   │   ├── engine.py              # execute_topology(resolved, user_prompt, tracker, callbacks, daap_memory)
│   │   ├── node_builder.py        # build_node(rnode, registry, daap_memory, tracker) — enriches prompts
│   │   ├── patterns.py            # run_parallel_instances, consolidate_outputs, run_execution_step
│   │   └── tracked_model.py       # TrackedOpenAIChatModel — records token usage per call
│   ├── tools/
│   │   ├── registry.py            # WebSearch, WebFetch, ReadFile, WriteFile, CodeExecution
│   │   └── token_tracker.py       # TokenTracker — per-session input/output accumulator
│   ├── feedback/
│   │   ├── store.py               # SQLite FeedbackStore
│   │   └── collector.py           # collect_run_feedback()
│   ├── memory/
│   │   ├── client.py              # DaapMemory (mem0ai wrapper) — user profile, run history, agent diary
│   │   ├── reader.py              # load_user_context_for_master(), load_agent_context_for_node()
│   │   ├── writer.py              # write_run_to_memory(), write_agent_learnings_from_run()
│   │   └── setup.py               # create_memory_client(mode) — production (Qdrant) or ephemeral
│   ├── topology/
│   │   ├── store.py               # TopologyStore — SQLite persistence for topologies + run history
│   │   ├── models.py              # StoredTopology, TopologyRun dataclasses
│   │   └── naming.py              # auto_name_from_prompt() — slug generator
│   └── tests/                     # pytest test suite
│       ├── test_model_tiers.py    # 14 tests — registry integrity, pricing, no deprecated models
│       ├── test_memory_wiring.py  # 20 tests — AST + functional tests for memory wiring
│       ├── test_topology_store.py # topology persistence tests
│       ├── test_topology_routes.py
│       ├── test_topology_naming.py
│       ├── test_topology_agent_tools.py
│       ├── test_resolver.py, test_schema.py, test_validator.py, test_estimator.py
│       └── test_memory.py, test_api.py, test_engine.py, ...
├── scripts/
│   ├── chat.py                    # CLI chat — inits DaapMemory, loads user context, passes to toolkit
│   └── e2e_test.py                # End-to-end integration test
├── docs/superpowers/
│   ├── specs/2026-04-14-topology-persistence-design.md
│   └── plans/2026-04-14-topology-persistence.md
├── README.md
├── DAAP_PROJECT_CONTEXT.md        # this file
├── DAAP_P0_Fix_Model_Tiers.md     # P0 implementation spec (done)
├── requirements.txt
├── daap_feedback.db               # SQLite feedback store
└── daap_topology.db               # SQLite topology store
```

---

## Session Object (`daap/api/sessions.py`)

```python
@dataclass
class Session:
    session_id: str
    created_at: float
    user_id: str = "default"

    master_agent: object | None = None          # live ReActAgent
    conversation: list[dict]                    # {"role", "content"} history

    pending_topology: dict | None = None        # raw TopologySpec dict after generate_topology
    pending_estimate: dict | None = None        # cost/latency estimate dict

    is_executing: bool = False
    execution_result: dict | None = None
    execution_progress: dict | None = None

    master_operator_config: dict | None = None
    subagent_operator_config: dict | None = None

    token_tracker: object | None = None         # TokenTracker
    pending_questions: list | None = None       # set by ask_user tool
    _resolve_answers: object | None = None      # callable — WebSocket/CLI delivers answers
    _ws_send: object | None = None              # async callable set by ws_handler for streaming
```

`create_session_scoped_toolkit(session, topology_store=None, daap_memory=None)` — registers all tools into the session closure.

---

## Session-Scoped Tools

| Tool | Registered when | What it does |
|------|----------------|-------------|
| `generate_topology` | always | Parses JSON → validates → resolves → estimates → stores pending |
| `ask_user` | always | Blocks agent, stores questions on session, waits via asyncio.Event |
| `get_execution_status` | always | Returns is_executing, progress, result |
| `execute_pending_topology` | always | Executes pending topology, streams via `_ws_send`, writes memory, auto-saves |
| `load_topology` | if topology_store | Loads saved topology into session.pending_topology |
| `persist_topology` | if topology_store | Saves pending_topology to TopologyStore |
| `rerun_topology` | if topology_store | Loads + immediately executes a saved topology |

---

## Approval Flow (IMPORTANT — no hardcoded keywords)

Old flow (removed): user typed "approve" → ws_handler detected keyword → ran `_execute_pending_topology`.

**Current flow:**
1. Agent calls `generate_topology` → presents plan
2. Agent calls `ask_user` with approval options: proceed / cheaper / cancel
3. If user approves → agent immediately calls `execute_pending_topology` tool
4. Tool streams progress via `session._ws_send`, auto-saves, writes memory

The `ws_handler.py` no longer has an `"approve"` message type. There is no `_execute_pending_topology` function in ws_handler. All execution is agent-driven.

Text shortcuts still work in ws_handler for convenience: "cheaper" → `_run_make_cheaper_flow`, "cancel" → clears pending.

---

## Topology Persistence (`daap/topology/`)

Full CRUD + versioning via SQLite (`daap_topology.db`).

### TopologyStore API
```python
store.save_topology(spec, user_id, overwrite=True)  → StoredTopology
store.get_topology(topology_id, version=None)        → StoredTopology | None
store.list_topologies(user_id, include_deleted=False) → list[StoredTopology]
store.get_versions(topology_id)                      → list[StoredTopology]
store.rename(topology_id, new_name)                  → None
store.set_max_runs(topology_id, max_runs)            → None
store.save_run(topology_id, version, user_id, result, user_prompt) → TopologyRun
store.get_runs(topology_id, limit=20)                → list[TopologyRun]
store.delete(topology_id)                            → None  (soft delete)
store.restore(topology_id)                           → None
store.purge_expired(ttl_days=30)                     → int   (count purged)
```

### REST Endpoints (prefix: `/api/v1/topologies`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | List user topologies |
| GET | `/{id}` | Get topology |
| GET | `/{id}/versions` | List versions (410 if all deleted) |
| POST | `/{id}/rerun` | Rerun topology (403 if wrong user) |
| PATCH | `/{id}/rename` | Rename |
| PATCH | `/{id}/max-runs` | Set run history cap |
| DELETE | `/{id}` | Soft delete |
| POST | `/{id}/restore` | Restore soft-deleted |
| GET | `/{id}/runs` | Run history |

Auto-save: `execute_pending_topology` tool auto-saves topology + run after execution.

---

## Memory System (P1 — active)

**Three memory scopes:**
- **User memory**: profile, preferences, run history (keyed by `user_id`)
- **Agent diary**: learnings per node role, shared across users (keyed by `agent_id = "daap_{role}"`)
- **Run memory**: per-execution summaries (keyed by `run_id`)

**Read path (pre-generation):**
- At session creation: `load_user_context_for_master(memory, user_id, prompt)` → injected into master agent system prompt
- At node build time: `load_agent_context_for_node(memory, role, task)` → prepended to node system_prompt

**Write path (post-execution):**
- `write_run_to_memory(memory, user_id, summary, result)` — called in `execute_pending_topology` after success
- `write_agent_learnings_from_run(memory, result, nodes)` — called in `execute_pending_topology` after success
- `write_user_feedback(memory, user_id, text)` — called in `POST /rate` endpoint

**Setup:**
```python
DaapMemory(mode="production")  # Qdrant-backed, needs QDRANT_HOST/PORT
DaapMemory(mode="ephemeral")   # in-memory Qdrant, no Docker needed (CLI default)
```
Memory always fails gracefully — never blocks execution.

**Extraction LLM:** `anthropic/claude-haiku-4-5-20251001` via OpenRouter.
**Embeddings:** `openai/text-embedding-3-small` via OpenRouter.

---

## Core Data Flow

### TopologySpec (the contract)
```python
TopologySpec
├── topology_id: "topo-" + 8 hex chars
├── version: int (1 on create)
├── user_prompt: str
├── nodes: list[NodeSpec]
│   ├── node_id, role, model_tier ("fast"|"smart"|"powerful")
│   ├── system_prompt, tools, inputs, outputs (≥1 required)
│   ├── instance_config: {parallel_instances, consolidation}
│   ├── agent_mode: "react" | "single"
│   └── max_react_iterations: int
├── edges: [{source_node_id, target_node_id, data_key}]
├── constraints: {max_cost_usd, max_latency_seconds, max_retries_per_node}
└── operator_config: null | {provider, base_url, api_key_env, model_map}
```

### Resolution chain
`resolve_topology(spec)` → `ResolvedTopology`
- model_tier → concrete model ID: `node.operator_override` → `topology.operator_config` → `MODEL_REGISTRY`
- tool names → concrete identifiers
- `execution_order: list[list[str]]` via Kahn's topological sort

### execute_topology signature
```python
async def execute_topology(
    resolved: ResolvedTopology,
    user_prompt: str,
    tracker: TokenTracker | None = None,
    on_node_start: callable | None = None,    # (node_id, model_id, step_num, total_steps)
    on_node_complete: callable | None = None, # (NodeResult)
    daap_memory=None,                         # DaapMemory | None — for node prompt enrichment
) -> ExecutionResult
```

---

## WebSocket Protocol (`/ws/{session_id}`)

**Client → Server:**
```json
{"type": "message",  "content": "..."}          // user chat message
{"type": "answer",   "answers": ["..."]}         // answers to ask_user questions
{"type": "make_cheaper"}                         // request cheaper topology
{"type": "cancel"}                               // cancel pending topology
```
Note: `{"type": "approve"}` no longer exists. Agent handles approval.

**Server → Client:**
```json
{"type": "response",  "content": "...", "usage": {...}}
{"type": "questions", "questions": [...]}
{"type": "plan",      "summary": "...", "cost_usd": ..., "latency_seconds": ...}
{"type": "executing", "topology_id": "...", "total_nodes": N}     // from execute_pending_topology tool
{"type": "progress",  "event": "node_start"|"node_complete", ...} // from execute_pending_topology tool
{"type": "result",    "output": "...", "latency_seconds": ...}    // from execute_pending_topology tool
{"type": "error",     "message": "..."}
```

---

## Available Tools for Subagents

| Abstract name | Notes |
|--------------|-------|
| `WebSearch` | DuckDuckGo (no API key) |
| `WebFetch` | httpx + BeautifulSoup, 8000 char limit |
| `ReadFile` | UTF-8 local read |
| `WriteFile` | UTF-8 local write |
| `CodeExecution` | Python only, 10s timeout |
| `mcp://linkedin` | Phase 2 — not implemented |
| `mcp://crunchbase` | Phase 2 — not implemented |

---

## REST API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/models` | Available model presets (dynamic from MODEL_REGISTRY) |
| POST | `/session` | Create session → `{session_id}` |
| GET | `/session/{id}/config` | Session model config |
| GET | `/sessions` | List all sessions |
| DELETE | `/session/{id}` | Delete session |
| GET | `/topology/{id}` | Pending topology for session |
| POST | `/rate` | Rate a completed run (1–5) |
| GET | `/runs/{id}` | Run history for session |
| WS | `/ws/{session_id}` | WebSocket conversation |
| GET/POST/etc | `/api/v1/topologies/...` | Topology persistence (11 endpoints) |

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| Agent-driven approval via `execute_pending_topology` tool | Removes hardcoded keyword detection; agent controls flow |
| `parallel_tool_calls=False` | Gemini strict: tool call + result count must match exactly |
| Session-scoped toolkit | Fixes concurrency bug: module-level ask_user caused cross-session answer delivery |
| `session._ws_send` callback | Decouples execution tool from WebSocket layer; CLI leaves it None |
| Topology persistence separate from FeedbackStore | Clean separation; schema ready for JWT auth (user_id is string) |
| Memory writes non-fatal | try/except everywhere — memory is optimization, never hard dependency |
| `agent_mode=single` uses max_iters=1 | ReActAgent with empty toolkit + 1 iteration = single LLM call |
| DAG only, no cycles | Validator enforces via Kahn's topological sort |
| OpenRouter as sole provider | Single API key, 100+ models; switch models without code changes |
| `sys.stdout` UTF-8 wrap on Windows | AgentScope prints `→` (U+2192); Windows cp1252 crashes without this |
| `writer.py` no module-level agentscope import | Lazy import of ExecutionResult keeps memory tests agentscope-free |

---

## Environment Setup

```bash
# .env
OPENROUTER_API_KEY=sk-or-v1-...
QDRANT_HOST=localhost          # optional, default localhost
QDRANT_PORT=6333               # optional, default 6333

# Install
pip install -r requirements.txt

# CLI
python scripts/chat.py

# API server
uvicorn daap.main:app --reload --port 8000

# Tests (no agentscope needed for core tests)
pytest daap/tests/test_model_tiers.py daap/tests/test_memory_wiring.py \
       daap/tests/test_resolver.py daap/tests/test_schema.py \
       daap/tests/test_validator.py daap/tests/test_estimator.py \
       daap/tests/test_topology_store.py daap/tests/test_topology_naming.py
```

---

## Phase Status

### Phase 1 — COMPLETE
- Master agent: `generate_topology`, `ask_user`, `get_execution_status`
- TopologySpec schema + validation + resolution + estimation
- Execution engine: DAG walk, parallel steps, fan-out, consolidation
- Token tracking per session
- Real-time progress callbacks
- CLI chat interface
- WebSocket API (FastAPI)
- SQLite feedback store

### Phase 2 — IN PROGRESS

| Task | Status | File(s) |
|------|--------|---------|
| **P0: Model tier migration** | ✅ DONE | `resolver.py`, `estimator.py`, `master/agent.py`, `master/prompts.py` |
| **P0: Remove approve keyword detection** | ✅ DONE | `ws_handler.py`, `sessions.py`, `scripts/chat.py`, `master/prompts.py` |
| **Topology persistence (SQLite)** | ✅ DONE | `topology/store.py`, `topology/models.py`, `topology/naming.py`, `api/topology_routes.py` |
| **P1: Per-user memory wiring** | ✅ DONE | `executor/engine.py`, `api/sessions.py`, `api/routes.py`, `scripts/chat.py`, `memory/writer.py` |
| MCP tools (LinkedIn, Crunchbase) | ⬜ PLANNED | Phase 2 |
| Redis session store | ⬜ PLANNED | Phase 2 |
| RL-based instance count tuning | ⬜ PLANNED | Phase 2 |
| Smarter retry + partial results | ⬜ PLANNED | Phase 3 |
| JWT auth | ⬜ PLANNED | Phase 3 |

### Test count
- 121 tests pass without agentscope installed (all spec/topology/memory tests)
- Full suite (requires agentscope): api, engine, patterns, node_builder, master_agent tests

---

## Files Added/Modified This Session (for diff reference)

**New files:**
- `daap/topology/store.py` — TopologyStore implementation
- `daap/topology/models.py` — StoredTopology, TopologyRun
- `daap/topology/naming.py` — auto_name_from_prompt()
- `daap/api/topology_routes.py` — 11 REST endpoints
- `daap/tests/test_topology_store.py`
- `daap/tests/test_topology_routes.py`
- `daap/tests/test_topology_naming.py`
- `daap/tests/test_topology_agent_tools.py`
- `daap/tests/test_model_tiers.py` — 14 tests for P0
- `daap/tests/test_memory_wiring.py` — 20 tests for P1
- `README.md`

**Modified:**
- `daap/spec/resolver.py` — new MODEL_REGISTRY, MODEL_PRICING, get_model_pricing()
- `daap/spec/estimator.py` — uses resolver pricing, per-model latency
- `daap/master/agent.py` — OPENROUTER_MASTER_MODEL = gemini-2.5-flash
- `daap/master/prompts.py` — real tier costs, agent-driven approval flow
- `daap/api/routes.py` — topology_store singleton, memory wiring, lifespan purge, updated /models
- `daap/api/sessions.py` — execute_pending_topology tool, logger, daap_memory param, memory writes
- `daap/api/ws_handler.py` — removed approve message type + dead _execute_pending_topology fn
- `daap/executor/engine.py` — daap_memory param → passed to build_node
- `daap/memory/writer.py` — lazy ExecutionResult import (no agentscope at module level)
- `scripts/chat.py` — DaapMemory init, user context, memory passed to toolkit
