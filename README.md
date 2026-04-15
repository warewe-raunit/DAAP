# DAAP — Dynamic Agent Automation Platform

B2B sales automation platform powered by multi-agent DAG pipelines. Describe a task, the master agent designs a workflow, you approve, it executes.

---

## How it works

```
User → Master Agent (ReActAgent)
         ├── ask_user          clarify intent
         ├── generate_topology design multi-agent DAG
         └── execute_pending_topology  run on approval

Execution Engine
         ├── resolve + validate TopologySpec
         ├── build sub-agents per node (react / single mode)
         └── walk topological order (parallel groups)
              → final output + token usage + latency
```

---

## Features

- **Conversational planning** — master agent asks targeted questions, designs topology, presents cost/latency estimate
- **Agent-driven approval** — agent calls `ask_user`, user approves, agent executes directly (no magic keywords)
- **DAG execution engine** — parallel node groups, topological sort, per-node token tracking
- **Agent modes** — `react` (iterative tool loop) or `single` (one LLM call)
- **Tool registry** — WebSearch, WebFetch, ReadFile, WriteFile, CodeExecution; MCP tool format ready (`mcp://service`)
- **Topology persistence** — save, version, rename, rerun, soft-delete topologies per user (SQLite)
- **Run history** — per-topology run log, capped at `max_runs`
- **REST API** — 11 endpoints under `/api/v1/topologies/`
- **WebSocket streaming** — real-time `node_start` / `node_complete` / `result` events
- **Feedback store** — SQLite run outcome logging
- **Per-user memory** — mem0ai: loads user profile + preferences at session start, enriches node prompts with agent learnings, writes run outcomes post-execution (optional credentials, fails gracefully)

---

## Stack

| Layer | Technology |
|---|---|
| Agent framework | AgentScope (`ReActAgent`, `Toolkit`, `InMemoryMemory`) |
| LLM routing | OpenRouter — 3 tiers: Gemini 2.5 Flash / DeepSeek V3.2 / Gemini 2.5 Flash Lite |
| API | FastAPI + uvicorn (port 8000) |
| Persistence | SQLite (topology store + feedback store) |
| Memory | mem0ai — per-user context + agent learnings (active, optional credentials) |
| Python | 3.11+ |

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENROUTER_API_KEY=your_key_here

# CLI chat
python scripts/chat.py

# REST + WebSocket API
uvicorn daap.main:app --reload --port 8000
```

---

## Project structure

```
daap/
├── master/        master agent, tools, system prompt
├── spec/          TopologySpec schema, resolver, validator, estimator
├── executor/      execution engine, node builder, parallel patterns
├── topology/      persistence store, models, naming
├── tools/         WebSearch, WebFetch, ReadFile, WriteFile, CodeExecution
├── feedback/      SQLite feedback store
├── memory/        mem0ai wrapper
├── api/           FastAPI routes, WebSocket handler, sessions
└── tests/         unit + integration tests

scripts/
├── chat.py        CLI interface
└── e2e_test.py    end-to-end test runner
```

---

## REST API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/topologies/` | List user topologies |
| GET | `/api/v1/topologies/{id}` | Get topology |
| GET | `/api/v1/topologies/{id}/versions` | List versions |
| POST | `/api/v1/topologies/{id}/rerun` | Rerun topology |
| PATCH | `/api/v1/topologies/{id}/rename` | Rename |
| PATCH | `/api/v1/topologies/{id}/max-runs` | Set run cap |
| DELETE | `/api/v1/topologies/{id}` | Soft delete |
| POST | `/api/v1/topologies/{id}/restore` | Restore deleted |
| GET | `/api/v1/topologies/{id}/runs` | Run history |
| WebSocket | `/ws/{session_id}` | Live agent session |

---

## Model tiers

| Tier | Model | Cost (in/out per 1M) | Use case |
|---|---|---|---|
| `fast` | `google/gemini-2.5-flash-lite` | $0.10 / $0.40 | search, extract, format |
| `smart` | `deepseek/deepseek-v3.2` | $0.26 / $0.38 | evaluate, score, write |
| `powerful` | `google/gemini-2.5-flash` | $0.30 / $2.50 | complex planning, master agent |

Override any tier per session via `POST /session?subagent_fast_model=...`

---

## Running tests

```bash
pytest daap/tests/
```

---

## Phase 2 status

| Item | Status |
|---|---|
| Model tier migration (Gemini 2.0 → 2.5 Flash / DeepSeek / Flash Lite) | Done |
| Agent-driven approval (removed keyword detection) | Done |
| Per-user memory — context load + node enrichment + post-run writes | Done |
| Topology persistence — save, version, rerun, soft-delete | Done |

## Roadmap (Phase 3)

- MCP tools: LinkedIn, Crunchbase
- Redis session store
- RL-based instance tuning
- JWT auth
- Smarter retry + partial results
