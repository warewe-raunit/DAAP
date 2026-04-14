# Topology Persistence — Design Spec
**Date:** 2026-04-14
**Status:** Approved
**Scope:** Save, list, version, edit, rerun, and delete topologies per user

---

## Problem

After a topology executes, `session.pending_topology` is nulled out. The topology vanishes. Users cannot:
- Retrieve a past topology by ID
- Rerun a topology without regenerating it from scratch
- Edit an existing topology
- Browse their topology library

This spec defines a persistent topology store that survives server restarts, scopes topologies per user, and integrates with both the agent chat interface and the REST API.

---

## Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| `user_id` model | Simple string now, schema ready for real auth | Phase 1 simplicity, zero breaking change when auth lands |
| Edit save mode | User chooses: overwrite or new version | Flexibility without forcing complexity |
| Edit mechanism | Agent-assisted (chat) + raw API patch | Maximum flexibility |
| Rerun prompt | Master agent infers from user intent | Natural UX — agent reads "again" vs "again but for X" |
| Naming | Auto-slug from prompt + user can rename | Readable by default, customisable |
| Run history | Capped N per topology (user-configurable, default 10) | Bounded storage + RL training data |
| Delete | Soft delete with TTL (default 30 days), auto-purge on startup | Recoverable + storage-bounded |
| Storage approach | Separate `TopologyStore` with own SQLite file | Single responsibility, mirrors `FeedbackStore` pattern |

---

## Architecture

### New package: `daap/topology/`

```
daap/
├── topology/
│   ├── __init__.py
│   ├── store.py          # TopologyStore — all SQLite CRUD
│   ├── models.py         # StoredTopology, TopologyRun dataclasses
│   └── naming.py         # auto_name_from_prompt() slug generator
```

Mirrors `daap/feedback/` exactly. `FeedbackStore` stays ratings/RL-only — no concerns mixed.

### New API module: `daap/api/topology_routes.py`

Mounted on FastAPI `app` in `main.py`. Follows same pattern as `routes.py`.

---

## Data Model

### Database: `daap_topology.db`

```sql
CREATE TABLE topologies (
    topology_id     TEXT    NOT NULL,
    version         INTEGER NOT NULL DEFAULT 1,
    user_id         TEXT    NOT NULL DEFAULT 'default',
    name            TEXT    NOT NULL,
    spec_json       TEXT    NOT NULL,
    created_at      REAL    NOT NULL,
    updated_at      REAL    NOT NULL,
    deleted_at      REAL,                  -- NULL = active
    delete_ttl_days INTEGER DEFAULT 30,
    max_runs        INTEGER DEFAULT 10,    -- per-topology run history cap
    PRIMARY KEY (topology_id, version)
);

-- list_topologies returns one row per topology_id (MAX(version) = latest).
-- get_runs returns runs across ALL versions of a topology_id.

CREATE TABLE topology_runs (
    run_id           TEXT    PRIMARY KEY,  -- uuid4
    topology_id      TEXT    NOT NULL,
    topology_version INTEGER NOT NULL,     -- which version was running
    user_id          TEXT    NOT NULL,
    ran_at           REAL    NOT NULL,
    user_prompt      TEXT,                 -- prompt used for this run
    result_json      TEXT,
    success          INTEGER,
    latency_seconds  REAL,
    input_tokens     INTEGER,
    output_tokens    INTEGER
);

CREATE INDEX idx_topologies_user  ON topologies(user_id, deleted_at);
CREATE INDEX idx_runs_topology    ON topology_runs(topology_id, ran_at DESC);
```

**`(topology_id, version)` PK** supports both overwrite (version=1, row replaced) and version history (new row, version incremented).

### Python models (`daap/topology/models.py`)

```python
@dataclass
class StoredTopology:
    topology_id: str
    version: int
    user_id: str
    name: str
    spec: dict
    created_at: float
    updated_at: float
    deleted_at: float | None

@dataclass
class TopologyRun:
    run_id: str
    topology_id: str
    topology_version: int
    user_id: str
    ran_at: float
    user_prompt: str | None
    result: dict | None
    success: bool
    latency_seconds: float
    input_tokens: int
    output_tokens: int
```

---

## `TopologyStore` Interface (`daap/topology/store.py`)

```python
class TopologyStore:
    def __init__(self, db_path: str = "daap_topology.db"): ...

    # ── Write ──────────────────────────────────────────────────────────
    def save_topology(
        self,
        spec: dict,
        user_id: str,
        name: str | None = None,       # None → auto_name_from_prompt()
        overwrite: bool = True,        # False → increment version
    ) -> StoredTopology: ...

    def rename_topology(self, topology_id: str, new_name: str) -> None: ...

    def save_run(
        self,
        topology_id: str,
        topology_version: int,
        user_id: str,
        result: dict,
        user_prompt: str | None = None,
    ) -> TopologyRun: ...
    # auto-enforces max_runs cap — deletes oldest run when over limit

    def set_max_runs(self, topology_id: str, max_runs: int) -> None: ...

    def delete_topology(self, topology_id: str, ttl_days: int = 30) -> None: ...
    def restore_topology(self, topology_id: str) -> None: ...
    def purge_expired(self) -> int: ...  # returns count of hard-deleted rows

    # ── Read ───────────────────────────────────────────────────────────
    def get_topology(
        self,
        topology_id: str,
        version: int | None = None,    # None → latest version
    ) -> StoredTopology | None: ...

    def list_topologies(
        self,
        user_id: str,
        include_deleted: bool = False,
    ) -> list[StoredTopology]: ...

    def get_runs(
        self,
        topology_id: str,
        limit: int | None = None,
    ) -> list[TopologyRun]: ...
```

---

## REST API (`daap/api/topology_routes.py`)

`user_id` sourced from query param `?user_id=xyz` now. Migrates to `X-User-ID` header when auth lands — zero breaking change.

| Method | Path | Action |
|--------|------|--------|
| `GET` | `/topologies` | List user's topologies |
| `GET` | `/topologies/{id}` | Get latest version |
| `GET` | `/topologies/{id}/versions` | List all versions |
| `GET` | `/topologies/{id}/v/{ver}` | Get specific version |
| `GET` | `/topologies/{id}/runs` | Get run history |
| `PATCH` | `/topologies/{id}` | Direct JSON patch |
| `PATCH` | `/topologies/{id}/rename` | Rename |
| `PATCH` | `/topologies/{id}/max-runs` | Set run cap N |
| `POST` | `/topologies/{id}/rerun` | Rerun (optional new prompt) |
| `DELETE` | `/topologies/{id}` | Soft delete |
| `POST` | `/topologies/{id}/restore` | Un-delete |

### Request models

```python
class TopologyPatchRequest(BaseModel):
    spec: dict
    save_mode: Literal["overwrite", "new_version"]

class RerunRequest(BaseModel):
    user_prompt: str | None = None   # None = use spec.user_prompt
    session_id: str                  # active session required

class RenameRequest(BaseModel):
    name: str

class MaxRunsRequest(BaseModel):
    max_runs: int                    # validated: 1–100
```

---

## Agent-Assisted Flows

### New master agent tools (registered in `create_session_scoped_toolkit`)

```python
async def load_topology(topology_id: str, version: int | None) -> ToolResponse:
    """Load saved topology into session.pending_topology for editing or rerun."""

async def save_topology(topology_id: str, save_mode: str) -> ToolResponse:
    """Persist session.pending_topology to TopologyStore."""

async def rerun_topology(topology_id: str, user_prompt: str | None) -> ToolResponse:
    """Load + execute saved topology. user_prompt=None uses original."""
```

### Edit flow
```
user: "change lead_researcher to also use WebFetch"
→ agent: load_topology(id)                  # loads into pending_topology
→ agent: generate_topology(patched_json)    # validates patched spec
→ agent: asks "overwrite v1 or save as v2?"
→ user picks
→ agent: save_topology(id, save_mode)
```

### Rerun flow
```
# Same prompt
user: "run the fintech leads topology again"
→ agent: rerun_topology("topo-a3f9c21b", None)

# New prompt
user: "run the fintech leads topology again but for healthtech"
→ agent: rerun_topology("topo-a3f9c21b", "find B2B leads in healthtech")
```

---

## Integration Points

### Auto-save after execution
Both `ws_handler.py` and `scripts/chat.py` call `topology_store.save_topology()` after every successful execution. User never needs to manually save.

### Auto-save run history
Both handlers call `topology_store.save_run()` after every execution (success or failure). RL training data accumulates automatically.

### Purge on startup
FastAPI `lifespan` event calls `topology_store.purge_expired()` once at server start. No scheduler or cron needed.

### `topology_store` singleton
Instantiated in `routes.py` alongside `feedback_store`. Passed to `topology_routes` router and to `ws_handler` via dependency injection.

---

## Error Handling

| Scenario | Response |
|----------|----------|
| `topology_id` not found | `404` |
| Patch spec fails Pydantic validation | `422` with field errors |
| Rerun on soft-deleted topology | `410 Gone` |
| Invalid `save_mode` | `400` |
| `max_runs` out of range (< 1 or > 100) | `400` |
| TopologyStore DB error | Log + `500`, never crash session |
| Auto-save fails post-execution | Log warning only — run result still returned |
| Agent `load_topology` fails | Error `ToolResponse` — agent tells user gracefully |

---

## Testing

New test files in `daap/tests/`:

```
test_topology_store.py       # unit: all store methods, versioning, run cap, purge
test_topology_routes.py      # integration: all endpoints via FastAPI TestClient
test_topology_naming.py      # unit: auto_name_from_prompt() edge cases
test_topology_agent_tools.py # unit: load/save/rerun tool response correctness
```

### Key test cases
- Save → get → spec matches
- Overwrite: version stays 1, `updated_at` changes
- New version: version increments, old version still retrievable via `/v/{ver}`
- Run cap: 11th run deletes oldest, count stays at 10
- Soft delete: hidden from `list_topologies`, restore brings back
- `purge_expired()`: only hard-deletes past TTL, leaves active + recent deleted
- Rerun `user_prompt=None` uses `spec.user_prompt`
- Auto-save fires after execution — topology appears in list
- `user_id` scoping: user A cannot see user B topologies

---

## Future Compatibility

| Current | Future (zero breaking change) |
|---------|-------------------------------|
| `user_id` query param | `X-User-ID` header from JWT |
| SQLite `daap_topology.db` | Postgres (swap `store.py` internals only) |
| `max_runs=10` hardcoded default | Per-user preference from user profile table |
| Manual `purge_expired()` on startup | Background task / cron job |
| Run history for RL | Direct input to Phase 2 RL optimizer |
