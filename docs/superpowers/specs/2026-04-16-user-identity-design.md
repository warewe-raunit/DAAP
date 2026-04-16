# User Identity & RL Visibility — Design Spec
**Date:** 2026-04-16
**Status:** Approved

## Problem

`user_id` exists in the data model but is not trustworthy as a per-user identifier:

- CLI always uses `"default"` — all users share one memory, one RL state, one topology history
- API `user_id` optional, defaults to `"default"` — callers can omit it silently
- `rerun_topology()` never passes `user_id` to `execute_topology()` — diary context lost on API reruns
- Dead fallbacks: `session.user_id or "anonymous"` in two places (user_id never falsy)
- Fragile memory gate: `if session.user_id:` — silent skip if empty

Result: RL personalization (the main USP) is invisible and untested per-user.

## Goal

Proper user distinction so RL customization accumulates per person and is visible to them.

---

## Section 1: Local Identity Persistence

**New module:** `daap/identity.py`
**Storage:** `~/.daap/user.json`

```json
{ "user_id": "alice" }
```

**API:**

```python
load_local_user() -> str | None      # read file, return user_id or None
save_local_user(user_id: str) -> None # write file, create ~/.daap/ if needed
resolve_cli_user() -> str             # load → if None, prompt + save → return
```

**Sanitization:** strip whitespace, lowercase, spaces→`-`, reject empty. Keeps user_id safe as DB/memory key.

**Migration:** existing `"default"` rows in SQLite untouched. New CLI sessions use real user_id forward.

---

## Section 2: CLI Startup Flow

In `scripts/chat.py`, before `session_mgr.create_session()`:

**First run** (no `~/.daap/user.json`):
```
What's your name? › alice
Identity saved. Welcome, Alice!
```

**Return visit:**
```
Welcome back, Alice — 12 runs · optimizer active · 3 memory facts
```

Welcome line assembled from three best-effort calls (any failure → segment silently omitted):
- `topology_store.count_runs(user_id)` → run count
- `optimizer/store.load_optimizer(user_id)` → trained arms present → "optimizer active" else "optimizer learning"
- `memory/reader.load_user_profile(user_id)` → profile fact count

Resolved `user_id` passed to `session_mgr.create_session(user_id=user_id)`.

---

## Section 3: Welcome Line + `/profile` Command

**`/profile` output:**
```
User:       alice
Runs:       12
Optimizer:
  master      → google/gemini-2.0-flash-001 (8 obs)
  researcher  → google/gemini-pro (3 obs)
Memory:
  - Prefers concise outputs
  - Works in B2B SaaS
  - [+ 1 more]
```

**Data sources:**
- `optimizer/store.load_optimizer(user_id)` — arm states per role
- `memory/reader.load_user_profile(user_id)` — profile facts

All best-effort — partial data shown, no crash if source unavailable.

**Help text updated:** `/help` and header line include `/profile` alongside `/skills` and `/mcp`.

---

## Section 4: API `user_id` Required

```python
# Before
user_id: str = "default"

# After
user_id: str
```

FastAPI returns 422 automatically if omitted. No extra validation code.

---

## Section 5: Bug Fixes

| File | Location | Bug | Fix |
|------|----------|-----|-----|
| `daap/api/topology_routes.py` | `rerun_topology()` | `execute_topology()` called without `user_id` | Add `user_id=session.user_id` |
| `daap/api/sessions.py:518` | RL tier recommendations | `session.user_id or "anonymous"` dead code | → `session.user_id` |
| `daap/api/routes.py:356` | RL record outcome | `session.user_id or "anonymous"` dead code | → `session.user_id` |
| `daap/api/sessions.py:597,648` | Memory write gate | `if session.user_id:` silently skips if empty | → `if daap_memory is not None:` |

---

## Files Changed

| File | Change |
|------|--------|
| `daap/identity.py` | **New** — `load_local_user`, `save_local_user`, `resolve_cli_user` |
| `scripts/chat.py` | First-run prompt, welcome line, `/profile` cmd, pass `user_id` to session |
| `daap/api/sessions.py` | `create_session(user_id)` accepts param; fix memory gate; fix dead fallback |
| `daap/api/routes.py` | `user_id: str` required; fix dead fallback |
| `daap/api/topology_routes.py` | Pass `user_id` to `execute_topology()` in rerun path |

## Out of Scope

- Auth/JWT — user_id remains app-level identifier, no server verification
- Multi-profile switching in CLI
- API key / PIN protection
