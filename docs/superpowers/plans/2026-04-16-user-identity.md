# User Identity & RL Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up per-user identity so RL personalization accumulates per person, is visible to them, and the CLI stops sharing a single "default" user.

**Architecture:** New `daap/identity.py` module handles local persistence (`~/.daap/user.json`). CLI resolves user at startup (first-run prompt, then greet on return). API makes `user_id` required. Three targeted bug fixes (rerun missing user_id, dead fallbacks, fragile memory gate). `/profile` CLI command surfaces RL + memory state.

**Tech Stack:** Python stdlib (`pathlib`, `json`), SQLite (already used), existing `BanditStore`, `TopologyStore`, `memory/reader.py`.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `daap/identity.py` | `load_local_user`, `save_local_user`, `resolve_cli_user` |
| Create | `daap/tests/test_identity.py` | identity module tests |
| Modify | `daap/topology/store.py` | add `count_runs(user_id)` |
| Modify | `daap/optimizer/store.py` | add `get_profile_summary(user_id)` |
| Modify | `daap/api/sessions.py` | `create_session(user_id)`, fix memory gate, fix rerun |
| Modify | `daap/api/routes.py` | `user_id: str` required, fix dead fallback |
| Modify | `daap/tests/test_api.py` | update tests that POST /session without user_id |
| Modify | `scripts/chat.py` | identity resolve, welcome line, `/profile` cmd |

---

## Task 1: `daap/identity.py` — local identity persistence

**Files:**
- Create: `daap/identity.py`
- Create: `daap/tests/test_identity.py`

- [ ] **Step 1: Write failing tests**

```python
# daap/tests/test_identity.py
import json
import pytest


def test_load_local_user_returns_none_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from daap import identity
    monkeypatch.setattr(identity, "_daap_dir", lambda: tmp_path / ".daap")
    assert identity.load_local_user() is None


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    from daap import identity
    monkeypatch.setattr(identity, "_daap_dir", lambda: tmp_path / ".daap")
    identity.save_local_user("alice")
    assert identity.load_local_user() == "alice"


def test_save_creates_directory(tmp_path, monkeypatch):
    from daap import identity
    daap_dir = tmp_path / ".daap"
    monkeypatch.setattr(identity, "_daap_dir", lambda: daap_dir)
    assert not daap_dir.exists()
    identity.save_local_user("bob")
    assert daap_dir.exists()
    assert (daap_dir / "user.json").exists()


def test_load_returns_none_on_corrupt_file(tmp_path, monkeypatch):
    from daap import identity
    daap_dir = tmp_path / ".daap"
    daap_dir.mkdir()
    (daap_dir / "user.json").write_text("NOT JSON")
    monkeypatch.setattr(identity, "_daap_dir", lambda: daap_dir)
    assert identity.load_local_user() is None


def test_sanitize_strips_and_lowercases():
    from daap.identity import _sanitize
    assert _sanitize("  Alice  ") == "alice"
    assert _sanitize("John Doe") == "john-doe"
    assert _sanitize("BOB123") == "bob123"


def test_sanitize_rejects_empty():
    from daap.identity import _sanitize
    assert _sanitize("   ") is None
    assert _sanitize("") is None
```

- [ ] **Step 2: Run to verify failure**

```
pytest daap/tests/test_identity.py -v
```
Expected: `ModuleNotFoundError` or `ImportError` — module doesn't exist yet.

- [ ] **Step 3: Create `daap/identity.py`**

```python
"""
DAAP local user identity — persist user_id to ~/.daap/user.json.

First run: prompt for name, save.
Return visits: load and greet.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def _daap_dir() -> Path:
    """Return ~/.daap directory path (overridable in tests)."""
    return Path.home() / ".daap"


def _sanitize(raw: str) -> str | None:
    """Lowercase, replace spaces with hyphens, strip non-alphanumeric-hyphen chars."""
    stripped = raw.strip()
    if not stripped:
        return None
    lowered = stripped.lower()
    hyphenated = re.sub(r"\s+", "-", lowered)
    cleaned = re.sub(r"[^a-z0-9\-]", "", hyphenated)
    return cleaned or None


def load_local_user() -> str | None:
    """Read ~/.daap/user.json and return user_id, or None if missing/corrupt."""
    path = _daap_dir() / "user.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        user_id = data.get("user_id", "")
        return user_id if user_id else None
    except Exception:
        return None


def save_local_user(user_id: str) -> None:
    """Write user_id to ~/.daap/user.json, creating the directory if needed."""
    d = _daap_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "user.json").write_text(
        json.dumps({"user_id": user_id}, indent=2),
        encoding="utf-8",
    )


def resolve_cli_user() -> str:
    """
    Resolve user identity for CLI sessions.

    - If ~/.daap/user.json exists: return saved user_id (caller shows welcome-back line).
    - Otherwise: prompt for name, sanitize, save, return.
    """
    saved = load_local_user()
    if saved:
        return saved

    while True:
        try:
            raw = input("What's your name? › ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = "user"

        user_id = _sanitize(raw)
        if user_id:
            save_local_user(user_id)
            print(f"Identity saved. Welcome, {raw.strip().title()}!")
            return user_id

        print("Name can't be empty. Try again.")
```

- [ ] **Step 4: Run tests**

```
pytest daap/tests/test_identity.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add daap/identity.py daap/tests/test_identity.py
git commit -m "feat(identity): local user persistence via ~/.daap/user.json"
```

---

## Task 2: `TopologyStore.count_runs(user_id)`

**Files:**
- Modify: `daap/topology/store.py`
- Modify: `daap/tests/test_api.py` (add test at end)

- [ ] **Step 1: Write failing test**

Add to `daap/tests/test_api.py`:

```python
def test_topology_store_count_runs(tmp_path):
    """count_runs returns 0 for unknown user, correct count after saves."""
    from daap.topology.store import TopologyStore

    store = TopologyStore(db_path=str(tmp_path / "topo.db"))
    assert store.count_runs("alice") == 0

    spec = {
        "topology_id": "t1",
        "name": "Test",
        "nodes": [],
        "execution_order": [],
        "user_prompt": "hi",
    }
    store.save_topology(spec=spec, user_id="alice")
    store.save_run(
        topology_id="t1",
        topology_version=1,
        user_id="alice",
        result={"success": True, "latency_seconds": 1.0},
    )
    store.save_run(
        topology_id="t1",
        topology_version=1,
        user_id="alice",
        result={"success": False, "latency_seconds": 0.5},
    )
    assert store.count_runs("alice") == 2
    assert store.count_runs("bob") == 0
```

- [ ] **Step 2: Run to verify failure**

```
pytest daap/tests/test_api.py::test_topology_store_count_runs -v
```
Expected: `AttributeError: 'TopologyStore' object has no attribute 'count_runs'`

- [ ] **Step 3: Add `count_runs` to `TopologyStore`**

Find the `list_topologies` method in `daap/topology/store.py` and add `count_runs` just before it:

```python
def count_runs(self, user_id: str) -> int:
    """Count total execution runs for a user."""
    with sqlite3.connect(self.db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM topology_runs WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return row[0] if row else 0
```

- [ ] **Step 4: Run test**

```
pytest daap/tests/test_api.py::test_topology_store_count_runs -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add daap/topology/store.py daap/tests/test_api.py
git commit -m "feat(topology): add count_runs(user_id) to TopologyStore"
```

---

## Task 3: `BanditStore.get_profile_summary(user_id)`

**Files:**
- Modify: `daap/optimizer/store.py`
- Modify: `daap/tests/test_optimizer.py`

- [ ] **Step 1: Write failing test**

Check what's in `daap/tests/test_optimizer.py` first. Add this test:

```python
def test_bandit_store_get_profile_summary_empty(tmp_path):
    """get_profile_summary returns [] for user with no bandit state."""
    from daap.optimizer.store import BanditStore
    store = BanditStore(db_path=str(tmp_path / "opt.db"))
    assert store.get_profile_summary("alice") == []


def test_bandit_store_get_profile_summary_returns_best_arm_per_role(tmp_path):
    """get_profile_summary returns one entry per role with highest-pull arm."""
    import numpy as np
    from daap.optimizer.store import BanditStore

    store = BanditStore(db_path=str(tmp_path / "opt.db"))
    dim = 4
    B = np.eye(dim)
    f = np.zeros(dim)

    store.save_arm_state("alice", "master", "fast", dim, B, f, n_pulls=2)
    store.save_arm_state("alice", "master", "smart", dim, B, f, n_pulls=8)
    store.save_arm_state("alice", "researcher", "fast", dim, B, f, n_pulls=3)

    summary = store.get_profile_summary("alice")
    roles = {s["role"]: s for s in summary}

    assert "master" in roles
    assert roles["master"]["best_arm"] == "smart"
    assert roles["master"]["n_pulls"] == 8

    assert "researcher" in roles
    assert roles["researcher"]["best_arm"] == "fast"
    assert roles["researcher"]["n_pulls"] == 3
```

- [ ] **Step 2: Run to verify failure**

```
pytest daap/tests/test_optimizer.py -k "profile_summary" -v
```
Expected: `AttributeError: 'BanditStore' object has no attribute 'get_profile_summary'`

- [ ] **Step 3: Add `get_profile_summary` to `BanditStore`**

Add after `get_user_run_count` in `daap/optimizer/store.py`:

```python
def get_profile_summary(self, user_id: str) -> list[dict]:
    """
    Return per-role optimizer summary for display in /profile.

    Returns one entry per role: the arm with the most pulls.
    Does not load numpy arrays — pure SQL, no heavy deps.
    """
    with self._connect() as conn:
        rows = conn.execute(
            """
            SELECT role, arm, n_pulls
            FROM bandit_state
            WHERE user_id = ?
            ORDER BY role ASC, n_pulls DESC
            """,
            (user_id,),
        ).fetchall()

    seen: set[str] = set()
    result: list[dict] = []
    for role, arm, n_pulls in rows:
        if role not in seen:
            seen.add(role)
            result.append({"role": role, "best_arm": arm, "n_pulls": n_pulls})
    return result
```

- [ ] **Step 4: Run tests**

```
pytest daap/tests/test_optimizer.py -k "profile_summary" -v
```
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add daap/optimizer/store.py daap/tests/test_optimizer.py
git commit -m "feat(optimizer): add get_profile_summary(user_id) to BanditStore"
```

---

## Task 4: `daap/api/sessions.py` — three targeted fixes

**Files:**
- Modify: `daap/api/sessions.py`
- Modify: `daap/tests/test_api.py`

Three fixes in one task (all in sessions.py, all small):

1. `SessionManager.create_session()` accepts `user_id` param
2. Memory gate: `if daap_memory is not None and session.user_id:` → `if daap_memory is not None:`
3. Rerun: `execute_topology(...)` missing `daap_memory=daap_memory, user_id=session.user_id`
4. Dead fallback: `session.user_id or "anonymous"` → `session.user_id`

- [ ] **Step 1: Write failing test for `create_session(user_id)`**

Add to `daap/tests/test_api.py`:

```python
def test_session_manager_create_session_accepts_user_id():
    """SessionManager.create_session accepts user_id param and stores it."""
    from daap.api.sessions import SessionManager
    mgr = SessionManager()
    session = mgr.create_session(user_id="alice")
    assert session.user_id == "alice"


def test_session_manager_create_session_default_user_id():
    """SessionManager.create_session uses 'default' when user_id omitted."""
    from daap.api.sessions import SessionManager
    mgr = SessionManager()
    session = mgr.create_session()
    assert session.user_id == "default"
```

- [ ] **Step 2: Run to verify failure**

```
pytest daap/tests/test_api.py::test_session_manager_create_session_accepts_user_id -v
```
Expected: `TypeError: create_session() got an unexpected keyword argument 'user_id'`

- [ ] **Step 3: Apply all four fixes to `daap/api/sessions.py`**

**Fix 1** — `create_session` signature (line ~78):

```python
# BEFORE
def create_session(self) -> Session:
    session_id = str(uuid.uuid4())[:8]
    session = Session(session_id=session_id, created_at=time.time())
    self._sessions[session_id] = session
    return session

# AFTER
def create_session(self, user_id: str = "default") -> Session:
    session_id = str(uuid.uuid4())[:8]
    session = Session(session_id=session_id, created_at=time.time(), user_id=user_id)
    self._sessions[session_id] = session
    return session
```

**Fix 2** — memory gate at line ~597 (first occurrence in `_on_node_complete`):

```python
# BEFORE
if daap_memory is not None and session.user_id:

# AFTER
if daap_memory is not None:
```

**Fix 3** — memory gate at line ~648 (in run summary block):

```python
# BEFORE
if daap_memory is not None and session.user_id:

# AFTER
if daap_memory is not None:
```

**Fix 4** — rerun `execute_topology` call at line ~417:

```python
# BEFORE
result = await execute_topology(
    resolved=resolved,
    user_prompt=prompt,
    tracker=session.token_tracker,
)

# AFTER
result = await execute_topology(
    resolved=resolved,
    user_prompt=prompt,
    tracker=session.token_tracker,
    daap_memory=daap_memory,
    user_id=session.user_id,
)
```

**Fix 5** — dead fallback at line ~518 (RL tier recommendations):

```python
# BEFORE
recs = get_tier_recommendations(
    user_id=session.user_id or "anonymous",

# AFTER
recs = get_tier_recommendations(
    user_id=session.user_id,
```

- [ ] **Step 4: Run tests**

```
pytest daap/tests/test_api.py::test_session_manager_create_session_accepts_user_id daap/tests/test_api.py::test_session_manager_create_session_default_user_id -v
```
Expected: both PASS.

- [ ] **Step 5: Run full test suite to check no regressions**

```
pytest daap/tests/test_api.py -v
```
Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add daap/api/sessions.py daap/tests/test_api.py
git commit -m "fix(sessions): create_session accepts user_id, fix memory gate and rerun user_id"
```

---

## Task 5: `daap/api/routes.py` — `user_id` required + fix dead fallback

**Files:**
- Modify: `daap/api/routes.py`
- Modify: `daap/tests/test_api.py`

- [ ] **Step 1: Write failing test**

Add to `daap/tests/test_api.py`:

```python
def test_create_session_requires_user_id(client):
    """POST /session without user_id returns 422."""
    with patch("daap.api.routes.create_master_agent_with_toolkit", return_value=MagicMock()):
        resp = client.post("/session")
    assert resp.status_code == 422


def test_create_session_with_user_id(client):
    """POST /session?user_id=alice creates session with correct user_id."""
    with patch("daap.api.routes.create_master_agent_with_toolkit", return_value=MagicMock()):
        resp = client.post("/session?user_id=alice")
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    session = client.app.state  # access via session_manager below
    # Verify via config endpoint
    config_resp = client.get(f"/session/{session_id}/config")
    assert config_resp.json()["user_id"] == "alice"
```

- [ ] **Step 2: Run to verify failure**

```
pytest daap/tests/test_api.py::test_create_session_requires_user_id -v
```
Expected: FAIL — currently returns 200 (user_id defaults to "default").

- [ ] **Step 3: Apply fixes to `daap/api/routes.py`**

**Fix 1** — make `user_id` required in `create_session` endpoint (line ~232):

```python
# BEFORE
@app.post("/session")
async def create_session(
    user_id: str = "default",

# AFTER
@app.post("/session")
async def create_session(
    user_id: str,
```

**Fix 2** — dead fallback in `rate_run` (line ~356):

```python
# BEFORE
record_run_outcome(
    user_id=session.user_id or "anonymous",

# AFTER
record_run_outcome(
    user_id=session.user_id,
```

- [ ] **Step 4: Update existing tests that call `POST /session` without `user_id`**

In `daap/tests/test_api.py`, find `_create_session_with_mock_agent` helper (line ~56) and update:

```python
# BEFORE
resp = client.post("/session")

# AFTER
resp = client.post("/session?user_id=test-user")
```

Also update every other `client.post("/session")` call in the test file that doesn't pass `user_id`:

```python
# In test_create_session (line ~80):
resp = client.post("/session?user_id=test-user")

# In test_create_session_with_model_selection (line ~92):
# Already passes params — add user_id=test-user to the params dict or query string

# In test_list_sessions (line ~152-153):
client.post("/session?user_id=test-user")
client.post("/session?user_id=test-user")
```

- [ ] **Step 5: Run full test suite**

```
pytest daap/tests/test_api.py -v
```
Expected: all tests pass including the two new ones.

- [ ] **Step 6: Commit**

```bash
git add daap/api/routes.py daap/tests/test_api.py
git commit -m "fix(api): user_id required on POST /session, remove dead anonymous fallback"
```

---

## Task 6: `scripts/chat.py` — identity resolve, welcome line, `/profile`

**Files:**
- Modify: `scripts/chat.py`

- [ ] **Step 1: Import identity module and add `_build_welcome_line` helper**

At the top of `scripts/chat.py`, find the existing imports block and add:

```python
from daap.identity import load_local_user, resolve_cli_user
```

After the existing helper functions (before `main()`), add:

```python
def _build_welcome_line(user_id: str, topology_store, is_new_user: bool) -> str:
    """Assemble 'Welcome back, X — N runs · optimizer active · M memory facts' line."""
    if is_new_user:
        return ""  # first-run greeting already printed by resolve_cli_user

    display = user_id.replace("-", " ").title()
    segments: list[str] = []

    # Run count
    try:
        run_count = topology_store.count_runs(user_id)
        if run_count > 0:
            segments.append(f"{run_count} run{'s' if run_count != 1 else ''}")
    except Exception:
        pass

    # Optimizer state
    try:
        from daap.optimizer.store import BanditStore
        summary = BanditStore().get_profile_summary(user_id)
        if summary:
            segments.append("optimizer active")
        else:
            segments.append("optimizer learning")
    except Exception:
        pass

    # Memory facts
    try:
        from daap.memory.reader import load_user_profile
        facts = load_user_profile(user_id)
        if facts:
            segments.append(f"{len(facts)} memory fact{'s' if len(facts) != 1 else ''}")
    except Exception:
        pass

    suffix = " · ".join(segments)
    if suffix:
        return f"Welcome back, {display} — {suffix}"
    return f"Welcome back, {display}"
```

- [ ] **Step 2: Update `main()` to resolve identity before session creation**

In `main()`, find:

```python
async def main():
    args = _parse_args()
    raw_output = bool(args.raw_output)
    _print_header(raw_output)

    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        _print_warn("OPENROUTER_API_KEY is not set. API requests will fail until you set it.")

    # Init session
    session_mgr = SessionManager()
    session = session_mgr.create_session()
    topology_store = TopologyStore()
```

Replace with:

```python
async def main():
    args = _parse_args()
    raw_output = bool(args.raw_output)
    _print_header(raw_output)

    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        _print_warn("OPENROUTER_API_KEY is not set. API requests will fail until you set it.")

    # Resolve user identity (first-run prompt or load saved)
    is_new_user = load_local_user() is None
    user_id = await asyncio.get_event_loop().run_in_executor(None, resolve_cli_user)

    # Init session
    session_mgr = SessionManager()
    topology_store = TopologyStore()
    session = session_mgr.create_session(user_id=user_id)
    session.token_tracker = TokenTracker()
```

- [ ] **Step 3: Show welcome line after topology_store is initialised**

In `main()`, find the line:

```python
    session.token_tracker = TokenTracker()
```

Add welcome line immediately after (note: `session.token_tracker = TokenTracker()` was already in the original — remove the duplicate if present):

```python
    # Welcome line (only for returning users — new users got greeting in resolve_cli_user)
    if not is_new_user:
        welcome = _build_welcome_line(user_id, topology_store, is_new_user=False)
        _print_system(welcome)
```

- [ ] **Step 4: Add `/profile` command to the chat loop**

Find the `/skills` command block added earlier and add `/profile` right after it:

```python
            if cmd == "profile":
                display = user_id.replace("-", " ").title()
                _print_system(f"User: {display} ({user_id})")

                # Run count
                try:
                    run_count = topology_store.count_runs(user_id)
                    _print_system(f"Runs: {run_count}")
                except Exception:
                    _print_system("Runs: unavailable")

                # Optimizer summary
                try:
                    from daap.optimizer.store import BanditStore
                    summary = BanditStore().get_profile_summary(user_id)
                    if summary:
                        _print_system("Optimizer:")
                        for entry in summary:
                            print(f"  {entry['role']:<16} → {entry['best_arm']}  ({entry['n_pulls']} obs)")
                    else:
                        _print_system("Optimizer: learning (no runs yet)")
                except Exception:
                    _print_system("Optimizer: unavailable")

                # Memory facts
                try:
                    from daap.memory.reader import load_user_profile
                    facts = load_user_profile(user_id)
                    if facts:
                        _print_system(f"Memory ({len(facts)} facts):")
                        for f in facts[:3]:
                            print(f"  - {f}")
                        if len(facts) > 3:
                            print(f"  [+ {len(facts) - 3} more]")
                    else:
                        _print_system("Memory: no facts stored yet")
                except Exception:
                    _print_system("Memory: unavailable")
                continue
```

- [ ] **Step 5: Update help text in two places**

Find (appears twice — header and `/help` handler):

```python
"Commands: /help /approve /cheaper /cancel /mcp /skills /raw /clean /quit"
```

Replace both with:

```python
"Commands: /help /approve /cheaper /cancel /mcp /skills /profile /raw /clean /quit"
```

- [ ] **Step 6: Manual smoke test**

```bash
python scripts/chat.py
```

**First run:** should prompt `What's your name? ›`, save identity, show session ready line.
**Second run:** should show `Welcome back, Alice — optimizer learning` (or with run/memory counts).
Type `/profile` — should show user stats.
Type `/help` — should list `/profile` in commands.

- [ ] **Step 7: Commit**

```bash
git add scripts/chat.py
git commit -m "feat(cli): user identity prompt, welcome line, /profile command"
```

---

## Task 7: Full test suite + final verification

- [ ] **Step 1: Run all tests**

```
pytest daap/tests/ -v --tb=short
```
Expected: all tests pass.

- [ ] **Step 2: Verify no "default" leakage in new code paths**

```bash
grep -n '"default"' daap/api/routes.py daap/api/sessions.py scripts/chat.py
```
Expected: `routes.py` — no `user_id`-related "default". `sessions.py` — only `user_id: str = "default"` in `create_session` signature (the fallback for API backward compat). `chat.py` — none.

- [ ] **Step 3: Commit**

```bash
git commit --allow-empty -m "test: full suite green after user identity wiring"
```
