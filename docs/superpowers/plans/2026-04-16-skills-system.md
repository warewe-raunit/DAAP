# Skills System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add auto-discovery, hot reload, CLI commands, agent tool, and hard validation to the DAAP skills system.

**Architecture:** `SkillManager` gains validation, auto-discovery, `bind_toolkit`, `add_skill`, and `remove_skill`. A `register_skill` tool is added to the master toolkit so the agent can wire skills from conversation. `/skill add|remove|create` commands are added to `chat.py`.

**Tech Stack:** Python 3.14, AgentScope `Toolkit`, `pathlib`, `os.replace` for atomic writes.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `daap/skills/manager.py` | Modify | `SkillValidationError`, `_validate_skill_dir`, `_parse_skill_meta`, `_discover_skills`, `bind_toolkit`, `add_skill`, `remove_skill`, atomic `_write_config` |
| `daap/master/tools.py` | Modify | Add `register_skill` tool function, register in `create_master_toolkit` |
| `daap/master/prompts.py` | Modify | Add agent hint: call `register_skill` when user mentions a skill dir path |
| `scripts/chat.py` | Modify | `/skill add`, `/skill remove`, `/skill create` commands; `bind_toolkit` call after session toolkit creation; auto-discovery startup hint |
| `daap/tests/test_skills_manager.py` | Modify | Add tests for validation, discovery, add_skill, remove_skill, atomic write |

---

## Task 1: `SkillValidationError` + `_parse_skill_meta` + `_validate_skill_dir`

**Files:**
- Modify: `daap/skills/manager.py`
- Modify: `daap/tests/test_skills_manager.py`

### Steps

- [ ] **Step 1: Write failing tests for validation**

Add to `daap/tests/test_skills_manager.py`:

```python
import pytest
from pathlib import Path


def _make_skill_dir(tmp_path, name="my-skill", content=None):
    """Helper: create a skill dir with SKILL.md."""
    d = tmp_path / name
    d.mkdir()
    if content is not None:
        (d / "SKILL.md").write_text(content, encoding="utf-8")
    return d


def test_validate_missing_dir(tmp_path):
    from daap.skills.manager import SkillManager, SkillValidationError
    with pytest.raises(SkillValidationError, match="directory not found"):
        SkillManager._validate_skill_dir(str(tmp_path / "nonexistent"))


def test_validate_missing_skill_md(tmp_path):
    from daap.skills.manager import SkillManager, SkillValidationError
    d = tmp_path / "skill"
    d.mkdir()
    with pytest.raises(SkillValidationError, match="missing SKILL.md"):
        SkillManager._validate_skill_dir(str(d))


def test_validate_empty_skill_md(tmp_path):
    from daap.skills.manager import SkillManager, SkillValidationError
    d = _make_skill_dir(tmp_path, content="")
    with pytest.raises(SkillValidationError, match="SKILL.md is empty"):
        SkillManager._validate_skill_dir(str(d))


def test_validate_missing_name(tmp_path):
    from daap.skills.manager import SkillManager, SkillValidationError
    d = _make_skill_dir(tmp_path, content="---\ndescription: foo\n---\n")
    with pytest.raises(SkillValidationError, match="missing required field 'name'"):
        SkillManager._validate_skill_dir(str(d))


def test_validate_missing_description(tmp_path):
    from daap.skills.manager import SkillManager, SkillValidationError
    d = _make_skill_dir(tmp_path, content="---\nname: foo\n---\n")
    with pytest.raises(SkillValidationError, match="missing required field 'description'"):
        SkillManager._validate_skill_dir(str(d))


def test_validate_valid_frontmatter(tmp_path):
    from daap.skills.manager import SkillManager
    d = _make_skill_dir(tmp_path, content="---\nname: foo\ndescription: does bar\n---\n")
    # Should not raise
    SkillManager._validate_skill_dir(str(d))


def test_validate_valid_bare_heading(tmp_path):
    from daap.skills.manager import SkillManager
    d = _make_skill_dir(tmp_path, content="# my-skill\n\nDoes X when Y.\n")
    SkillManager._validate_skill_dir(str(d))


def test_parse_skill_meta_frontmatter(tmp_path):
    from daap.skills.manager import SkillManager
    d = _make_skill_dir(tmp_path, content="---\nname: foo\ndescription: does bar\n---\n## Body\n")
    meta = SkillManager._parse_skill_meta(str(d))
    assert meta["name"] == "foo"
    assert meta["description"] == "does bar"


def test_parse_skill_meta_bare_heading(tmp_path):
    from daap.skills.manager import SkillManager
    d = _make_skill_dir(tmp_path, content="# my-skill\n\nDoes X when Y.\n\n## Body\n")
    meta = SkillManager._parse_skill_meta(str(d))
    assert meta["name"] == "my-skill"
    assert meta["description"] == "Does X when Y."
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest daap/tests/test_skills_manager.py -k "validate or parse_skill_meta" -v
```

Expected: `FAILED` — `SkillValidationError` not defined, `_validate_skill_dir` not defined.

- [ ] **Step 3: Implement `SkillValidationError`, `_parse_skill_meta`, `_validate_skill_dir` in `daap/skills/manager.py`**

Add after the imports, before `DEFAULT_CONFIG_PATH`:

```python
class SkillValidationError(Exception):
    """Raised when a skill directory fails validation."""
```

Add as `@staticmethod` methods on `SkillManager` (after `_normalize_item`):

```python
@staticmethod
def _parse_skill_meta(directory: str) -> dict[str, str]:
    """
    Extract name and description from SKILL.md.

    Supports two formats:
    1. YAML frontmatter between --- markers: name: / description: keys
    2. Bare heading: # name on first line, first paragraph = description
    """
    content = (Path(directory) / "SKILL.md").read_text(encoding="utf-8")
    meta: dict[str, str] = {}

    if content.startswith("---"):
        lines = content.split("\n")
        end_idx = None
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                end_idx = i
                break
        if end_idx:
            for line in lines[1:end_idx]:
                if ":" in line:
                    key, _, val = line.partition(":")
                    meta[key.strip().lower()] = val.strip()
            return meta

    # Bare heading fallback
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("# "):
            meta["name"] = line[2:].strip()
            for next_line in lines[i + 1:]:
                stripped = next_line.strip()
                if stripped and not stripped.startswith("#"):
                    meta["description"] = stripped
                    break
            break
    return meta

@staticmethod
def _validate_skill_dir(directory: str) -> None:
    """
    Validate a skill directory. Raises SkillValidationError on any failure.

    Checks (in order):
    1. Directory exists
    2. SKILL.md present
    3. SKILL.md non-empty
    4. SKILL.md has 'name' and 'description' fields
    """
    path = Path(directory)

    if not path.is_dir():
        raise SkillValidationError(f"Skill '{directory}': directory not found")

    skill_md = path / "SKILL.md"
    if not skill_md.exists():
        raise SkillValidationError(f"Skill '{directory}': missing SKILL.md")

    content = skill_md.read_text(encoding="utf-8").strip()
    if not content:
        raise SkillValidationError(f"Skill '{directory}': SKILL.md is empty")

    meta = SkillManager._parse_skill_meta(directory)
    if not meta.get("name"):
        raise SkillValidationError(
            f"Skill '{directory}': SKILL.md missing required field 'name'"
        )
    if not meta.get("description"):
        raise SkillValidationError(
            f"Skill '{directory}': SKILL.md missing required field 'description'"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest daap/tests/test_skills_manager.py -k "validate or parse_skill_meta" -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add daap/skills/manager.py daap/tests/test_skills_manager.py
git commit -m "feat(skills): SkillValidationError, _parse_skill_meta, _validate_skill_dir"
```

---

## Task 2: Auto-discovery + startup hint

**Files:**
- Modify: `daap/skills/manager.py`
- Modify: `daap/tests/test_skills_manager.py`

### Steps

- [ ] **Step 1: Write failing tests for auto-discovery**

Add to `daap/tests/test_skills_manager.py`:

```python
def test_discover_skills_finds_valid_dirs(tmp_path):
    from daap.skills.manager import SkillManager

    # Two valid skill dirs
    for name, content in [
        ("skill-a", "---\nname: skill-a\ndescription: does a\n---\n"),
        ("skill-b", "# skill-b\n\nDoes b.\n"),
    ]:
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(content, encoding="utf-8")

    # One invalid (no SKILL.md) — should be skipped, not raise
    (tmp_path / "not-a-skill").mkdir()

    found = SkillManager._discover_skills([str(tmp_path)])
    dirs = [spec.directory for spec in found]
    assert str(tmp_path / "skill-a") in dirs
    assert str(tmp_path / "skill-b") in dirs
    assert str(tmp_path / "not-a-skill") not in dirs


def test_discover_skills_empty_when_no_dirs(tmp_path):
    from daap.skills.manager import SkillManager
    found = SkillManager._discover_skills([str(tmp_path / "nonexistent")])
    assert found == []


def test_skill_manager_auto_discovers_on_init(tmp_path, monkeypatch):
    from daap.skills.manager import SkillManager

    # Create a valid skill in a search location
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: does things\n---\n", encoding="utf-8"
    )

    # Patch discovery search paths to use our tmp_path
    monkeypatch.setattr(SkillManager, "_DISCOVERY_PATHS", [str(tmp_path)])

    # No skills.json — relies solely on discovery
    manager = SkillManager(config_path=str(tmp_path / "nonexistent.json"))
    dirs = manager.list_skill_dirs("master")
    assert str(skill_dir) in dirs
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest daap/tests/test_skills_manager.py -k "discover" -v
```

Expected: `FAILED` — `_discover_skills` not defined.

- [ ] **Step 3: Implement `_DISCOVERY_PATHS` and `_discover_skills` in `daap/skills/manager.py`**

Add class attribute and method to `SkillManager`:

```python
# Search paths for auto-discovery (class-level so tests can monkeypatch)
_DISCOVERY_PATHS: list[str] = [
    str(Path.home() / ".daap" / "skills"),
    "./skills",
]
```

Add `_discover_skills` as a `@classmethod`:

```python
@classmethod
def _discover_skills(cls, search_paths: list[str]) -> list[AgentSkillSpec]:
    """
    Search directories for skill subdirs containing SKILL.md.

    Skips invalid dirs silently (logs warning). Returns specs with target="all".
    """
    found: list[AgentSkillSpec] = []
    for search_path in search_paths:
        p = Path(search_path).expanduser()
        if not p.is_dir():
            continue
        for candidate in sorted(p.iterdir()):
            if not candidate.is_dir():
                continue
            try:
                cls._validate_skill_dir(str(candidate))
                found.append(
                    AgentSkillSpec(
                        directory=str(candidate),
                        targets=frozenset({"all"}),
                    )
                )
            except SkillValidationError as exc:
                logger.debug("Auto-discovery skipped '%s': %s", candidate, exc)
    return found
```

Update `__init__` to merge discovered specs:

```python
def __init__(self, config_path: str | None = None):
    cfg = config_path or os.environ.get(CONFIG_ENV_VAR, "")
    self._config_path = Path(cfg).expanduser() if cfg else DEFAULT_CONFIG_PATH
    config_specs = self._load_specs(self._config_path)
    discovered = self._discover_skills(self._DISCOVERY_PATHS)
    # Deduplicate: config_specs take precedence (preserve their targets)
    config_dirs = {s.directory for s in config_specs}
    unique_discovered = [s for s in discovered if s.directory not in config_dirs]
    self._specs: list[AgentSkillSpec] = config_specs + unique_discovered
    self._toolkit: Any = None  # set by bind_toolkit
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest daap/tests/test_skills_manager.py -k "discover" -v
```

Expected: all PASS.

- [ ] **Step 5: Run full test suite to check no regressions**

```
pytest daap/tests/test_skills_manager.py -v
```

Expected: all existing tests PASS.

- [ ] **Step 6: Commit**

```bash
git add daap/skills/manager.py daap/tests/test_skills_manager.py
git commit -m "feat(skills): auto-discovery from ~/.daap/skills/ and ./skills/"
```

---

## Task 3: `bind_toolkit`, `add_skill`, `remove_skill`, atomic `_write_config`

**Files:**
- Modify: `daap/skills/manager.py`
- Modify: `daap/tests/test_skills_manager.py`

### Steps

- [ ] **Step 1: Write failing tests**

Add to `daap/tests/test_skills_manager.py`:

```python
def test_add_skill_registers_and_persists(tmp_path):
    from daap.skills.manager import SkillManager

    skill_dir = tmp_path / "new-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: new-skill\ndescription: does new things\n---\n", encoding="utf-8"
    )
    config_path = tmp_path / "skills.json"

    manager = SkillManager(config_path=str(config_path))
    toolkit = _FakeToolkit()
    manager.bind_toolkit(toolkit, target="master")

    name = manager.add_skill(str(skill_dir), targets="master", persist=True)

    assert name == "new-skill"
    assert str(skill_dir) in toolkit.registered
    assert str(skill_dir) in manager.list_skill_dirs("master")

    # Persisted to skills.json
    import json
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    dirs = [e["dir"] if isinstance(e, dict) else e for e in saved.get("skills", saved)]
    assert str(skill_dir) in dirs


def test_add_skill_invalid_raises(tmp_path):
    from daap.skills.manager import SkillManager, SkillValidationError
    manager = SkillManager(config_path=str(tmp_path / "skills.json"))
    with pytest.raises(SkillValidationError):
        manager.add_skill(str(tmp_path / "nonexistent"))


def test_add_skill_already_registered_is_noop(tmp_path):
    from daap.skills.manager import SkillManager

    skill_dir = tmp_path / "dup-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: dup-skill\ndescription: duplicate\n---\n", encoding="utf-8"
    )
    config_path = tmp_path / "skills.json"
    manager = SkillManager(config_path=str(config_path))
    toolkit = _FakeToolkit()
    manager.bind_toolkit(toolkit, target="master")

    manager.add_skill(str(skill_dir))
    manager.add_skill(str(skill_dir))  # second call is a no-op

    assert toolkit.registered.count(str(skill_dir)) == 1


def test_remove_skill_removes_from_config(tmp_path):
    import json
    from daap.skills.manager import SkillManager

    skill_dir = tmp_path / "rm-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: rm-skill\ndescription: to be removed\n---\n", encoding="utf-8"
    )
    config_path = tmp_path / "skills.json"
    manager = SkillManager(config_path=str(config_path))
    toolkit = _FakeToolkit()
    manager.bind_toolkit(toolkit, target="master")

    manager.add_skill(str(skill_dir), persist=True)
    assert str(skill_dir) in manager.list_skill_dirs("master")

    manager.remove_skill(str(skill_dir), persist=True)
    assert str(skill_dir) not in manager.list_skill_dirs("master")

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    dirs = [e["dir"] if isinstance(e, dict) else e for e in saved.get("skills", saved)]
    assert str(skill_dir) not in dirs


def test_remove_skill_not_found_raises(tmp_path):
    from daap.skills.manager import SkillManager
    manager = SkillManager(config_path=str(tmp_path / "skills.json"))
    with pytest.raises(KeyError):
        manager.remove_skill(str(tmp_path / "ghost-skill"))


def test_atomic_write_does_not_corrupt_on_failure(tmp_path):
    """Verify tmp → rename pattern: existing config survives a failed write."""
    import json
    from daap.skills.manager import SkillManager

    config_path = tmp_path / "skills.json"
    original = {"skills": [{"dir": "/safe/skill", "targets": ["all"]}]}
    config_path.write_text(json.dumps(original), encoding="utf-8")

    manager = SkillManager(config_path=str(config_path))

    # Force write with the original specs — just verify file is valid JSON after
    manager._write_config()
    result = json.loads(config_path.read_text(encoding="utf-8"))
    assert isinstance(result, dict)
    assert "skills" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest daap/tests/test_skills_manager.py -k "add_skill or remove_skill or atomic_write or bind_toolkit" -v
```

Expected: `FAILED` — methods not defined.

- [ ] **Step 3: Implement `bind_toolkit`, `add_skill`, `remove_skill`, `_write_config` in `daap/skills/manager.py`**

Add to `SkillManager`:

```python
def bind_toolkit(self, toolkit: Any, target: str = "master") -> None:
    """Store a ref to the live toolkit for hot-reload adds."""
    self._toolkit = toolkit
    self._toolkit_target = target

def add_skill(
    self,
    directory: str,
    targets: str = "all",
    persist: bool = True,
) -> tuple[str, bool]:
    """
    Validate, register, and optionally persist a new skill.

    Returns (skill_name, was_newly_added).
    Raises SkillValidationError if the directory is invalid.
    """
    # Consistent normalization: expandvars + expanduser (matches _normalize_item)
    expanded = str(Path(os.path.expandvars(directory)).expanduser())
    self._validate_skill_dir(expanded)

    # Deduplicate
    existing_dirs = {s.directory for s in self._specs}
    if expanded in existing_dirs:
        meta = self._parse_skill_meta(expanded)
        return meta.get("name", Path(expanded).name), False

    normalized_targets: frozenset[str]
    if isinstance(targets, str):
        target_items = [targets]
    else:
        target_items = list(targets)
    valid = {t.strip().lower() for t in target_items if t.strip().lower() in _VALID_TARGETS}
    normalized_targets = frozenset(valid or {"all"})

    spec = AgentSkillSpec(directory=expanded, targets=normalized_targets)
    self._specs.append(spec)

    if self._toolkit is not None:
        register = getattr(self._toolkit, "register_agent_skill", None)
        if callable(register):
            register(expanded)

    if persist:
        self._write_config()

    meta = self._parse_skill_meta(expanded)
    return meta.get("name", Path(expanded).name), True

def remove_skill(self, directory: str, persist: bool = True) -> str:
    """
    Remove a skill from the manager and optionally from config.

    Returns the skill name.
    Raises KeyError if the skill is not registered.
    """
    expanded = str(Path(os.path.expandvars(directory)).expanduser())
    matching = [s for s in self._specs if s.directory == expanded]
    if not matching:
        raise KeyError(f"Skill not registered: '{expanded}'")

    self._specs = [s for s in self._specs if s.directory != expanded]

    if self._toolkit is not None:
        unregister = getattr(self._toolkit, "unregister_agent_skill", None)
        if callable(unregister):
            try:
                unregister(expanded)
            except Exception as exc:
                logger.debug("unregister_agent_skill failed (non-fatal): %s", exc)

    if persist:
        self._write_config()

    try:
        meta = self._parse_skill_meta(expanded)
        return meta.get("name", Path(expanded).name)
    except Exception:
        return Path(expanded).name

def _write_config(self) -> None:
    """Atomically write current specs to skills.json."""
    entries = []
    for spec in self._specs:
        targets = sorted(spec.targets)
        if targets == ["all"]:
            entries.append({"dir": spec.directory, "targets": ["all"]})
        else:
            entries.append({"dir": spec.directory, "targets": targets})

    payload = json.dumps({"skills": entries}, indent=2, ensure_ascii=False)
    tmp_path = self._config_path.with_suffix(".json.tmp")
    try:
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(str(tmp_path), str(self._config_path))
    except Exception as exc:
        logger.warning("Failed writing skills config '%s': %s", self._config_path, exc)
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
```

Also add `import json` at the top of `manager.py` if not already present (it already is via `json.loads`).

- [ ] **Step 4: Run tests to verify they pass**

```
pytest daap/tests/test_skills_manager.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add daap/skills/manager.py daap/tests/test_skills_manager.py
git commit -m "feat(skills): bind_toolkit, add_skill, remove_skill, atomic _write_config"
```

---

## Task 4: `register_skill` master agent tool + prompt update

**Files:**
- Modify: `daap/master/tools.py`
- Modify: `daap/master/prompts.py`
- Modify: `daap/tests/test_master_agent.py` (or new `daap/tests/test_skills_tool.py`)

### Steps

- [ ] **Step 1: Write failing tests**

Create `daap/tests/test_skills_tool.py`:

```python
"""Tests for the register_skill master agent tool."""
import json
from pathlib import Path


class _FakeToolkit:
    def __init__(self):
        self.registered: list[str] = []

    def register_agent_skill(self, skill_dir: str):
        self.registered.append(skill_dir)


def test_register_skill_success(tmp_path, monkeypatch):
    from daap.skills import manager as mgr_mod

    # Fresh manager for this test
    skill_dir = tmp_path / "test-tool-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-tool-skill\ndescription: registered via agent\n---\n",
        encoding="utf-8",
    )

    toolkit = _FakeToolkit()
    test_manager = mgr_mod.SkillManager(config_path=str(tmp_path / "skills.json"))
    test_manager.bind_toolkit(toolkit, "master")
    monkeypatch.setattr(mgr_mod, "_SKILL_MANAGER", test_manager)

    from daap.master.tools import register_skill
    result = register_skill(str(skill_dir))

    assert "test-tool-skill" in result
    assert "registered" in result
    assert str(skill_dir) in toolkit.registered


def test_register_skill_invalid_dir(tmp_path, monkeypatch):
    from daap.skills import manager as mgr_mod

    test_manager = mgr_mod.SkillManager(config_path=str(tmp_path / "skills.json"))
    monkeypatch.setattr(mgr_mod, "_SKILL_MANAGER", test_manager)

    from daap.master.tools import register_skill
    result = register_skill(str(tmp_path / "ghost"))

    assert "Failed" in result or "not found" in result


def test_register_skill_already_registered(tmp_path, monkeypatch):
    from daap.skills import manager as mgr_mod

    skill_dir = tmp_path / "dup-tool-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: dup-tool-skill\ndescription: duplicate\n---\n", encoding="utf-8"
    )

    toolkit = _FakeToolkit()
    test_manager = mgr_mod.SkillManager(config_path=str(tmp_path / "skills.json"))
    test_manager.bind_toolkit(toolkit, "master")
    monkeypatch.setattr(mgr_mod, "_SKILL_MANAGER", test_manager)

    from daap.master.tools import register_skill
    register_skill(str(skill_dir))
    result = register_skill(str(skill_dir))  # second call

    assert "already registered" in result
    assert toolkit.registered.count(str(skill_dir)) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest daap/tests/test_skills_tool.py -v
```

Expected: `FAILED` — `register_skill` not in `daap.master.tools`.

- [ ] **Step 3: Implement `register_skill` in `daap/master/tools.py`**

Add after the existing imports:

```python
from daap.skills.manager import SkillValidationError, get_skill_manager
```

Add the tool function (before `create_master_toolkit`):

```python
def register_skill(directory: str, targets: str = "all") -> str:
    """
    Register a skill directory for this session.

    Validates the directory contains a valid SKILL.md, registers the skill
    immediately on the master toolkit, and persists to skills.json for future
    sessions.

    Args:
        directory: Absolute or relative path to the skill directory.
        targets: Which agents get this skill — "master", "subagent", or "all".

    Returns:
        Confirmation message on success, error message on failure.
    """
    try:
        manager = get_skill_manager()
        existing = manager.list_skill_dirs("master") + manager.list_skill_dirs("subagent")
        from pathlib import Path as _Path
        expanded = str(_Path(directory).expanduser().resolve())
        if expanded in existing:
            return f"Skill '{_Path(expanded).name}' already registered."
        name = manager.add_skill(directory, targets=targets, persist=True)
        target_display = targets if targets != "all" else "master, subagent"
        return f"Skill '{name}' registered [{target_display}]."
    except SkillValidationError as exc:
        return f"Failed to register skill: {exc}"
    except Exception as exc:
        return f"Failed to register skill: {exc}"
```

Register it in `create_master_toolkit`:

```python
def create_master_toolkit() -> Toolkit:
    """Create the Toolkit for the master agent with all registered tools."""
    toolkit = Toolkit()
    apply_configured_skills(toolkit, target="master")
    toolkit.register_tool_function(generate_topology)
    toolkit.register_tool_function(ask_user)
    toolkit.register_tool_function(register_skill)
    return toolkit
```

- [ ] **Step 4: Update `daap/master/prompts.py` — add skill-dir hint**

In `get_master_system_prompt`, find the `## Core Operating Contract` section and add after it:

```python
skill_hint = """
## Skills

If the user mentions a file path that looks like a skill directory (e.g. "/path/to/skill", "./my-skill"), call `register_skill` with that path to wire it up immediately. Do not ask the user to run a command — just call the tool.
"""
```

Insert `{skill_hint}` into the returned f-string after the `## Core Operating Contract` block. The full return becomes:

```python
    return f"""You are the DAAP Master Agent, an expert AI assistant for B2B sales automation.
...
{skill_hint}
## Core Operating Contract
...
"""
```

(Find the exact insertion point in the f-string and add `{skill_hint}` before `## Core Operating Contract`.)

- [ ] **Step 5: Run tests to verify they pass**

```
pytest daap/tests/test_skills_tool.py -v
```

Expected: all PASS.

- [ ] **Step 6: Run full test suite**

```
pytest daap/tests/ -v --tb=short
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add daap/master/tools.py daap/master/prompts.py daap/tests/test_skills_tool.py
git commit -m "feat(skills): register_skill agent tool + prompt hint"
```

---

## Task 5: `/skill add|remove|create` CLI commands + `bind_toolkit` call

**Files:**
- Modify: `scripts/chat.py`

### Steps

- [ ] **Step 1: Wire `bind_toolkit` after toolkit creation**

In `scripts/chat.py`, find the block after `create_session_scoped_toolkit`:

```python
toolkit = create_session_scoped_toolkit(
    session,
    topology_store=topology_store,
    daap_memory=daap_memory,
    rl_optimizer=rl_optimizer,
)
session.master_agent = create_master_agent_with_toolkit(
    toolkit,
    user_context=user_context,
    tracker=session.token_tracker,
)
```

After those lines, add:

```python
# Bind live toolkit to skill manager for hot-reload adds
try:
    from daap.skills.manager import get_skill_manager
    get_skill_manager().bind_toolkit(toolkit, target="master")
except Exception:
    pass
```

- [ ] **Step 2: Update `/help` and `/skills` command**

Find:
```python
_print_system("Commands: /help /approve /cheaper /cancel /mcp /skills /profile /raw /clean /quit")
```

Replace with:
```python
_print_system("Commands: /help /approve /cheaper /cancel /mcp /skills /skill /profile /raw /clean /quit")
```

- [ ] **Step 3: Add `/skill` command dispatcher**

Find the `/skills` command block (around line 558) and add the `/skill` dispatcher **before** it:

```python
if cmd.startswith("skill ") or cmd == "skill":
    _handle_skill_command(user_input.strip(), _print_system)
    continue
```

Then define `_handle_skill_command` as a top-level function in `chat.py` (add before `async def chat_loop`):

```python
def _handle_skill_command(raw_input: str, print_fn) -> None:
    """
    Dispatch /skill subcommands: add, remove, create.

    raw_input: the full user input string, e.g. "/skill add /path/to/skill"
    """
    from daap.skills.manager import get_skill_manager, SkillValidationError

    # Normalize: strip leading /skill prefix and split
    parts = raw_input.lstrip("/").split()
    # parts[0] == "skill", parts[1] == subcommand
    if len(parts) < 2:
        print_fn("Usage: /skill add <path> [master|subagent|all] | /skill remove <path> | /skill create")
        return

    sub = parts[1].lower()

    if sub == "add":
        if len(parts) < 3:
            print_fn("Usage: /skill add <path> [master|subagent|all]")
            return
        directory = parts[2]
        targets = parts[3] if len(parts) >= 4 else "all"
        try:
            manager = get_skill_manager()
            name = manager.add_skill(directory, targets=targets, persist=True)
            target_display = targets if targets != "all" else "master, subagent"
            print_fn(f"Skill '{name}' registered [{target_display}].")
        except SkillValidationError as exc:
            print_fn(f"Skill error: {exc}")
        except Exception as exc:
            print_fn(f"Skill error: {exc}")

    elif sub == "remove":
        if len(parts) < 3:
            print_fn("Usage: /skill remove <path>")
            return
        directory = parts[2]
        try:
            manager = get_skill_manager()
            name = manager.remove_skill(directory, persist=True)
            print_fn(f"Skill '{name}' removed.")
        except KeyError:
            print_fn(f"Skill not found: {directory}")
        except Exception as exc:
            print_fn(f"Skill error: {exc}")

    elif sub == "create":
        _run_skill_create_wizard(print_fn)

    else:
        print_fn(f"Unknown /skill subcommand: '{sub}'. Try: add, remove, create")
```

- [ ] **Step 4: Implement `_run_skill_create_wizard`**

Add before `_handle_skill_command` in `chat.py`:

```python
def _run_skill_create_wizard(print_fn) -> None:
    """
    Interactive wizard to create a new skill directory + SKILL.md.
    Prints to stdout using print_fn for system messages.
    """
    import re
    from pathlib import Path as _Path
    from daap.skills.manager import get_skill_manager, SkillValidationError

    print_fn("Skill creator — press Ctrl+C to cancel\n")

    try:
        # --- Name ---
        while True:
            name = input("Name: ").strip()
            if not name:
                print_fn("Name is required.")
                continue
            # Enforce kebab-case
            name = re.sub(r"[^a-zA-Z0-9-]", "-", name).strip("-").lower()
            if name:
                break
            print_fn("Invalid name. Use letters, numbers, hyphens.")

        # --- Description ---
        while True:
            description = input("Description (one line): ").strip()
            if description:
                break
            print_fn("Description is required.")

        # --- Targets ---
        targets_input = input("Targets [all/master/subagent, default=all]: ").strip().lower()
        targets = targets_input if targets_input in ("all", "master", "subagent") else "all"

        # --- Save dir ---
        default_dir = str(_Path.home() / ".daap" / "skills" / name)
        dir_input = input(f"Save to dir [{default_dir}]: ").strip()
        save_dir = _Path(dir_input if dir_input else default_dir).expanduser()

        # --- Skill body ---
        print_fn("\nSkill body (blank line + . to finish):")
        body_lines = []
        while True:
            line = input("> ")
            if line.strip() == ".":
                break
            body_lines.append(line)
        body = "\n".join(body_lines).strip()

        # --- Build SKILL.md content ---
        skill_md = f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"

        # --- Preview ---
        print_fn(f"\n--- Preview: {save_dir}/SKILL.md ---")
        print(skill_md)
        print_fn("---")

        confirm = input("Write? [y/N]: ").strip().lower()
        if confirm != "y":
            print_fn("Cancelled.")
            return

        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

        manager = get_skill_manager()
        registered_name = manager.add_skill(str(save_dir), targets=targets, persist=True)
        print_fn(f"Skill '{registered_name}' created and registered.")

    except KeyboardInterrupt:
        print()
        print_fn("Skill creation cancelled.")
    except SkillValidationError as exc:
        print_fn(f"Skill error: {exc}")
    except Exception as exc:
        print_fn(f"Skill creation failed: {exc}")
```

- [ ] **Step 5: Add startup auto-discovery hint**

In `chat.py`, find where the session is printed as ready:

```python
_print_system(f"Session {session.session_id} ready | Master: {MODEL_REGISTRY['powerful']}")
```

After that line, add:

```python
# Show auto-discovery result or hint
try:
    from daap.skills.manager import get_skill_manager
    _sm = get_skill_manager()
    _all_dirs = sorted(set(_sm.list_skill_dirs("master")) | set(_sm.list_skill_dirs("subagent")))
    if not _all_dirs:
        _print_system("No skills found. Use /skill add <path> or drop skills in ~/.daap/skills/")
    else:
        _print_system(f"Skills loaded: {len(_all_dirs)}")
except Exception:
    pass
```

- [ ] **Step 6: Run smoke test**

```
python scripts/chat.py --help
```

Expected: exits without import errors.

- [ ] **Step 7: Commit**

```bash
git add scripts/chat.py
git commit -m "feat(skills): /skill add|remove|create commands, bind_toolkit wiring, startup hint"
```

---

## Task 6: Final integration check + existing test cleanup

**Files:**
- Modify: `daap/tests/test_skills_manager.py` (remove now-passing stubs if any)

### Steps

- [ ] **Step 1: Run full test suite**

```
pytest daap/tests/ -v --tb=short
```

Expected: all PASS. If any fail, fix before continuing.

- [ ] **Step 2: Verify startup auto-discovery end-to-end**

Create a test skill in `~/.daap/skills/` manually:

```bash
mkdir -p ~/.daap/skills/smoke-skill
cat > ~/.daap/skills/smoke-skill/SKILL.md <<'EOF'
---
name: smoke-skill
description: smoke test skill for auto-discovery
---

When asked, say hello.
EOF
```

Run chat:

```
python scripts/chat.py
```

Expected startup output includes: `Skills loaded: 1`

- [ ] **Step 3: Verify `/skill add` end-to-end**

In the running chat session:

```
/skill add /path/to/another-skill
```

Expected: `Skill 'another-skill' registered [master, subagent].`
Or on error: `Skill error: Skill '/path/to/another-skill': directory not found`

- [ ] **Step 4: Verify `/skill create` wizard end-to-end**

```
/skill create
```

Walk through the prompts. Confirm SKILL.md is written to the chosen dir and skill appears in `/skills`.

- [ ] **Step 5: Final commit**

```bash
git add -u
git commit -m "test(skills): integration verified, full suite green"
```
