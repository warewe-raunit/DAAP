# Skills System — Design Spec

**Date:** 2026-04-16
**Status:** Approved

## Overview

Four improvements to the DAAP skills system:

1. **Auto-discovery** — find skills without manual `skills.json` config
2. **Hot reload** — add/remove skills mid-session without restart
3. **`/skill` CLI commands** — add, remove, create wizard
4. **Skill validation** — hard fail with clear errors instead of silent ignore

## Architecture

### Components changed

| Component | Change |
|---|---|
| `daap/skills/manager.py` | Validation, hot reload, auto-discovery, toolkit ref |
| `daap/master/tools.py` | `register_skill` tool added to master toolkit |
| `daap/master/prompts.py` | Agent hint: call `register_skill` when user mentions a path |
| `scripts/chat.py` | `/skill add`, `/skill remove`, `/skill create` commands |

---

## Section 1: Auto-discovery

At session start, `SkillManager` searches these locations in order:

1. `~/.daap/skills/` — user global skills
2. `./skills/` — project-local skills (cwd at startup)

Any **subdirectory** containing a `SKILL.md` file is a candidate. Found dirs are registered with `targets="all"` by default.

If no skills found from any source (auto-discovery + `skills.json`), print once at startup:

```
[DAAP] No skills found. Use /skill add <path> or drop skills in ~/.daap/skills/
```

Auto-discovery runs **before** the master agent starts — agent sees all discovered skills immediately.

---

## Section 2: Skill Validation

`SkillManager` validates every skill dir before registering — at startup, from `skills.json`, from auto-discovery, and from runtime adds.

### `SkillValidationError`

New exception class in `daap/skills/manager.py`.

### Validation rules

| Check | Error message |
|---|---|
| Dir exists | `"Skill '<path>': directory not found"` |
| `SKILL.md` present | `"Skill '<path>': missing SKILL.md"` |
| `SKILL.md` non-empty | `"Skill '<path>': SKILL.md is empty"` |
| Has `name:` field | `"Skill '<path>': SKILL.md missing required field 'name'"` |
| Has `description:` field | `"Skill '<path>': SKILL.md missing required field 'description'"` |

### `name` and `description` detection

Accepts two formats:

**Frontmatter:**
```markdown
---
name: my-skill
description: Does X when Y
---
```

**Bare heading:**
```markdown
# my-skill

Does X when Y
```

For bare heading: first `# ` line = name, first non-empty paragraph after heading = description.

### Behavior on failure

- **At startup / `skills.json` load** → log error + skip that skill (don't crash session)
- **At runtime add** (`/skill add` or agent `register_skill` tool) → raise `SkillValidationError`, surface exact message to user, do not register

---

## Section 3: Hot Reload + Persistence

### `SkillManager` changes

Holds a ref to the live master toolkit (set after toolkit creation):

```python
manager.bind_toolkit(toolkit, target="master")
```

#### `add_skill(directory, targets="all", persist=True) -> str`

1. Validate dir (raises `SkillValidationError` on failure)
2. Call `toolkit.register_agent_skill(directory)` — live, no restart
3. Append spec to `self._specs`
4. If `persist=True` → write to `skills.json` (atomic: write temp → rename)
5. Return skill name from `SKILL.md`

#### `remove_skill(directory, persist=True) -> str`

1. Remove matching spec from `self._specs`
2. Attempt `toolkit.unregister_agent_skill(directory)` if method exists
3. If AgentScope doesn't support unregister → mark inactive in manager, note to user: `"Skill removed from config. Takes full effect on restart."`
4. If `persist=True` → rewrite `skills.json` without the entry (atomic)
5. Return skill name

### `skills.json` write

Atomic write: write to `skills.json.tmp` → `os.replace()` to `skills.json`. Never corrupts existing config on failure.

---

## Section 4: CLI Slash Commands

New commands in `scripts/chat.py`, consistent with existing `/skills`, `/profile`, `/mcp`.

### `/skill add <path> [master|subagent|all]`

```
/skill add /home/user/my-skill
/skill add ./tools/summarizer master
```

Flow:
1. Parse path (required) + target (optional, default `all`)
2. Call `skill_manager.add_skill(path, targets, persist=True)`
3. Success: `[DAAP] Skill 'my-skill' registered [master, subagent].`
4. `SkillValidationError`: print exact error, no registration

### `/skill remove <path>`

```
/skill remove /home/user/my-skill
```

Flow:
1. Call `skill_manager.remove_skill(path, persist=True)`
2. Success: `[DAAP] Skill 'my-skill' removed.` (+ restart note if live unregister unsupported)
3. Not found: `[DAAP] Skill not found: /home/user/my-skill`

### `/skill create` — Interactive Wizard

```
[DAAP] Skill creator — press Ctrl+C to cancel

Name: my-skill
Description (one line): Does X when Y
Targets [all/master/subagent, default=all]:
Save to dir [~/.daap/skills/my-skill]:

--- Skill body (blank line + . to finish) ---
> When user asks for X...
> Do Y.
> .

--- Preview ---
# my-skill

Does X when Y

When user asks for X...
Do Y.

Write to ~/.daap/skills/my-skill/SKILL.md? [y/N]: y
[DAAP] Skill 'my-skill' created and registered.
```

Flow:
1. Prompt: name (required, no spaces → kebab-case enforced)
2. Prompt: description (one line, required)
3. Prompt: targets (default `all`)
4. Prompt: save dir (default `~/.daap/skills/<name>`)
5. Prompt: skill body (multi-line, terminated by `.` on blank line)
6. Show preview of generated `SKILL.md`
7. Confirm → create dir + write `SKILL.md` → call `add_skill()` → registered + persisted

### `/skills` (existing, unchanged)

Lists all registered skills with targets. No change.

---

## Section 5: Master Agent Tool — `register_skill`

New tool added to master toolkit in `daap/master/tools.py`:

```python
def register_skill(directory: str, targets: str = "all") -> str:
    """
    Register a skill directory for this session.

    Validates the directory contains a valid SKILL.md, registers the skill
    immediately on the master toolkit, and persists to skills.json.

    Args:
        directory: Absolute or relative path to the skill directory.
        targets: Which agents get this skill — "master", "subagent", or "all".

    Returns:
        Confirmation message on success, error message on failure.
    """
```

### Agent prompt addition (`daap/master/prompts.py`)

Add to system prompt:

> If the user mentions a file path that looks like a skill directory, call `register_skill` with that path to wire it up immediately.

### Behavior

- Success → `"Skill 'my-skill' registered [all]."`
- `SkillValidationError` → agent surfaces exact error to user
- Already registered → `"Skill 'my-skill' already registered."`

---

## Data Flow

```
Session start
  → SkillManager._load_specs(skills.json)      # existing config
  → SkillManager._discover_skills()             # auto-discovery
  → create_master_toolkit()
  → manager.bind_toolkit(toolkit, "master")
  → agent starts

Mid-session: user says "add skill at /path/foo"
  → agent calls register_skill("/path/foo")
  → SkillManager.add_skill() → validate → register live → persist

Mid-session: user types /skill add /path/foo
  → chat.py parses command
  → SkillManager.add_skill() → validate → register live → persist

Mid-session: user types /skill create
  → chat.py wizard → write SKILL.md → SkillManager.add_skill()
```

---

## Error Handling

| Scenario | Behavior |
|---|---|
| `skills.json` missing | Treat as empty, no error |
| `skills.json` malformed JSON | Log warning, treat as empty |
| Skill dir missing at startup | Log `SkillValidationError`, skip |
| Skill dir missing at runtime add | Raise, surface to user |
| AgentScope `register_agent_skill` raises | Log + surface, mark not registered |
| `skills.json` write fails | Log warning, skill still registered in-session |

---

## Testing

- `test_skills_manager.py` — add tests for: `SkillValidationError` cases, `add_skill`, `remove_skill`, `_discover_skills`, atomic write
- `test_skills_cli.py` — wizard flow, `/skill add` parse, `/skill remove` parse
- All existing tests remain green
