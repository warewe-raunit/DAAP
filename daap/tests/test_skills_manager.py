"""Tests for DAAP Agent Skill manager and toolkit integration."""

import json

import pytest


@pytest.fixture(autouse=True)
def _disable_default_discovery(monkeypatch):
    from daap.skills.manager import SkillManager

    monkeypatch.setattr(SkillManager, "_DISCOVERY_PATHS", [])


class _FakeToolkit:
    def __init__(self):
        self.registered: list[str] = []
        self.unregistered: list[str] = []

    def register_agent_skill(self, skill_dir: str):
        self.registered.append(skill_dir)

    def unregister_agent_skill(self, skill_dir: str):
        self.unregistered.append(skill_dir)


def _make_skill_dir(tmp_path, name="my-skill", content=None):
    d = tmp_path / name
    d.mkdir()
    if content is not None:
        (d / "SKILL.md").write_text(content, encoding="utf-8")
    return d


def test_skill_manager_registers_targeted_skills(tmp_path):
    from daap.skills.manager import SkillManager

    master_only = _make_skill_dir(
        tmp_path,
        "master-only",
        "---\nname: master-only\ndescription: master\n---\n",
    )
    sub_only = _make_skill_dir(
        tmp_path,
        "sub-only",
        "---\nname: sub-only\ndescription: subagent\n---\n",
    )
    both = _make_skill_dir(
        tmp_path,
        "both",
        "---\nname: both\ndescription: all targets\n---\n",
    )
    default_all = _make_skill_dir(
        tmp_path,
        "default-all",
        "---\nname: default-all\ndescription: defaults to all\n---\n",
    )

    config_path = tmp_path / "skills.json"
    config_path.write_text(
        json.dumps(
            {
                "skills": [
                    {"dir": str(master_only), "targets": ["master"]},
                    {"dir": str(sub_only), "targets": ["subagent"]},
                    {"dir": str(both), "targets": ["all"]},
                    str(default_all),
                ]
            }
        ),
        encoding="utf-8",
    )

    manager = SkillManager(config_path=str(config_path))
    master_toolkit = _FakeToolkit()
    subagent_toolkit = _FakeToolkit()

    manager.register_toolkit_skills(master_toolkit, target="master")
    manager.register_toolkit_skills(subagent_toolkit, target="subagent")

    assert str(master_only) in master_toolkit.registered
    assert str(sub_only) not in master_toolkit.registered
    assert str(both) in master_toolkit.registered
    assert str(default_all) in master_toolkit.registered

    assert str(master_only) not in subagent_toolkit.registered
    assert str(sub_only) in subagent_toolkit.registered
    assert str(both) in subagent_toolkit.registered
    assert str(default_all) in subagent_toolkit.registered


def test_skill_manager_returns_empty_for_invalid_config(tmp_path):
    from daap.skills.manager import SkillManager

    config_path = tmp_path / "skills.json"
    config_path.write_text(json.dumps({"unexpected": []}), encoding="utf-8")

    manager = SkillManager(config_path=str(config_path))
    toolkit = _FakeToolkit()
    loaded = manager.register_toolkit_skills(toolkit, target="master")

    assert loaded == []
    assert toolkit.registered == []


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


def test_discover_skills_finds_valid_dirs(tmp_path):
    from daap.skills.manager import SkillManager

    for name, content in [
        ("skill-a", "---\nname: skill-a\ndescription: does a\n---\n"),
        ("skill-b", "# skill-b\n\nDoes b.\n"),
    ]:
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(content, encoding="utf-8")

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

    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: does things\n---\n", encoding="utf-8"
    )

    monkeypatch.setattr(SkillManager, "_DISCOVERY_PATHS", [str(tmp_path)])
    manager = SkillManager(config_path=str(tmp_path / "nonexistent.json"))
    dirs = manager.list_skill_dirs("master")
    assert str(skill_dir) in dirs


def test_add_skill_registers_and_persists(tmp_path):
    from daap.skills.manager import SkillManager

    skill_dir = tmp_path / "new-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: new-skill\ndescription: does new things\n---\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "skills.json"

    manager = SkillManager(config_path=str(config_path))
    toolkit = _FakeToolkit()
    manager.bind_toolkit(toolkit, target="master")

    name, added = manager.add_skill(str(skill_dir), targets="master", persist=True)

    assert name == "new-skill"
    assert added is True
    assert str(skill_dir) in toolkit.registered
    assert str(skill_dir) in manager.list_skill_dirs("master")

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

    _, added_first = manager.add_skill(str(skill_dir))
    _, added_second = manager.add_skill(str(skill_dir))

    assert added_first is True
    assert added_second is False
    assert toolkit.registered.count(str(skill_dir)) == 1


def test_remove_skill_removes_from_config(tmp_path):
    from daap.skills.manager import SkillManager

    skill_dir = tmp_path / "rm-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: rm-skill\ndescription: to be removed\n---\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "skills.json"
    manager = SkillManager(config_path=str(config_path))
    toolkit = _FakeToolkit()
    manager.bind_toolkit(toolkit, target="master")

    manager.add_skill(str(skill_dir), persist=True)
    assert str(skill_dir) in manager.list_skill_dirs("master")

    removed_name = manager.remove_skill(str(skill_dir), persist=True)
    assert removed_name == "rm-skill"
    assert str(skill_dir) not in manager.list_skill_dirs("master")
    assert str(skill_dir) in toolkit.unregistered

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    dirs = [e["dir"] if isinstance(e, dict) else e for e in saved.get("skills", saved)]
    assert str(skill_dir) not in dirs


def test_remove_skill_not_found_raises(tmp_path):
    from daap.skills.manager import SkillManager

    manager = SkillManager(config_path=str(tmp_path / "skills.json"))
    with pytest.raises(KeyError):
        manager.remove_skill(str(tmp_path / "ghost-skill"))


def test_atomic_write_does_not_corrupt_on_failure(tmp_path, monkeypatch):
    from daap.skills import manager as mgr_mod

    config_path = tmp_path / "skills.json"
    original = {"skills": [{"dir": "/safe/skill", "targets": ["all"]}]}
    config_path.write_text(json.dumps(original), encoding="utf-8")
    manager = mgr_mod.SkillManager(config_path=str(config_path))

    skill_dir = tmp_path / "new-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: new-skill\ndescription: does new things\n---\n",
        encoding="utf-8",
    )
    manager.add_skill(str(skill_dir), persist=False)

    def _fail_replace(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr(mgr_mod.os, "replace", _fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        manager._write_config()

    result = json.loads(config_path.read_text(encoding="utf-8"))
    assert result == original
