"""Tests for the register_skill master-agent tool."""


class _FakeToolkit:
    def __init__(self):
        self.registered: list[str] = []

    def register_agent_skill(self, skill_dir: str):
        self.registered.append(skill_dir)


def test_register_skill_success(tmp_path, monkeypatch):
    from daap.skills import manager as mgr_mod
    from daap.master.tools import register_skill

    skill_dir = tmp_path / "test-tool-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-tool-skill\ndescription: registered via agent\n---\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mgr_mod.SkillManager, "_DISCOVERY_PATHS", [])
    toolkit = _FakeToolkit()
    test_manager = mgr_mod.SkillManager(config_path=str(tmp_path / "skills.json"))
    test_manager.bind_toolkit(toolkit, "master")
    monkeypatch.setattr(mgr_mod, "_SKILL_MANAGER", test_manager)

    result = register_skill(str(skill_dir))

    assert "test-tool-skill" in result
    assert "registered" in result
    assert str(skill_dir) in toolkit.registered


def test_register_skill_invalid_dir(tmp_path, monkeypatch):
    from daap.skills import manager as mgr_mod
    from daap.master.tools import register_skill

    monkeypatch.setattr(mgr_mod.SkillManager, "_DISCOVERY_PATHS", [])
    test_manager = mgr_mod.SkillManager(config_path=str(tmp_path / "skills.json"))
    monkeypatch.setattr(mgr_mod, "_SKILL_MANAGER", test_manager)

    result = register_skill(str(tmp_path / "ghost"))

    assert "Failed" in result
    assert "directory not found" in result


def test_register_skill_already_registered(tmp_path, monkeypatch):
    from daap.skills import manager as mgr_mod
    from daap.master.tools import register_skill

    skill_dir = tmp_path / "dup-tool-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: dup-tool-skill\ndescription: duplicate\n---\n", encoding="utf-8"
    )

    monkeypatch.setattr(mgr_mod.SkillManager, "_DISCOVERY_PATHS", [])
    toolkit = _FakeToolkit()
    test_manager = mgr_mod.SkillManager(config_path=str(tmp_path / "skills.json"))
    test_manager.bind_toolkit(toolkit, "master")
    monkeypatch.setattr(mgr_mod, "_SKILL_MANAGER", test_manager)

    register_skill(str(skill_dir))
    result = register_skill(str(skill_dir))

    assert "already registered" in result
    assert toolkit.registered.count(str(skill_dir)) == 1
