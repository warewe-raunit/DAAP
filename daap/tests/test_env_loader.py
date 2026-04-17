"""Tests for project .env loading helpers."""

from pathlib import Path


def test_load_project_env_fallback_loads_openrouter_api_key(tmp_path, monkeypatch):
    from daap import env as env_mod

    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=test-from-file\n", encoding="utf-8")

    monkeypatch.setattr(env_mod, "_dotenv_load", None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER", raising=False)

    loaded = env_mod.load_project_env(env_file)

    assert loaded is True
    assert env_mod.os.environ["OPENROUTER_API_KEY"] == "test-from-file"


def test_load_project_env_maps_legacy_openrouter_key(tmp_path, monkeypatch):
    from daap import env as env_mod

    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER=legacy-key\n", encoding="utf-8")

    monkeypatch.setattr(env_mod, "_dotenv_load", None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER", raising=False)

    loaded = env_mod.load_project_env(env_file)

    assert loaded is True
    assert env_mod.os.environ["OPENROUTER_API_KEY"] == "legacy-key"


def test_load_project_env_overrides_existing_api_key_with_file_value(tmp_path, monkeypatch):
    from daap import env as env_mod

    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=file-value\n", encoding="utf-8")

    monkeypatch.setattr(env_mod, "_dotenv_load", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "existing")
    monkeypatch.delenv("OPENROUTER", raising=False)

    env_mod.load_project_env(Path(env_file))

    assert env_mod.os.environ["OPENROUTER_API_KEY"] == "file-value"


def test_load_project_env_overrides_empty_api_key_from_file(tmp_path, monkeypatch):
    from daap import env as env_mod

    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=file-value\n", encoding="utf-8")

    monkeypatch.setattr(env_mod, "_dotenv_load", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    env_mod.load_project_env(Path(env_file))

    assert env_mod.os.environ["OPENROUTER_API_KEY"] == "file-value"


def test_load_project_env_uses_dotenv_override_true(tmp_path, monkeypatch):
    from daap import env as env_mod

    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=file-value\n", encoding="utf-8")

    seen: dict[str, object] = {}

    def _fake_dotenv(path, override=False):
        seen["path"] = path
        seen["override"] = override
        return True

    monkeypatch.setattr(env_mod, "_dotenv_load", _fake_dotenv)

    env_mod.load_project_env(Path(env_file))

    assert seen["override"] is True
