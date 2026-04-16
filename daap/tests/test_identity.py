"""Tests for daap.identity — local user persistence."""
import json


def test_load_local_user_returns_none_when_file_missing(tmp_path, monkeypatch):
    import daap.identity as identity
    monkeypatch.setattr(identity, "_daap_dir", lambda: tmp_path / ".daap")
    assert identity.load_local_user() is None


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    import daap.identity as identity
    monkeypatch.setattr(identity, "_daap_dir", lambda: tmp_path / ".daap")
    identity.save_local_user("alice")
    assert identity.load_local_user() == "alice"


def test_save_creates_directory(tmp_path, monkeypatch):
    import daap.identity as identity
    daap_dir = tmp_path / ".daap"
    monkeypatch.setattr(identity, "_daap_dir", lambda: daap_dir)
    assert not daap_dir.exists()
    identity.save_local_user("bob")
    assert daap_dir.exists()
    assert (daap_dir / "user.json").exists()


def test_load_returns_none_on_corrupt_file(tmp_path, monkeypatch):
    import daap.identity as identity
    daap_dir = tmp_path / ".daap"
    daap_dir.mkdir()
    (daap_dir / "user.json").write_text("NOT JSON")
    monkeypatch.setattr(identity, "_daap_dir", lambda: daap_dir)
    assert identity.load_local_user() is None


def test_load_returns_none_on_empty_user_id(tmp_path, monkeypatch):
    import daap.identity as identity
    daap_dir = tmp_path / ".daap"
    daap_dir.mkdir()
    (daap_dir / "user.json").write_text(json.dumps({"user_id": ""}))
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


def test_sanitize_strips_special_chars():
    from daap.identity import _sanitize
    assert _sanitize("user@example.com") == "userexamplecom"
