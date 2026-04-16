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
    """Return ~/.daap directory path (overridable in tests via monkeypatch)."""
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

    - If ~/.daap/user.json exists: return saved user_id.
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
