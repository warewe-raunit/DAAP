"""Environment loading helpers for DAAP entrypoints."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv as _dotenv_load
except ImportError:
    _dotenv_load = None


def _load_env_file(path: Path) -> bool:
    """Load KEY=VALUE entries from a .env file without third-party deps."""
    if not path.exists() or not path.is_file():
        return False

    loaded_any = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ[key] = value
        loaded_any = True

    return loaded_any


def _alias_legacy_openrouter_key() -> bool:
    """Map legacy OPENROUTER key name to OPENROUTER_API_KEY when needed."""
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return False

    legacy = os.environ.get("OPENROUTER", "").strip()
    if not legacy:
        return False

    os.environ["OPENROUTER_API_KEY"] = legacy
    return True


def load_project_env(dotenv_path: str | Path | None = None) -> bool:
    """
    Load project .env values into process env.

    Prefers python-dotenv when available, with a built-in parser fallback.
    Returns True when any relevant env values were loaded or mapped.
    """
    path = Path(dotenv_path) if dotenv_path is not None else (
        Path(__file__).resolve().parent.parent / ".env"
    )

    loaded = False
    if _dotenv_load is not None:
        # override=True ensures .env replaces empty shell exports like:
        #   $env:OPENROUTER_API_KEY=""
        loaded = bool(_dotenv_load(path, override=True))
    else:
        loaded = _load_env_file(path)

    aliased = _alias_legacy_openrouter_key()
    return loaded or aliased
