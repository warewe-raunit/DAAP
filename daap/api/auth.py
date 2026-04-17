"""
DAAP API Key Authentication.

Set DAAP_API_KEY env var to enable auth.
If unset: open/dev mode (no enforcement).

Usage:
  HTTP:  X-API-Key: <key>
  WS:    ws://host/ws/{session_id}?token=<key>
"""
import os
import secrets

from fastapi import Header, HTTPException, Query

# Read once at import time — restart server after changing.
_API_KEY: str | None = os.environ.get("DAAP_API_KEY") or None


def _validate_key(key: str | None) -> None:
    """Raise 401 if key is wrong. No-op if DAAP_API_KEY not configured."""
    if not _API_KEY:
        return  # dev/local mode
    if not key or not secrets.compare_digest(key, _API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """FastAPI dependency — validate X-API-Key header."""
    _validate_key(x_api_key)


def validate_ws_token(token: str | None) -> None:
    """Validate WS query-param token before accepting the socket."""
    _validate_key(token)
