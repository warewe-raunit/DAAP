"""
DAAP Server Entry Point.

Run with:
  python -m daap.main
"""

import os
# Must be set before pydantic/uvicorn import to avoid jiter Rust DLL (blocked by App Control)
os.environ.setdefault("PYDANTIC_PURE_PYTHON", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import warnings

# uvicorn's websockets backend uses the legacy 2-arg ws_handler form deprecated
# in websockets 12+. Module filter omitted intentionally — stack attribution
# may land on uvicorn code, not websockets, depending on call depth.
warnings.filterwarnings(
    "ignore",
    message="remove second argument of ws_handler",
    category=DeprecationWarning,
)
# qdrant_client emits this when it cannot reach the server during version
# negotiation. Memory already falls back gracefully; suppress the noise.
warnings.filterwarnings(
    "ignore",
    message="Failed to obtain server version",
    category=UserWarning,
)

import uvicorn

from daap.env import load_project_env

load_project_env()
from daap.api.routes import app  # noqa: F401 — re-exported for uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "daap.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ws="wsproto",  # avoids websockets legacy handler deprecation warning
    )
