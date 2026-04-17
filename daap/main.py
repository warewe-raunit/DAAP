"""
DAAP Server Entry Point.

Run with:
  uvicorn daap.main:app --reload --port 8000
"""

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
    )
