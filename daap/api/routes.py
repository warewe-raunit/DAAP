"""
DAAP FastAPI Routes — HTTP and WebSocket endpoints.

REST:
  GET  /health
  POST /session
    GET  /session/{session_id}/config
  GET  /sessions
  DEL  /session/{session_id}
  GET  /topology/{session_id}
  POST /rate
  GET  /runs/{session_id}

WebSocket:
  WS   /ws/{session_id}
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from daap.api.sessions import SessionManager, create_session_scoped_toolkit
from daap.api.topology_routes import (
    router as topology_router,
    set_session_manager as set_topology_session_manager,
    set_store as set_topology_store,
)
from daap.api.ws_handler import handle_websocket
from daap.feedback.store import FeedbackStore
from daap.master.agent import create_master_agent_with_toolkit
from daap.topology.store import TopologyStore
from daap.tools.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

# Global singletons
session_manager = SessionManager()
feedback_store = FeedbackStore()
topology_store = TopologyStore()

set_topology_store(topology_store)
set_topology_session_manager(session_manager)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPERATOR_PROVIDER = "openrouter"
DEFAULT_OPERATOR_KEY_ENV = "OPENROUTER_API_KEY"

# Memory — optional. Disabled gracefully if credentials are missing.
_daap_memory = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: purge expired, soft-deleted topologies.
    try:
        purged = topology_store.purge_expired()
        if purged:
            logger.info("Startup: purged %d expired topologies", purged)
    except Exception as exc:
        logger.warning("Startup purge failed (non-fatal): %s", exc)
    yield


app = FastAPI(
    title="DAAP API",
    description="Dynamic Agent Architecture Protocol — API Layer",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Phase 3: restrict to known origins
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(topology_router)

def _get_memory():
    """Lazy-init DaapMemory. Returns None if setup fails."""
    global _daap_memory
    if _daap_memory is not None:
        return _daap_memory
    try:
        from daap.memory.client import DaapMemory
        _daap_memory = DaapMemory()
        return _daap_memory
    except Exception as exc:
        logger.warning("Memory disabled: %s", exc)
        return None


def _build_master_operator_config(master_model: str | None) -> dict | None:
    """Build operator_config for the master agent from user model selection."""
    if not master_model or not master_model.strip():
        return None

    return {
        "provider": DEFAULT_OPERATOR_PROVIDER,
        "base_url": OPENROUTER_BASE_URL,
        "api_key_env": DEFAULT_OPERATOR_KEY_ENV,
        "model_map": {"smart": master_model.strip()},
    }


def _build_subagent_operator_config(
    subagent_model: str | None,
    subagent_fast_model: str | None,
    subagent_smart_model: str | None,
    subagent_powerful_model: str | None,
) -> dict | None:
    """Build default operator/model map for topology subagents."""
    model_map: dict[str, str] = {}

    if subagent_model and subagent_model.strip():
        selected = subagent_model.strip()
        model_map = {
            "fast": selected,
            "smart": selected,
            "powerful": selected,
        }

    if subagent_fast_model and subagent_fast_model.strip():
        model_map["fast"] = subagent_fast_model.strip()
    if subagent_smart_model and subagent_smart_model.strip():
        model_map["smart"] = subagent_smart_model.strip()
    if subagent_powerful_model and subagent_powerful_model.strip():
        model_map["powerful"] = subagent_powerful_model.strip()

    if not model_map:
        return None

    return {
        "provider": DEFAULT_OPERATOR_PROVIDER,
        "base_url": OPENROUTER_BASE_URL,
        "api_key_env": DEFAULT_OPERATOR_KEY_ENV,
        "model_map": model_map,
    }


# ---------------------------------------------------------------------------
# Health + model listing
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def list_models():
    """Return available model presets and how to select them via POST /session."""
    return {
        "provider": "openrouter",
        "presets": {
            "master_agent": {
                "default": "google/gemini-2.0-flash-001",
                "options": [
                    {"id": "google/gemini-2.0-flash-001",          "cost": "$$0.10/$0.40 per 1M",  "note": "default — fast + cheap"},
                    {"id": "google/gemini-flash-1.5",              "cost": "$0.075/$0.30 per 1M", "note": "cheaper Gemini"},
                    {"id": "openrouter/free",                      "cost": "$0",                  "note": "free, random model"},
                    {"id": "openai/gpt-4o-mini",                   "cost": "$0.15/$0.60 per 1M",  "note": "OpenAI cheap"},
                    {"id": "qwen/qwen3.5-plus-02-15",             "cost": "$0.26/$1.56 per 1M",  "note": "big context"},
                    {"id": "anthropic/claude-sonnet-4-6",          "cost": "$3/$15 per 1M",       "note": "high quality"},
                    {"id": "anthropic/claude-opus-4-6",            "cost": "$15/$75 per 1M",      "note": "best quality"},
                ],
            },
            "subagents": {
                "tiers": ["fast", "smart", "powerful"],
                "defaults": {
                    "fast":     "google/gemini-2.0-flash-001",
                    "smart":    "google/gemini-2.0-flash-001",
                    "powerful": "google/gemini-2.0-flash-001",
                },
            },
        },
        "usage": {
            "master_model":          "POST /session?master_model=anthropic/claude-opus-4-6",
            "subagent_model":        "POST /session?subagent_model=anthropic/claude-haiku-4-5-20251001  (all tiers)",
            "subagent_fast_model":   "POST /session?subagent_fast_model=openai/gpt-4o-mini",
            "subagent_smart_model":  "POST /session?subagent_smart_model=openai/gpt-4o",
            "subagent_powerful_model": "POST /session?subagent_powerful_model=anthropic/claude-opus-4-6",
        },
    }


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@app.post("/session")
async def create_session(
    user_id: str = "default",
    master_model: str | None = None,
    subagent_model: str | None = None,
    subagent_fast_model: str | None = None,
    subagent_smart_model: str | None = None,
    subagent_powerful_model: str | None = None,
):
    """Create a new session and optionally set master/subagent model selection."""
    session = session_manager.create_session()
    session.user_id = user_id
    session.master_operator_config = _build_master_operator_config(master_model)
    session.subagent_operator_config = _build_subagent_operator_config(
        subagent_model,
        subagent_fast_model,
        subagent_smart_model,
        subagent_powerful_model,
    )

    # Load user context from memory (None for first-time users — graceful)
    user_context = None
    memory = _get_memory()
    if memory:
        try:
            from daap.memory.reader import load_user_context_for_master
            user_context = load_user_context_for_master(memory, user_id)
        except Exception as exc:
            logger.warning("Failed to load user context: %s", exc)

    session.token_tracker = TokenTracker()
    toolkit = create_session_scoped_toolkit(session, topology_store=topology_store)
    session.master_agent = create_master_agent_with_toolkit(
        toolkit,
        user_context=user_context,
        operator_config=session.master_operator_config,
        tracker=session.token_tracker,
    )
    return {"session_id": session.session_id}


@app.get("/session/{session_id}/config")
async def get_session_config(session_id: str):
    """Return the selected master/subagent model configuration for a session."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "master_operator_config": session.master_operator_config,
        "subagent_operator_config": session.subagent_operator_config,
    }


@app.get("/sessions")
async def list_sessions():
    return {"sessions": session_manager.list_sessions()}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    session_manager.delete_session(session_id)
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Topology inspection
# ---------------------------------------------------------------------------

@app.get("/topology/{session_id}")
async def get_topology(session_id: str):
    """Return the pending topology for a session (if any)."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "topology": session.pending_topology,
        "estimate": session.pending_estimate,
    }


# ---------------------------------------------------------------------------
# Feedback / rating
# ---------------------------------------------------------------------------

class RatingRequest(BaseModel):
    session_id: str
    rating: int     # 1-5
    comment: str = ""


@app.post("/rate")
async def rate_run(req: RatingRequest):
    """Rate the output of a completed execution run."""
    session = session_manager.get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.execution_result is None:
        raise HTTPException(status_code=400, detail="No execution result to rate")

    # SQLite feedback
    feedback_store.store_rating(
        session_id=req.session_id,
        rating=req.rating,
        comment=req.comment,
        topology_json=session.pending_topology,
        execution_result=session.execution_result,
    )

    # Mem0 feedback (optional)
    memory = _get_memory()
    if memory:
        try:
            from daap.memory.writer import write_user_feedback
            write_user_feedback(
                memory,
                user_id=session.user_id,
                feedback_text=(
                    f"User rated run {req.rating}/5."
                    + (f" Comment: {req.comment}" if req.comment else "")
                ),
            )
        except Exception as exc:
            logger.warning("Failed to write feedback to memory: %s", exc)

    return {"status": "rated", "rating": req.rating}


@app.get("/runs/{session_id}")
async def get_run_history(session_id: str):
    """Get execution history for a session."""
    return {"runs": feedback_store.get_runs_for_session(session_id)}


# ---------------------------------------------------------------------------
# WebSocket conversation
# ---------------------------------------------------------------------------

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time conversation with the master agent.

    Client must first POST /session to get a session_id, then connect here.
    """
    session = session_manager.get_session(session_id)
    if session is None:
        await websocket.close(code=4004, reason="Session not found")
        return
    if session.master_agent is None:
        await websocket.close(code=4005, reason="Session not initialised")
        return

    await handle_websocket(
        websocket,
        session,
        daap_memory=_get_memory(),
        topology_store=topology_store,
    )
