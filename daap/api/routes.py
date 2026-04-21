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

from fastapi import Depends, FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from daap.api.auth import require_api_key, validate_ws_token
from daap.api.sessions import SessionManager, SessionStore, create_session_scoped_toolkit
from daap.api.topology_routes import (
    router as topology_router,
    set_session_manager as set_topology_session_manager,
    set_store as set_topology_store,
)
from daap.api.ws_handler import handle_websocket
from daap.feedback.store import FeedbackStore
from daap.master.agent import create_master_agent_with_toolkit
from daap.master.runtime import build_master_runtime_snapshot
from daap.optimizer.store import BanditStore
from daap.topology.store import TopologyStore
from daap.tools.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chat UI — single-file HTML served at GET /
# ---------------------------------------------------------------------------

_CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DAAP Chat</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    height: 100dvh;
    display: flex;
    flex-direction: column;
  }
  header {
    padding: 12px 20px;
    border-bottom: 1px solid #1e2430;
    display: flex;
    align-items: center;
    gap: 10px;
    background: #141921;
  }
  header h1 { font-size: 1rem; font-weight: 600; color: #94a3b8; letter-spacing: 0.05em; }
  #status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #ef4444; flex-shrink: 0;
    transition: background 0.3s;
  }
  #status-dot.connected { background: #22c55e; }
  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .msg {
    max-width: 720px;
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.55;
    font-size: 0.9rem;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .msg.user { background: #1e40af; align-self: flex-end; color: #e0e7ff; border-bottom-right-radius: 3px; }
  .msg.agent { background: #1e2430; align-self: flex-start; border-bottom-left-radius: 3px; }
  .msg.error { background: #450a0a; color: #fca5a5; align-self: flex-start; border-left: 3px solid #ef4444; }
  .msg.system { background: transparent; color: #64748b; align-self: center; font-size: 0.8rem; font-style: italic; }

  /* Plan card */
  .plan-card {
    background: #1a2235;
    border: 1px solid #2d3f5e;
    border-radius: 12px;
    padding: 16px;
    align-self: flex-start;
    max-width: 560px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .plan-card h3 { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }
  .plan-card .summary { font-size: 0.9rem; color: #e2e8f0; }
  .plan-meta { display: flex; gap: 16px; font-size: 0.8rem; color: #64748b; }
  .plan-actions { display: flex; gap: 8px; }
  .btn {
    padding: 7px 16px; border: none; border-radius: 7px;
    font-size: 0.85rem; font-weight: 500; cursor: pointer;
    transition: opacity 0.15s;
  }
  .btn:hover { opacity: 0.85; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-approve { background: #16a34a; color: #fff; }
  .btn-cheaper { background: #1d4ed8; color: #fff; }
  .btn-cancel  { background: #374151; color: #d1d5db; }

  /* Questions form */
  .questions-card {
    background: #1a2235;
    border: 1px solid #2d3f5e;
    border-radius: 12px;
    padding: 16px;
    align-self: flex-start;
    max-width: 560px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .questions-card h3 { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }
  .q-item { display: flex; flex-direction: column; gap: 5px; }
  .q-item label { font-size: 0.85rem; color: #cbd5e1; }
  .q-item input, .q-item select {
    background: #0f1117; border: 1px solid #374151; border-radius: 6px;
    color: #e2e8f0; padding: 7px 10px; font-size: 0.875rem;
    outline: none;
  }
  .q-item input:focus, .q-item select:focus { border-color: #3b82f6; }

  /* Executing indicator */
  .executing-row {
    display: flex; align-items: center; gap: 10px;
    align-self: flex-start; color: #94a3b8; font-size: 0.85rem;
  }
  .spinner {
    width: 16px; height: 16px;
    border: 2px solid #374151;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Result card */
  .result-card {
    background: #052e16;
    border: 1px solid #166534;
    border-radius: 12px;
    padding: 16px;
    align-self: flex-start;
    max-width: 720px;
  }
  .result-card h3 { font-size: 0.85rem; color: #4ade80; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.08em; }
  .result-card pre {
    font-family: "SF Mono", "Fira Code", monospace;
    font-size: 0.82rem;
    white-space: pre-wrap;
    word-break: break-word;
    color: #bbf7d0;
  }
  .result-meta { font-size: 0.78rem; color: #4ade80; margin-top: 8px; opacity: 0.7; }

  /* Bottom bar */
  #bottom {
    padding: 14px 20px;
    border-top: 1px solid #1e2430;
    background: #141921;
    display: flex;
    gap: 10px;
  }
  #input {
    flex: 1;
    background: #1e2430;
    border: 1px solid #374151;
    border-radius: 10px;
    color: #e2e8f0;
    padding: 10px 14px;
    font-size: 0.9rem;
    resize: none;
    outline: none;
    max-height: 160px;
    line-height: 1.4;
  }
  #input:focus { border-color: #3b82f6; }
  #send-btn {
    padding: 10px 18px;
    background: #3b82f6;
    color: #fff;
    border: none;
    border-radius: 10px;
    font-size: 0.9rem;
    font-weight: 500;
    cursor: pointer;
    align-self: flex-end;
    transition: background 0.15s;
  }
  #send-btn:hover { background: #2563eb; }
  #send-btn:disabled { background: #374151; cursor: not-allowed; }

  /* Setup overlay */
  #setup-overlay {
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.7);
    display: flex; align-items: center; justify-content: center;
    z-index: 100;
  }
  #setup-box {
    background: #141921;
    border: 1px solid #2d3f5e;
    border-radius: 16px;
    padding: 32px;
    width: 360px;
    display: flex;
    flex-direction: column;
    gap: 18px;
  }
  #setup-box h2 { font-size: 1.1rem; color: #e2e8f0; }
  #setup-box p { font-size: 0.85rem; color: #64748b; }
  .field { display: flex; flex-direction: column; gap: 6px; }
  .field label { font-size: 0.85rem; color: #94a3b8; }
  .field input {
    background: #0f1117; border: 1px solid #374151; border-radius: 8px;
    color: #e2e8f0; padding: 9px 12px; font-size: 0.9rem; outline: none;
  }
  .field input:focus { border-color: #3b82f6; }
  #start-btn {
    padding: 10px; background: #3b82f6; color: #fff;
    border: none; border-radius: 8px; font-size: 0.95rem;
    font-weight: 600; cursor: pointer;
  }
  #start-btn:hover { background: #2563eb; }
  #setup-error { color: #fca5a5; font-size: 0.82rem; display: none; }
</style>
</head>
<body>

<div id="setup-overlay">
  <div id="setup-box">
    <h2>DAAP Chat</h2>
    <p>Dynamic Agent Automation Platform</p>
    <div class="field">
      <label for="uid-input">Your name / user ID</label>
      <input id="uid-input" type="text" placeholder="e.g. alice" autocomplete="off">
    </div>
    <div class="field" id="key-field" style="display:none">
      <label for="key-input">API Key</label>
      <input id="key-input" type="password" placeholder="DAAP_API_KEY">
    </div>
    <div id="setup-error"></div>
    <button id="start-btn">Start chatting</button>
  </div>
</div>

<header>
  <div id="status-dot"></div>
  <h1>DAAP</h1>
  <span id="session-label" style="font-size:0.75rem;color:#475569;margin-left:auto"></span>
</header>

<div id="messages"></div>

<div id="bottom">
  <textarea id="input" rows="1" placeholder="Describe a task…" disabled></textarea>
  <button id="send-btn" disabled>Send</button>
</div>

<script>
const AUTH_ENABLED = __AUTH_ENABLED__;
let ws = null;
let sessionId = null;
let apiKey = "";
let pendingPlanEl = null;

// ── Setup overlay ──────────────────────────────────────────────────────────
const overlay = document.getElementById("setup-overlay");
const keyField = document.getElementById("key-field");
if (AUTH_ENABLED) keyField.style.display = "";

document.getElementById("start-btn").addEventListener("click", async () => {
  const userId = document.getElementById("uid-input").value.trim();
  if (!userId) { showSetupError("Enter a user ID"); return; }
  apiKey = document.getElementById("key-input").value.trim();
  if (AUTH_ENABLED && !apiKey) { showSetupError("Enter the API key"); return; }
  await startSession(userId);
});

document.getElementById("uid-input").addEventListener("keydown", e => {
  if (e.key === "Enter") document.getElementById("start-btn").click();
});

function showSetupError(msg) {
  const el = document.getElementById("setup-error");
  el.textContent = msg;
  el.style.display = "";
}

// ── Session + WebSocket ────────────────────────────────────────────────────
async function startSession(userId) {
  const btn = document.getElementById("start-btn");
  btn.disabled = true;
  btn.textContent = "Connecting…";

  const headers = { "Content-Type": "application/json" };
  if (apiKey) headers["X-API-Key"] = apiKey;

  const r = await fetch(`/session?user_id=${encodeURIComponent(userId)}`, {
    method: "POST", headers,
  });

  if (!r.ok) {
    const j = await r.json().catch(() => ({}));
    showSetupError(j.detail || `Error ${r.status}`);
    btn.disabled = false; btn.textContent = "Start chatting";
    return;
  }

  const { session_id } = await r.json();
  sessionId = session_id;
  document.getElementById("session-label").textContent = `session: ${session_id.slice(0,8)}…`;
  overlay.style.display = "none";
  connectWS();
}

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}/ws/${sessionId}${apiKey ? "?token=" + encodeURIComponent(apiKey) : ""}`;
  ws = new WebSocket(url);

  ws.onopen = () => {
    document.getElementById("status-dot").classList.add("connected");
    document.getElementById("input").disabled = false;
    document.getElementById("send-btn").disabled = false;
    document.getElementById("input").focus();
    appendSystem("Connected. Describe a task to get started.");
  };

  ws.onclose = () => {
    document.getElementById("status-dot").classList.remove("connected");
    document.getElementById("input").disabled = true;
    document.getElementById("send-btn").disabled = true;
    appendSystem("Disconnected. Refresh to reconnect.");
  };

  ws.onerror = () => appendError("WebSocket error.");

  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    handleMessage(data);
  };
}

// ── Message handlers ───────────────────────────────────────────────────────
function handleMessage(data) {
  switch (data.type) {
    case "response":
      removePlanCard();
      appendAgent(data.content, data.usage);
      break;
    case "questions":
      appendQuestions(data.questions);
      break;
    case "plan":
      appendPlan(data);
      break;
    case "executing":
      appendExecuting(data.topology_id);
      break;
    case "result":
      appendResult(data);
      break;
    case "error":
      appendError(data.message);
      break;
  }
}

// ── Append helpers ─────────────────────────────────────────────────────────
function scrollBottom() {
  const m = document.getElementById("messages");
  m.scrollTop = m.scrollHeight;
}

function appendMsg(el) {
  document.getElementById("messages").appendChild(el);
  scrollBottom();
  return el;
}

function appendUser(text) {
  const d = document.createElement("div");
  d.className = "msg user";
  d.textContent = text;
  appendMsg(d);
}

function appendAgent(text, usage) {
  const d = document.createElement("div");
  d.className = "msg agent";
  d.textContent = text;
  if (usage && usage.total_tokens) {
    const m = document.createElement("div");
    m.style.cssText = "font-size:0.75rem;color:#475569;margin-top:6px;";
    m.textContent = `${usage.total_tokens} tokens · $${(usage.total_cost_usd||0).toFixed(4)}`;
    d.appendChild(m);
  }
  appendMsg(d);
}

function appendSystem(text) {
  const d = document.createElement("div");
  d.className = "msg system";
  d.textContent = text;
  appendMsg(d);
}

function appendError(text) {
  const d = document.createElement("div");
  d.className = "msg error";
  d.textContent = "Error: " + text;
  appendMsg(d);
}

function appendExecuting(topoId) {
  const d = document.createElement("div");
  d.className = "executing-row";
  d.innerHTML = `<div class="spinner"></div><span>Executing pipeline${topoId ? " · " + topoId.slice(0,8) + "…" : ""}…</span>`;
  appendMsg(d);
}

function removePlanCard() {
  if (pendingPlanEl) {
    pendingPlanEl.remove();
    pendingPlanEl = null;
  }
}

function appendPlan(data) {
  removePlanCard();
  const card = document.createElement("div");
  card.className = "plan-card";
  const cost = (data.cost_usd || 0).toFixed(4);
  const minCost = (data.min_cost_usd || 0).toFixed(4);
  const lat = (data.latency_seconds || 0).toFixed(1);
  card.innerHTML = `
    <h3>Proposed Plan</h3>
    <div class="summary">${esc(data.summary || "")}</div>
    <div class="plan-meta">
      <span>Est. cost: <b>$${cost}</b> (min $${minCost})</span>
      <span>Est. time: <b>${lat}s</b></span>
    </div>
    <div class="plan-actions">
      <button class="btn btn-approve">Approve</button>
      <button class="btn btn-cheaper">Make cheaper</button>
      <button class="btn btn-cancel">Cancel</button>
    </div>`;
  card.querySelector(".btn-approve").addEventListener("click", () => {
    ws.send(JSON.stringify({ type: "message", content: "approve" }));
    disablePlanBtns(card);
    appendUser("approve");
  });
  card.querySelector(".btn-cheaper").addEventListener("click", () => {
    ws.send(JSON.stringify({ type: "make_cheaper" }));
    disablePlanBtns(card);
    appendUser("make cheaper");
  });
  card.querySelector(".btn-cancel").addEventListener("click", () => {
    ws.send(JSON.stringify({ type: "cancel" }));
    disablePlanBtns(card);
    appendUser("cancel");
  });
  pendingPlanEl = appendMsg(card);
}

function disablePlanBtns(card) {
  card.querySelectorAll(".btn").forEach(b => b.disabled = true);
}

function appendQuestions(questions) {
  const card = document.createElement("div");
  card.className = "questions-card";
  card.innerHTML = `<h3>Questions</h3>`;
  const inputs = [];
  questions.forEach((q, i) => {
    const item = document.createElement("div");
    item.className = "q-item";
    const label = document.createElement("label");
    label.textContent = (i + 1) + ". " + (typeof q === "string" ? q : q.question || JSON.stringify(q));
    item.appendChild(label);
    let input;
    if (typeof q === "object" && q.options) {
      input = document.createElement("select");
      q.options.forEach(opt => {
        const o = document.createElement("option");
        o.value = opt; o.textContent = opt;
        input.appendChild(o);
      });
    } else {
      input = document.createElement("input");
      input.type = "text";
      input.placeholder = "Your answer…";
    }
    item.appendChild(input);
    inputs.push(input);
    card.appendChild(item);
  });
  const submitBtn = document.createElement("button");
  submitBtn.className = "btn btn-approve";
  submitBtn.textContent = "Submit answers";
  submitBtn.addEventListener("click", () => {
    const answers = inputs.map(inp => inp.value || inp.options?.[inp.selectedIndex]?.value || "");
    ws.send(JSON.stringify({ type: "answer", answers }));
    submitBtn.disabled = true;
    inputs.forEach(i => i.disabled = true);
  });
  card.appendChild(submitBtn);
  appendMsg(card);
}

function appendResult(data) {
  const card = document.createElement("div");
  card.className = "result-card";
  const output = typeof data.output === "string" ? data.output : JSON.stringify(data.output, null, 2);
  const cost = data.cost_usd != null ? `$${parseFloat(data.cost_usd).toFixed(4)}` : "";
  const lat = data.latency_seconds != null ? `${parseFloat(data.latency_seconds).toFixed(1)}s` : "";
  card.innerHTML = `<h3>Result</h3><pre>${esc(output)}</pre>`;
  if (cost || lat) {
    const meta = document.createElement("div");
    meta.className = "result-meta";
    meta.textContent = [cost, lat].filter(Boolean).join(" · ");
    card.appendChild(meta);
  }
  appendMsg(card);
}

function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Send ───────────────────────────────────────────────────────────────────
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");

function send() {
  const text = inputEl.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "message", content: text }));
  appendUser(text);
  inputEl.value = "";
  inputEl.style.height = "";
}

sendBtn.addEventListener("click", send);
inputEl.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
});
</script>
</body>
</html>"""

# Global singletons
_session_store = SessionStore()
session_manager = SessionManager(store=_session_store)
feedback_store = FeedbackStore()
topology_store = TopologyStore()
bandit_store = BanditStore()

set_topology_store(topology_store)
set_topology_session_manager(session_manager)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPERATOR_PROVIDER = "openrouter"
DEFAULT_OPERATOR_KEY_ENV = "OPENROUTER_API_KEY"

# Memory — optional. Disabled gracefully if credentials are missing.
_daap_memory = None
_rl_optimizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    mcp_manager = None

    # Startup: purge expired data by shared retention policy
    try:
        purged_sessions = _session_store.purge_expired()
        purged_feedback = feedback_store.purge_expired()
        purged_topologies = topology_store.purge_expired()
        purged_runs = topology_store.purge_old_runs()
        purged_optimizer = bandit_store.purge_expired()
        logger.info(
            (
                "Startup purge complete: sessions=%d feedback=%d topologies=%d "
                "topology_runs=%d optimizer_observations=%d"
            ),
            purged_sessions,
            purged_feedback,
            purged_topologies,
            purged_runs,
            purged_optimizer,
        )
    except Exception as exc:
        logger.warning("Startup purge failed (non-fatal): %s", exc)

    # Startup: MCP servers (optional)
    try:
        from daap.mcpx.manager import get_mcp_manager

        mcp_manager = get_mcp_manager()
        await mcp_manager.start_all()
        connected = mcp_manager.list_connected()
        if connected:
            logger.info("MCP servers connected: %s", ", ".join(connected))
    except Exception as exc:
        logger.warning("MCP disabled (non-fatal): %s", exc)

    try:
        yield
    finally:
        if mcp_manager is not None:
            try:
                await mcp_manager.stop_all()
            except Exception as exc:
                logger.warning("MCP shutdown failed (non-fatal): %s", exc)


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
    """Lazy-init DaapMemory. Returns None if setup fails or keys missing."""
    global _daap_memory
    if _daap_memory is not None:
        return _daap_memory
    try:
        from daap.memory.palace import DaapMemory
        mem = DaapMemory()
        if mem.available:
            _daap_memory = mem
            return _daap_memory
        return None
    except Exception as exc:
        logger.warning("Memory disabled: %s", exc)
        return None


def _get_optimizer():
    """Return True if LinTS optimizer is available, None otherwise."""
    global _rl_optimizer
    if _rl_optimizer is not None:
        return _rl_optimizer
    try:
        from daap.optimizer.integration import get_tier_recommendations  # noqa: F401
        _rl_optimizer = True
        return _rl_optimizer
    except Exception as exc:
        logger.warning("RL optimizer disabled: %s", exc)
        return None


def _build_master_operator_config(master_model: str | None) -> dict | None:
    """Build operator_config for the master agent from user model selection."""
    if not master_model or not master_model.strip():
        return None

    return {
        "provider": DEFAULT_OPERATOR_PROVIDER,
        "base_url": OPENROUTER_BASE_URL,
        "api_key_env": DEFAULT_OPERATOR_KEY_ENV,
        "model_map": {"powerful": master_model.strip()},
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


def _build_master_runtime_context(toolkit, memory, optimizer) -> dict:
    """Build runtime infra snapshot passed into the master system prompt."""
    return build_master_runtime_snapshot(
        toolkit,
        execution_mode="api-session",
        memory_enabled=bool(memory),
        optimizer_enabled=bool(optimizer),
        topology_store_enabled=True,
        feedback_store_enabled=True,
        session_store_enabled=True,
    )


# ---------------------------------------------------------------------------
# Health + model listing
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    import os
    auth_enabled = bool(os.environ.get("DAAP_API_KEY"))
    return HTMLResponse(_CHAT_HTML.replace("__AUTH_ENABLED__", "true" if auth_enabled else "false"))


@app.get("/models")
async def list_models():
    """Return available model presets and how to select them via POST /session."""
    from daap.spec.resolver import MODEL_REGISTRY
    return {
        "provider": "openrouter",
        "presets": {
            "master_agent": {
                "default": MODEL_REGISTRY["powerful"],
                "options": [
                    {"id": "google/gemini-2.5-flash",      "cost": "$0.30/$2.50 per 1M", "note": "default — powerful tier, thinking mode"},
                    {"id": "google/gemini-2.5-flash-lite", "cost": "$0.10/$0.40 per 1M", "note": "fast tier, cheapest"},
                    {"id": "deepseek/deepseek-v3.2",       "cost": "$0.26/$0.38 per 1M", "note": "smart tier, GPT-5 class"},
                    {"id": "openai/gpt-4o-mini",           "cost": "$0.15/$0.60 per 1M", "note": "OpenAI cheap option"},
                    {"id": "anthropic/claude-sonnet-4-6",  "cost": "$3/$15 per 1M",      "note": "high quality"},
                    {"id": "anthropic/claude-opus-4-6",    "cost": "$15/$75 per 1M",     "note": "best quality"},
                ],
            },
            "subagents": {
                "tiers": ["fast", "smart", "powerful"],
                "defaults": MODEL_REGISTRY,
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

@app.post("/session", dependencies=[Depends(require_api_key)])
async def create_session(
    user_id: str,
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
    optimizer = _get_optimizer()
    if memory:
        try:
            user_context = memory.format_for_master_prompt(user_id, "")
        except Exception as exc:
            logger.warning("Failed to load user context: %s", exc)

    session.token_tracker = TokenTracker()
    toolkit = create_session_scoped_toolkit(
        session,
        topology_store=topology_store,
        daap_memory=memory,
        rl_optimizer=optimizer,
        session_store=_session_store,
    )
    session.master_agent = create_master_agent_with_toolkit(
        toolkit,
        user_context=user_context,
        operator_config=session.master_operator_config,
        tracker=session.token_tracker,
        runtime_context=_build_master_runtime_context(toolkit, memory, optimizer),
    )
    # Persist operator config so the agent can be recreated on reconnect
    session_manager.persist(session.session_id)
    return {"session_id": session.session_id}


@app.get("/session/{session_id}/config", dependencies=[Depends(require_api_key)])
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


@app.get("/sessions", dependencies=[Depends(require_api_key)])
async def list_sessions():
    return {"sessions": session_manager.list_sessions()}


@app.delete("/session/{session_id}", dependencies=[Depends(require_api_key)])
async def delete_session(session_id: str):
    session_manager.delete_session(session_id)
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Topology inspection
# ---------------------------------------------------------------------------

@app.get("/topology/{session_id}", dependencies=[Depends(require_api_key)])
async def get_topology(session_id: str):
    """Return the pending topology for a session (if any)."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "topology": session.pending_topology,
        "estimate": session.pending_estimate,
    }


@app.get("/api/v1/topologies/{user_id}", dependencies=[Depends(require_api_key)])
async def list_user_topologies(user_id: str, limit: int = 10):
    """List saved topologies for a user, newest first."""
    import datetime
    topos = topology_store.list_topologies(user_id)[:limit]
    return {
        "user_id": user_id,
        "topologies": [
            {
                "topology_id": t.topology_id,
                "name": t.name or "(unnamed)",
                "updated_at": datetime.datetime.fromtimestamp(t.updated_at).strftime("%Y-%m-%d %H:%M"),
                "version": t.version,
            }
            for t in topos
        ],
    }


@app.get("/api/v1/topologies/{user_id}/{topology_id}/runs", dependencies=[Depends(require_api_key)])
async def list_topology_runs(user_id: str, topology_id: str, limit: int = 5):
    """List recent runs for a topology."""
    import datetime
    runs = topology_store.get_runs(topology_id, limit=limit)
    return {
        "topology_id": topology_id,
        "runs": [
            {
                "run_id": r.run_id,
                "ran_at": datetime.datetime.fromtimestamp(r.ran_at).strftime("%Y-%m-%d %H:%M"),
                "success": r.success,
                "latency_seconds": round(r.latency_seconds, 1),
            }
            for r in runs
        ],
    }


# ---------------------------------------------------------------------------
# Feedback / rating
# ---------------------------------------------------------------------------

class RatingRequest(BaseModel):
    session_id: str
    rating: int     # 1-5
    comment: str = ""


@app.post("/rate", dependencies=[Depends(require_api_key)])
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

    optimizer = _get_optimizer()
    if optimizer is not None:
        try:
            from daap.optimizer.integration import record_run_outcome
            topo = session.pending_topology or {}
            nodes = topo.get("nodes", []) or []
            constraints = topo.get("constraints", {}) or {}
            result = session.execution_result or {}
            record_run_outcome(
                user_id=session.user_id,
                user_prompt=topo.get("user_prompt", ""),
                node_configs={
                    n.get("role", ""): n.get("model_tier", "smart")
                    for n in nodes if isinstance(n, dict)
                },
                user_rating=req.rating,
                actual_cost_usd=float(result.get("cost_usd", 0) or 0),
                budget_usd=float(constraints.get("max_cost_usd", 0) or 0),
                latency_seconds=float(result.get("latency_seconds", 0) or 0),
                timeout_seconds=float(constraints.get("max_latency_seconds", 0) or 0),
                topology_id=result.get("topology_id"),
                node_count=len(nodes),
                has_parallel=any(
                    isinstance(n, dict) and
                    n.get("instance_config", {}).get("parallel_instances", 1) > 1
                    for n in nodes
                ),
            )
        except Exception as exc:
            logger.warning("RL record_run_outcome failed (non-fatal): %s", exc)

    # Mem0 feedback (optional)
    memory = _get_memory()
    if memory:
        try:
            topo = session.pending_topology or {}
            nodes = topo.get("nodes", []) or []
            result = session.execution_result or {}
            topology_summary = f"{len(nodes)} nodes, ${float(result.get('cost_usd', 0) or 0):.4f}"
            memory.remember_run(
                user_id=session.user_id,
                topology=topo,
                execution_result=result,
                user_rating=req.rating,
            )
            memory.remember_correction(
                user_id=session.user_id,
                rating=req.rating,
                comment=req.comment or None,
                topology_summary=topology_summary,
            )
        except Exception as exc:
            logger.warning("Failed to write memory feedback: %s", exc)

    return {"status": "rated", "rating": req.rating}


@app.get("/runs/{session_id}", dependencies=[Depends(require_api_key)])
async def get_run_history(session_id: str):
    """Get execution history for a session."""
    return {"runs": feedback_store.get_runs_for_session(session_id)}


# ---------------------------------------------------------------------------
# Memory inspection
# ---------------------------------------------------------------------------

@app.get("/api/v1/memory/{user_id}/profile", dependencies=[Depends(require_api_key)])
async def get_memory_profile(user_id: str):
    """Return user profile memories stored in Mem0."""
    memory = _get_memory()
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not available")
    try:
        profile = memory.get_user_profile(user_id)
        return {"user_id": user_id, "profile": profile}
    except Exception as exc:
        logger.warning("Failed to load memory profile: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/memory/{user_id}/history", dependencies=[Depends(require_api_key)])
async def get_memory_history(user_id: str, q: str = ""):
    """Return past run summaries from Mem0 for a user."""
    memory = _get_memory()
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not available")
    try:
        history = memory.get_past_runs(user_id, query=q)
        return {"user_id": user_id, "history": history}
    except Exception as exc:
        logger.warning("Failed to load memory history: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/memory/status", dependencies=[Depends(require_api_key)])
async def get_memory_status():
    """Return current memory availability/degraded status with error counters."""
    _get_memory()  # triggers lazy init and status updates
    from daap.memory.observability import get_memory_status as _get_memory_status

    return _get_memory_status()



@app.delete("/api/v1/memory/{user_id}", dependencies=[Depends(require_api_key)])
async def delete_user_memory(user_id: str):
    """Delete all Mem0 memories for a user."""
    memory = _get_memory()
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not available")
    try:
        from daap.memory.config import get_memory_client
        client = get_memory_client()
        client.delete_all(user_id=user_id)
        return {"status": "deleted", "user_id": user_id}
    except Exception as exc:
        logger.warning("Failed to delete user memory: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# WebSocket conversation
# ---------------------------------------------------------------------------

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, token: str | None = None):
    """
    WebSocket endpoint for real-time conversation with the master agent.

    Client must first POST /session to get a session_id, then connect here.
    Pass the API key as ?token=<key> query parameter.
    """
    try:
        validate_ws_token(token)
    except HTTPException:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    session = session_manager.get_session(session_id)
    if session is None:
        await websocket.close(code=4004, reason="Session not found")
        return

    # Recreate master_agent if session was restored from DB after a restart
    if session.master_agent is None:
        try:
            memory = _get_memory()
            optimizer = _get_optimizer()
            user_context = None
            if memory:
                try:
                    user_context = memory.format_for_master_prompt(session.user_id, "")
                except Exception:
                    pass
            session.token_tracker = TokenTracker()
            toolkit = create_session_scoped_toolkit(
                session,
                topology_store=topology_store,
                daap_memory=memory,
                rl_optimizer=optimizer,
                session_store=_session_store,
            )
            session.master_agent = create_master_agent_with_toolkit(
                toolkit,
                user_context=user_context,
                operator_config=session.master_operator_config,
                tracker=session.token_tracker,
                runtime_context=_build_master_runtime_context(toolkit, memory, optimizer),
            )
            logger.info("Recreated master_agent for restored session %s", session_id)
        except Exception as exc:
            logger.error("Failed to recreate agent for session %s: %s", session_id, exc)
            await websocket.close(code=4005, reason="Session agent recreation failed")
            return

    await handle_websocket(
        websocket,
        session,
        daap_memory=_get_memory(),
        topology_store=topology_store,
        persist_fn=session_manager.persist,
    )
