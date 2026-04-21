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

_CHAT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DAAP Chat</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0f1117;color:#e2e8f0;height:100dvh;display:flex;flex-direction:column}
header{padding:10px 18px;border-bottom:1px solid #1e2430;display:flex;align-items:center;gap:10px;background:#141921;min-height:44px;flex-shrink:0}
header h1{font-size:.95rem;font-weight:600;color:#94a3b8;letter-spacing:.05em}
#status-dot{width:8px;height:8px;border-radius:50%;background:#ef4444;flex-shrink:0;transition:background .3s}
#status-dot.connected{background:#22c55e}
#header-right{margin-left:auto;display:flex;align-items:center;gap:12px}
#session-label{font-size:.72rem;color:#475569}
#raw-badge{font-size:.7rem;padding:2px 7px;border-radius:4px;background:#1d4ed8;color:#bfdbfe;display:none}
#sidebar-toggle{background:none;border:none;color:#64748b;cursor:pointer;font-size:1.1rem;padding:2px 6px;border-radius:5px;line-height:1}
#sidebar-toggle:hover{background:#1e2430;color:#94a3b8}
#app-body{flex:1;display:flex;overflow:hidden}
#sidebar{width:220px;background:#0c0f16;border-right:1px solid #1e2430;display:flex;flex-direction:column;flex-shrink:0;transition:width .2s,opacity .2s}
#sidebar.collapsed{width:0;opacity:0;overflow:hidden}
#sidebar-header{padding:10px 12px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #1e2430;flex-shrink:0}
#sidebar-header span{font-size:.72rem;color:#475569;text-transform:uppercase;letter-spacing:.08em}
#new-session-btn{padding:3px 9px;background:#1d4ed8;color:#bfdbfe;border:none;border-radius:5px;font-size:.75rem;font-weight:600;cursor:pointer}
#new-session-btn:hover{background:#2563eb}
#session-list{flex:1;overflow-y:auto;padding:6px 0}
.sess-item{padding:8px 12px;cursor:pointer;border-bottom:1px solid #111827;transition:background .12s}
.sess-item:hover{background:#141921}
.sess-item.active{background:#1a2235;border-left:3px solid #3b82f6}
.sess-item.active .sess-id{color:#93c5fd}
.sess-id{font-size:.75rem;font-family:"SF Mono","Fira Code",monospace;color:#64748b;word-break:break-all}
.sess-meta{font-size:.68rem;color:#374151;margin-top:2px}
.sess-badge{display:inline-block;font-size:.62rem;padding:1px 5px;border-radius:3px;margin-left:4px;vertical-align:middle}
.sess-badge.exec{background:#78350f;color:#fbbf24}
.sess-badge.topo{background:#1e3a5f;color:#93c5fd}
#chat-area{flex:1;display:flex;flex-direction:column;overflow:hidden}
#messages{flex:1;overflow-y:auto;padding:18px;display:flex;flex-direction:column;gap:12px}
.msg{max-width:740px;padding:11px 15px;border-radius:12px;line-height:1.55;font-size:.875rem;white-space:pre-wrap;word-break:break-word}
.msg.user{background:#1e40af;align-self:flex-end;color:#e0e7ff;border-bottom-right-radius:3px}
.msg.agent{background:#1e2430;align-self:flex-start;border-bottom-left-radius:3px}
.msg.error{background:#450a0a;color:#fca5a5;align-self:flex-start;border-left:3px solid #ef4444}
.msg.system{background:transparent;color:#64748b;align-self:center;font-size:.78rem;font-style:italic;max-width:90%;text-align:center}
.usage-line{font-size:.72rem;color:#475569;margin-top:5px}
.plan-card{background:#1a2235;border:1px solid #2d3f5e;border-radius:12px;padding:15px;align-self:flex-start;max-width:560px;display:flex;flex-direction:column;gap:10px}
.plan-card h3{font-size:.78rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em}
.plan-card .summary{font-size:.875rem;color:#e2e8f0}
.plan-meta{display:flex;gap:16px;font-size:.78rem;color:#64748b}
.plan-actions{display:flex;gap:8px;flex-wrap:wrap}
.btn{padding:6px 14px;border:none;border-radius:7px;font-size:.82rem;font-weight:500;cursor:pointer;transition:opacity .15s}
.btn:hover{opacity:.85}.btn:disabled{opacity:.4;cursor:not-allowed}
.btn-approve{background:#16a34a;color:#fff}.btn-cheaper{background:#1d4ed8;color:#fff}.btn-cancel{background:#374151;color:#d1d5db}
.questions-card{background:#1a2235;border:1px solid #2d3f5e;border-radius:12px;padding:15px;align-self:flex-start;max-width:560px;display:flex;flex-direction:column;gap:11px}
.questions-card h3{font-size:.78rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em}
.q-item{display:flex;flex-direction:column;gap:4px}
.q-item label{font-size:.82rem;color:#cbd5e1}
.q-item input,.q-item select{background:#0f1117;border:1px solid #374151;border-radius:6px;color:#e2e8f0;padding:6px 10px;font-size:.85rem;outline:none}
.q-item input:focus,.q-item select:focus{border-color:#3b82f6}
.executing-row{display:flex;align-items:center;gap:10px;align-self:flex-start;color:#94a3b8;font-size:.82rem}
.spinner{width:15px;height:15px;border:2px solid #374151;border-top-color:#3b82f6;border-radius:50%;animation:spin .8s linear infinite;flex-shrink:0}
@keyframes spin{to{transform:rotate(360deg)}}
.result-card{background:#052e16;border:1px solid #166534;border-radius:12px;padding:15px;align-self:flex-start;max-width:740px}
.result-card h3{font-size:.78rem;color:#4ade80;margin-bottom:8px;text-transform:uppercase;letter-spacing:.08em}
.result-card pre{font-family:"SF Mono","Fira Code",monospace;font-size:.8rem;white-space:pre-wrap;word-break:break-word;color:#bbf7d0}
.result-meta{font-size:.75rem;color:#4ade80;margin-top:7px;opacity:.7}
#bottom{padding:12px 18px;border-top:1px solid #1e2430;background:#141921;display:flex;flex-direction:column;gap:6px;position:relative;flex-shrink:0}
#cmd-suggestions{display:none;position:absolute;bottom:100%;left:18px;right:18px;background:#1a2235;border:1px solid #2d3f5e;border-radius:8px;overflow:hidden;max-height:220px;overflow-y:auto;z-index:50}
.cmd-item{padding:7px 12px;cursor:pointer;display:flex;gap:12px;font-size:.82rem}
.cmd-item:hover,.cmd-item.active{background:#2d3f5e}
.cmd-item .cmd-name{color:#93c5fd;font-family:"SF Mono","Fira Code",monospace;flex-shrink:0;min-width:110px}
.cmd-item .cmd-desc{color:#64748b}
#input-row{display:flex;gap:8px}
#input{flex:1;background:#1e2430;border:1px solid #374151;border-radius:10px;color:#e2e8f0;padding:9px 13px;font-size:.875rem;resize:none;outline:none;max-height:160px;line-height:1.4}
#input:focus{border-color:#3b82f6}
#send-btn{padding:9px 16px;background:#3b82f6;color:#fff;border:none;border-radius:10px;font-size:.875rem;font-weight:500;cursor:pointer;align-self:flex-end;transition:background .15s}
#send-btn:hover{background:#2563eb}
#send-btn:disabled{background:#374151;cursor:not-allowed}
#hint{font-size:.72rem;color:#334155;padding:0 2px}
#setup-overlay{position:fixed;inset:0;background:rgba(0,0,0,.75);display:flex;align-items:center;justify-content:center;z-index:100}
#setup-box{background:#141921;border:1px solid #2d3f5e;border-radius:16px;padding:30px;width:360px;display:flex;flex-direction:column;gap:16px}
#setup-box h2{font-size:1.05rem;color:#e2e8f0}
#setup-box p{font-size:.82rem;color:#64748b}
.field{display:flex;flex-direction:column;gap:5px}
.field label{font-size:.82rem;color:#94a3b8}
.field input{background:#0f1117;border:1px solid #374151;border-radius:8px;color:#e2e8f0;padding:8px 12px;font-size:.875rem;outline:none}
.field input:focus{border-color:#3b82f6}
#start-btn{padding:10px;background:#3b82f6;color:#fff;border:none;border-radius:8px;font-size:.9rem;font-weight:600;cursor:pointer}
#start-btn:hover{background:#2563eb}
#setup-error{color:#fca5a5;font-size:.8rem;display:none}
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
  <button id="sidebar-toggle" title="Toggle sessions panel">☰</button>
  <div id="status-dot"></div>
  <h1>DAAP</h1>
  <div id="header-right">
    <span id="raw-badge">RAW</span>
    <span id="session-label"></span>
  </div>
</header>

<div id="app-body">
  <div id="sidebar">
    <div id="sidebar-header">
      <span>Sessions</span>
      <button id="new-session-btn" title="Start new session">+ New</button>
    </div>
    <div id="session-list"></div>
  </div>
  <div id="chat-area">
    <div id="messages"></div>
    <div id="bottom">
      <div id="cmd-suggestions"></div>
      <div id="input-row">
        <textarea id="input" rows="1" placeholder="Message or /command..." disabled></textarea>
        <button id="send-btn" disabled>Send</button>
      </div>
      <div id="hint">Enter to send · Shift+Enter for newline · /help for commands</div>
    </div>
  </div>
</div>

<script>
const AUTH_ENABLED = __AUTH_ENABLED__;

const CMD_DEFS = [
  ['/help',     'Show all commands'],
  ['/approve',  'Approve pending plan'],
  ['/cheaper',  'Make plan cheaper'],
  ['/cancel',   'Cancel pending plan'],
  ['/clear',    'Clear conversation display'],
  ['/history',  'List saved topologies'],
  ['/topology', 'List topologies  /topology load <id-prefix>'],
  ['/memory',   'Show memory  /memory search <q>  /memory clear'],
  ['/profile',  'Show user profile + memory facts'],
  ['/sessions', 'List active server sessions'],
  ['/rate',     '/rate <1-5> [comment]  rate last execution'],
  ['/raw',      'Show raw agent output including JSON'],
  ['/clean',    'Show clean agent output (default)'],
  ['/mcp',      'MCP server status'],
  ['/skills',   'List loaded skills'],
  ['/quit',     'Clear saved session and disconnect'],
];

let ws = null, sessionId = null, userId = null, apiKey = '';
let rawMode = false, hasResult = false;
let pendingPlanEl = null, executingEl = null;
let reconnectTimer = null, reconnectDelay = 2000, intentionalClose = false;
let cmdSuggIdx = -1;
let sidebarVisible = true;
let sessionPollTimer = null;

function saveSession() {
  try { localStorage.setItem('daap_s', JSON.stringify({sessionId, userId, apiKey})); } catch(e) {}
}
function clearStoredSession() {
  try { localStorage.removeItem('daap_s'); } catch(e) {}
}
function loadStoredSession() {
  try { return JSON.parse(localStorage.getItem('daap_s') || 'null'); } catch(e) { return null; }
}

// ---------------------------------------------------------------------------
// Sidebar: session list
// ---------------------------------------------------------------------------

function toggleSidebar() {
  sidebarVisible = !sidebarVisible;
  document.getElementById('sidebar').classList.toggle('collapsed', !sidebarVisible);
}

function fmtTime(ts) {
  if (!ts) return '';
  var d = new Date(ts * 1000);
  var now = new Date();
  if (d.toDateString() === now.toDateString()) {
    return d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  }
  return d.toLocaleDateString([], {month:'short', day:'numeric'});
}

async function loadSessionList() {
  if (!userId) return;
  try {
    var url = '/sessions?user_id=' + encodeURIComponent(userId);
    var hdrs = apiKey ? {'X-API-Key': apiKey} : {};
    var r = await fetch(url, {headers: hdrs});
    if (!r.ok) return;
    var data = await r.json();
    renderSessionList(data.sessions || []);
  } catch(e) {}
}

function renderSessionList(sessions) {
  var list = document.getElementById('session-list');
  list.innerHTML = '';
  // Sort newest first by created_at
  sessions.sort(function(a, b) { return (b.created_at || 0) - (a.created_at || 0); });
  if (!sessions.length) {
    var empty = document.createElement('div');
    empty.style.cssText = 'padding:14px 12px;font-size:.72rem;color:#374151;text-align:center';
    empty.textContent = 'No sessions yet';
    list.appendChild(empty);
    return;
  }
  sessions.forEach(function(s) {
    var item = document.createElement('div');
    item.className = 'sess-item' + (s.session_id === sessionId ? ' active' : '');
    var shortId = s.session_id.slice(0, 12) + '…';
    var meta = (s.message_count || 0) + ' msg · ' + fmtTime(s.created_at);
    var badges = '';
    if (s.is_executing) badges += '<span class="sess-badge exec">running</span>';
    else if (s.has_pending_topology) badges += '<span class="sess-badge topo">plan ready</span>';
    item.innerHTML =
      '<div class="sess-id">' + shortId + badges + '</div>' +
      '<div class="sess-meta">' + meta + '</div>';
    item.addEventListener('click', function() { switchToSession(s.session_id); });
    list.appendChild(item);
  });
}

async function switchToSession(targetId) {
  if (targetId === sessionId) return;
  // Tear down current WS gracefully
  intentionalClose = true;
  clearTimeout(reconnectTimer);
  if (ws) { ws.close(); ws = null; }
  clearTimeout(sessionPollTimer);

  sessionId = targetId;
  saveSession();
  document.getElementById('session-label').textContent = 'session: ' + sessionId.slice(0,8) + '...';
  document.getElementById('messages').innerHTML = '';
  hasResult = false; rawMode = false;
  pendingPlanEl = null; executingEl = null;

  appendSystem('Switched to session ' + sessionId.slice(0,8) + '…');
  connectWS();
  loadSessionList();
  startSessionPoll();
}

function startSessionPoll() {
  clearTimeout(sessionPollTimer);
  sessionPollTimer = setTimeout(function() {
    loadSessionList();
    startSessionPoll();
  }, 8000);
}

document.getElementById('sidebar-toggle').addEventListener('click', toggleSidebar);
document.getElementById('new-session-btn').addEventListener('click', async function() {
  if (!userId) return;
  this.disabled = true; this.textContent = '…';
  try { await startSession(userId); } finally {
    this.disabled = false; this.textContent = '+ New';
  }
});

window.addEventListener('DOMContentLoaded', async () => {
  if (AUTH_ENABLED) document.getElementById('key-field').style.display = '';
  const stored = loadStoredSession();
  if (stored && stored.sessionId && stored.userId) {
    const hdrs = stored.apiKey ? {'X-API-Key': stored.apiKey} : {};
    try {
      const r = await fetch('/session/' + stored.sessionId + '/config', {headers: hdrs});
      if (r.ok) {
        sessionId = stored.sessionId;
        userId    = stored.userId;
        apiKey    = stored.apiKey || '';
        document.getElementById('session-label').textContent = 'session: ' + sessionId.slice(0,8) + '...';
        document.getElementById('setup-overlay').style.display = 'none';
        connectWS();
        loadSessionList();
        startSessionPoll();
        return;
      }
    } catch(e) {}
    document.getElementById('uid-input').value = stored.userId || '';
    if (stored.apiKey) document.getElementById('key-input').value = stored.apiKey;
  } else if (stored && stored.userId) {
    document.getElementById('uid-input').value = stored.userId;
  }
});

document.getElementById('start-btn').addEventListener('click', async () => {
  const uid = document.getElementById('uid-input').value.trim();
  if (!uid) { showSetupErr('Enter a user ID'); return; }
  const key = document.getElementById('key-input').value.trim();
  if (AUTH_ENABLED && !key) { showSetupErr('Enter the API key'); return; }
  apiKey = key;
  await startSession(uid);
});
document.getElementById('uid-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') document.getElementById('start-btn').click();
});
function showSetupErr(msg) {
  const el = document.getElementById('setup-error');
  el.textContent = msg; el.style.display = '';
}

async function startSession(uid) {
  const btn = document.getElementById('start-btn');
  btn.disabled = true; btn.textContent = 'Connecting...';
  const hdrs = {'Content-Type': 'application/json'};
  if (apiKey) hdrs['X-API-Key'] = apiKey;
  try {
    const r = await fetch('/session?user_id=' + encodeURIComponent(uid), {method: 'POST', headers: hdrs});
    if (!r.ok) {
      const j = await r.json().catch(function() { return {}; });
      showSetupErr(j.detail || 'Error ' + r.status);
      btn.disabled = false; btn.textContent = 'Start chatting';
      return;
    }
    const data = await r.json();
    sessionId = data.session_id;
    userId    = uid;
    saveSession();
    document.getElementById('session-label').textContent = 'session: ' + sessionId.slice(0,8) + '...';
    document.getElementById('setup-overlay').style.display = 'none';
    connectWS();
    loadSessionList();
    startSessionPoll();
  } catch(e) {
    showSetupErr('Connection failed: ' + e.message);
    btn.disabled = false; btn.textContent = 'Start chatting';
  }
}

function connectWS() {
  clearTimeout(reconnectTimer);
  intentionalClose = false;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = proto + '://' + location.host + '/ws/' + sessionId + (apiKey ? '?token=' + encodeURIComponent(apiKey) : '');
  ws = new WebSocket(url);

  ws.onopen = function() {
    reconnectDelay = 2000;
    document.getElementById('status-dot').classList.add('connected');
    document.getElementById('input').disabled = false;
    document.getElementById('send-btn').disabled = false;
    document.getElementById('input').focus();
    appendSystem('Connected. Type a message or /help for commands.');
  };

  ws.onclose = function() {
    document.getElementById('status-dot').classList.remove('connected');
    document.getElementById('input').disabled = true;
    document.getElementById('send-btn').disabled = true;
    if (intentionalClose) return;
    appendSystem('Disconnected. Reconnecting in ' + (reconnectDelay/1000).toFixed(0) + 's...');
    reconnectTimer = setTimeout(function() {
      reconnectDelay = Math.min(reconnectDelay * 2, 30000);
      connectWS();
    }, reconnectDelay);
  };

  ws.onerror = function() {};
  ws.onmessage = function(e) { handleWS(JSON.parse(e.data)); };
}

function handleWS(data) {
  if (data.type === 'response') {
    removeExecuting(); removePlanCard(); appendAgent(data.content, data.usage);
  } else if (data.type === 'questions') {
    removeExecuting(); appendQuestions(data.questions);
  } else if (data.type === 'plan') {
    removeExecuting(); appendPlan(data);
  } else if (data.type === 'executing') {
    removeExecuting(); executingEl = appendExecuting(data.topology_id);
  } else if (data.type === 'progress') {
    updateProgress(data);
  } else if (data.type === 'result') {
    removeExecuting(); appendResult(data);
    hasResult = true;
    appendSystem('Rate this run: /rate <1-5> [comment]');
  } else if (data.type === 'error') {
    removeExecuting(); appendError(data.message);
  }
}

function scrollBottom() { var m = document.getElementById('messages'); m.scrollTop = m.scrollHeight; }
function appendMsg(el) { document.getElementById('messages').appendChild(el); scrollBottom(); return el; }

function appendUser(text) {
  var d = document.createElement('div');
  d.className = 'msg user'; d.textContent = text;
  return appendMsg(d);
}

function looksJson(s) {
  var t = s.trim();
  if (t.length < 2) return false;
  if (!((t[0]==='{' && t[t.length-1]==='}') || (t[0]==='[' && t[t.length-1]===']'))) return false;
  try { JSON.parse(t); return true; } catch(e) { return false; }
}
function summarizeJson(s) {
  try {
    var p = JSON.parse(s);
    if (Array.isArray(p)) return 'Structured list (' + p.length + ' items). Use /raw to inspect.';
    return 'Structured data (' + Object.keys(p).slice(0,4).join(', ') + '...). Use /raw to inspect.';
  } catch(e) { return 'Structured data generated. Use /raw to inspect.'; }
}
function sanitize(text) {
  if (rawMode) return text || '';
  var s = (text || '').replace(/```(?:json)?\s*([\s\S]*?)```/gi, function(_, inner) {
    inner = inner.trim();
    if (looksJson(inner)) return summarizeJson(inner);
    return inner;
  });
  if (looksJson(s.trim())) return summarizeJson(s.trim());
  return s.replace(/\n{3,}/g, '\n\n').trim() || 'Done.';
}

function appendAgent(text, usage) {
  var d = document.createElement('div');
  d.className = 'msg agent';
  d.textContent = sanitize(text);
  if (usage && usage.total_tokens) {
    var u = document.createElement('div');
    u.className = 'usage-line';
    u.textContent = usage.total_tokens + ' tokens  $' + (usage.total_cost_usd||0).toFixed(4);
    d.appendChild(u);
  }
  return appendMsg(d);
}
function appendSystem(text) {
  var d = document.createElement('div');
  d.className = 'msg system'; d.textContent = text;
  return appendMsg(d);
}
function appendError(text) {
  var d = document.createElement('div');
  d.className = 'msg error'; d.textContent = 'Error: ' + text;
  return appendMsg(d);
}
function appendExecuting(topoId) {
  var d = document.createElement('div');
  d.className = 'executing-row';
  d.innerHTML = '<div class="spinner"></div><span id="exec-label">Executing pipeline' + (topoId ? ' ' + topoId.slice(0,8) + '...' : '') + '</span>';
  return appendMsg(d);
}
function removeExecuting() {
  if (executingEl) { executingEl.remove(); executingEl = null; }
}
function updateProgress(data) {
  var label = document.getElementById('exec-label');
  if (!label) return;
  var done = data.completed_nodes || 0, total = data.total_nodes || 0;
  var filled = total ? Math.round((done/total)*20) : 0;
  var bar = '█'.repeat(filled) + '░'.repeat(20-filled);
  if (data.event === 'node_start') {
    label.textContent = '[' + bar + '] ' + done + '/' + total + '  running: ' + (data.node_id||'');
  } else if (data.event === 'node_complete') {
    label.textContent = '[' + bar + '] ' + done + '/' + total + (done >= total ? '  done' : '  done: ' + (data.node_id||''));
  }
}
function removePlanCard() { if (pendingPlanEl) { pendingPlanEl.remove(); pendingPlanEl = null; } }

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function appendPlan(data) {
  removePlanCard();
  var card = document.createElement('div');
  card.className = 'plan-card';
  var cost = (data.cost_usd||0).toFixed(4), minC = (data.min_cost_usd||0).toFixed(4), lat = (data.latency_seconds||0).toFixed(1);
  card.innerHTML =
    '<h3>Proposed Plan</h3>' +
    '<div class="summary">' + esc(data.summary||'') + '</div>' +
    '<div class="plan-meta"><span>Cost: <b>$' + cost + '</b> (min $' + minC + ')</span><span>Time: <b>' + lat + 's</b></span></div>' +
    '<div class="plan-actions">' +
      '<button class="btn btn-approve">Approve</button>' +
      '<button class="btn btn-cheaper">Make cheaper</button>' +
      '<button class="btn btn-cancel">Cancel</button>' +
    '</div>';
  card.querySelector('.btn-approve').onclick = function() {
    ws.send(JSON.stringify({type:'message',content:'approve'}));
    disablePlanBtns(card); appendUser('approve');
  };
  card.querySelector('.btn-cheaper').onclick = function() {
    ws.send(JSON.stringify({type:'make_cheaper'}));
    disablePlanBtns(card); appendUser('/cheaper');
  };
  card.querySelector('.btn-cancel').onclick = function() {
    ws.send(JSON.stringify({type:'cancel'}));
    disablePlanBtns(card); appendUser('/cancel');
  };
  pendingPlanEl = appendMsg(card);
}
function disablePlanBtns(card) { card.querySelectorAll('.btn').forEach(function(b) { b.disabled = true; }); }

function appendQuestions(questions) {
  var card = document.createElement('div');
  card.className = 'questions-card';
  card.innerHTML = '<h3>Questions</h3>';
  var inputs = [];
  questions.forEach(function(q, i) {
    var item = document.createElement('div');
    item.className = 'q-item';
    var lbl = document.createElement('label');
    lbl.textContent = (i+1) + '. ' + (typeof q === 'string' ? q : q.question || JSON.stringify(q));
    item.appendChild(lbl);
    var inp;
    if (typeof q === 'object' && q.options) {
      inp = document.createElement('select');
      q.options.forEach(function(o) {
        var opt = document.createElement('option');
        var label = typeof o === 'string' ? o : (o.label || JSON.stringify(o));
        opt.value = label;
        opt.textContent = label + (o.description ? ' - ' + o.description : '');
        inp.appendChild(opt);
      });
    } else {
      inp = document.createElement('input');
      inp.type = 'text'; inp.placeholder = 'Your answer...';
    }
    item.appendChild(inp); inputs.push(inp); card.appendChild(item);
  });
  var sub = document.createElement('button');
  sub.className = 'btn btn-approve'; sub.textContent = 'Submit answers';
  sub.onclick = function() {
    var answers = inputs.map(function(i) {
      return i.tagName === 'SELECT' ? (i.options[i.selectedIndex] ? i.options[i.selectedIndex].value : '') : i.value;
    });
    ws.send(JSON.stringify({type:'answer', answers:answers}));
    sub.disabled = true;
    inputs.forEach(function(i) { i.disabled = true; });
  };
  card.appendChild(sub);
  appendMsg(card);
}

function appendResult(data) {
  var card = document.createElement('div');
  card.className = 'result-card';
  var out = typeof data.output === 'string' ? data.output : JSON.stringify(data.output, null, 2);
  var cost = data.cost_usd != null ? '$' + parseFloat(data.cost_usd).toFixed(4) : '';
  var lat  = data.latency_seconds != null ? parseFloat(data.latency_seconds).toFixed(1) + 's' : '';
  card.innerHTML = '<h3>Result</h3><pre>' + esc(out) + '</pre>';
  if (cost || lat) {
    var m = document.createElement('div');
    m.className = 'result-meta'; m.textContent = [cost, lat].filter(Boolean).join(' · ');
    card.appendChild(m);
  }
  appendMsg(card);
}

// Command palette
var suggestEl = document.getElementById('cmd-suggestions');

function showSuggestions(prefix) {
  var matches = CMD_DEFS.filter(function(c) { return c[0].startsWith(prefix); });
  if (!matches.length || prefix === '/') {
    if (prefix === '/') matches = CMD_DEFS;
    else { hideSuggestions(); return; }
  }
  suggestEl.innerHTML = '';
  cmdSuggIdx = -1;
  matches.forEach(function(pair) {
    var item = document.createElement('div');
    item.className = 'cmd-item';
    item.innerHTML = '<span class="cmd-name">' + pair[0] + '</span><span class="cmd-desc">' + pair[1] + '</span>';
    item.addEventListener('mousedown', function(e) { e.preventDefault(); fillCommand(pair[0]); });
    suggestEl.appendChild(item);
  });
  suggestEl.style.display = '';
}
function hideSuggestions() { suggestEl.style.display = 'none'; cmdSuggIdx = -1; }
function fillCommand(cmd) {
  var inp = document.getElementById('input');
  inp.value = cmd + ' ';
  inp.focus(); hideSuggestions();
}
function moveSugg(dir) {
  var items = suggestEl.querySelectorAll('.cmd-item');
  if (!items.length) return;
  if (cmdSuggIdx >= 0) items[cmdSuggIdx].classList.remove('active');
  cmdSuggIdx = (cmdSuggIdx + dir + items.length) % items.length;
  items[cmdSuggIdx].classList.add('active');
  items[cmdSuggIdx].scrollIntoView({block:'nearest'});
}

function authHeaders() {
  var h = {'Content-Type':'application/json'};
  if (apiKey) h['X-API-Key'] = apiKey;
  return h;
}
function apiHeaders() {
  return apiKey ? {'X-API-Key': apiKey} : {};
}

async function runCommand(input) {
  var parts = input.trim().split(/\s+/);
  var cmd = parts[0].toLowerCase();
  var args = parts.slice(1);

  if (cmd === '/help') {
    var lines = CMD_DEFS.map(function(c) { return (c[0] + '             ').slice(0,14) + ' ' + c[1]; });
    appendSystem('Commands:\n' + lines.join('\n'));
  } else if (cmd === '/approve') {
    if (!ws || ws.readyState !== WebSocket.OPEN) { appendSystem('Not connected.'); return; }
    ws.send(JSON.stringify({type:'message', content:'approve'}));
    appendUser('approve');
  } else if (cmd === '/cheaper') {
    if (!ws || ws.readyState !== WebSocket.OPEN) { appendSystem('Not connected.'); return; }
    ws.send(JSON.stringify({type:'make_cheaper'}));
    appendUser('/cheaper');
  } else if (cmd === '/cancel') {
    if (!ws || ws.readyState !== WebSocket.OPEN) { appendSystem('Not connected.'); return; }
    ws.send(JSON.stringify({type:'cancel'}));
    appendUser('/cancel');
  } else if (cmd === '/clear') {
    document.getElementById('messages').innerHTML = '';
    appendSystem('Conversation display cleared.');
  } else if (cmd === '/raw') {
    rawMode = true;
    document.getElementById('raw-badge').style.display = '';
    appendSystem('Raw mode on - JSON shown as-is.');
  } else if (cmd === '/clean') {
    rawMode = false;
    document.getElementById('raw-badge').style.display = 'none';
    appendSystem('Clean mode on.');
  } else if (cmd === '/quit') {
    intentionalClose = true;
    clearTimeout(sessionPollTimer);
    clearStoredSession();
    if (ws) ws.close();
    appendSystem('Session cleared. Use "+ New" in the sidebar or refresh to start a new session.');
  } else if (cmd === '/rate') {
    await cmdRate(args);
  } else if (cmd === '/history' || cmd === '/topology') {
    await cmdTopology(args);
  } else if (cmd === '/memory') {
    await cmdMemory(args);
  } else if (cmd === '/profile') {
    await cmdProfile();
  } else if (cmd === '/sessions') {
    await cmdSessions();
  } else if (cmd === '/mcp') {
    appendSystem('MCP servers run server-side. The agent uses them automatically.\nTo add/remove: edit ~/.daap/mcpx.json on the server and restart.');
  } else if (cmd === '/skills') {
    appendSystem('Skills are loaded server-side and injected into the master agent prompt automatically.');
  } else {
    appendSystem('Unknown command: ' + cmd + '. Type /help.');
  }
}

async function cmdRate(args) {
  if (!hasResult) { appendSystem('No execution result to rate yet. Run a topology first.'); return; }
  var n = parseInt(args[0]);
  if (!n || n < 1 || n > 5) { appendSystem('Usage: /rate <1-5> [comment]'); return; }
  var comment = args.slice(1).join(' ');
  try {
    var r = await fetch('/rate', {method:'POST', headers:authHeaders(), body:JSON.stringify({session_id:sessionId, rating:n, comment:comment})});
    if (r.ok) {
      appendSystem('Rated ' + '★'.repeat(n) + '☆'.repeat(5-n) + ' (' + n + '/5). Optimizer updated.');
      hasResult = false;
    } else {
      appendSystem('Rating failed: ' + r.status);
    }
  } catch(e) { appendSystem('Rating error: ' + e.message); }
}

async function cmdTopology(args) {
  if (args[0] === 'load' && args[1]) {
    if (!ws || ws.readyState !== WebSocket.OPEN) { appendSystem('Not connected.'); return; }
    ws.send(JSON.stringify({type:'message', content:'Load topology ' + args.slice(1).join(' ') + ' and prepare it for execution'}));
    appendUser('/topology load ' + args.slice(1).join(' '));
    return;
  }
  try {
    var r = await fetch('/api/v1/topologies/' + encodeURIComponent(userId), {headers:apiHeaders()});
    if (!r.ok) { appendSystem('Could not fetch topologies: ' + r.status); return; }
    var data = await r.json();
    var topos = data.topologies;
    if (!topos.length) { appendSystem('No topologies saved yet.'); return; }
    var lines = topos.map(function(t) { return t.topology_id.slice(0,12) + '  ' + t.updated_at + '  ' + t.name + '  [v' + t.version + ']'; });
    appendSystem('Saved topologies (' + topos.length + '):\n' + lines.join('\n') + '\nUse /topology load <id-prefix> to reload.');
  } catch(e) { appendSystem('Topology error: ' + e.message); }
}

async function cmdMemory(args) {
  var sub = (args[0] || '').toLowerCase();
  if (sub === 'clear') {
    if (!confirm('Delete ALL memory facts for ' + userId + '?')) { appendSystem('Cancelled.'); return; }
    try {
      var r = await fetch('/api/v1/memory/' + encodeURIComponent(userId), {method:'DELETE', headers:apiHeaders()});
      appendSystem(r.ok ? 'All memory deleted.' : 'Delete failed: ' + r.status);
    } catch(e) { appendSystem('Memory error: ' + e.message); }
    return;
  }
  if (sub === 'search' && args[1]) {
    var q = args.slice(1).join(' ');
    try {
      var r = await fetch('/api/v1/memory/' + encodeURIComponent(userId) + '/history?q=' + encodeURIComponent(q), {headers:apiHeaders()});
      if (!r.ok) { appendSystem('Search failed: ' + r.status); return; }
      var data = await r.json();
      var hist = data.history;
      if (!hist.length) { appendSystem('No results for "' + q + '".'); return; }
      appendSystem('Memory search "' + q + '" (' + hist.length + '):\n' + hist.map(function(h,i) { return (i+1)+'. '+h; }).join('\n'));
    } catch(e) { appendSystem('Memory error: ' + e.message); }
    return;
  }
  try {
    var r = await fetch('/api/v1/memory/' + encodeURIComponent(userId) + '/profile', {headers:apiHeaders()});
    if (!r.ok) { appendSystem('Memory unavailable: ' + r.status); return; }
    var data = await r.json();
    var profile = data.profile;
    if (!profile.length) { appendSystem('No memory facts stored.\nUsage: /memory search <query>  /memory clear'); return; }
    appendSystem('Memory facts (' + profile.length + '):\n' + profile.map(function(f,i) { return (i+1)+'. '+f; }).join('\n') + '\nUsage: /memory search <query>  /memory clear');
  } catch(e) { appendSystem('Memory error: ' + e.message); }
}

async function cmdProfile() {
  var lines = ['User: ' + userId, 'Session: ' + sessionId];
  try {
    var r = await fetch('/api/v1/memory/' + encodeURIComponent(userId) + '/profile', {headers:apiHeaders()});
    if (r.ok) {
      var data = await r.json();
      var profile = data.profile;
      if (profile.length) {
        lines.push('Memory (' + profile.length + ' facts):');
        profile.slice(0,5).forEach(function(f) { lines.push('  - ' + f); });
        if (profile.length > 5) lines.push('  [+ ' + (profile.length-5) + ' more]');
      } else {
        lines.push('Memory: no facts yet');
      }
    } else {
      lines.push('Memory: unavailable');
    }
  } catch(e) { lines.push('Memory: unavailable'); }
  appendSystem(lines.join('\n'));
}

async function cmdSessions() {
  try {
    var url = '/sessions' + (userId ? '?user_id=' + encodeURIComponent(userId) : '');
    var r = await fetch(url, {headers:apiHeaders()});
    if (!r.ok) { appendSystem('Could not fetch sessions: ' + r.status); return; }
    var data = await r.json();
    var sessions = data.sessions;
    if (!sessions.length) { appendSystem('No sessions. Use the sidebar "+ New" button to create one.'); return; }
    sessions.sort(function(a,b){return (b.created_at||0)-(a.created_at||0);});
    var lines = sessions.map(function(s) {
      var tag = s.session_id === sessionId ? ' ← current' : '';
      var state = s.is_executing ? ' [running]' : (s.has_pending_topology ? ' [plan ready]' : '');
      return s.session_id.slice(0,14) + '  ' + (s.message_count||0) + ' msg' + state + tag;
    });
    appendSystem('Your sessions (' + sessions.length + '):\n' + lines.join('\n') + '\nUse the sidebar to switch sessions.');
    loadSessionList();
  } catch(e) { appendSystem('Sessions error: ' + e.message); }
}

var inputEl = document.getElementById('input');
var sendBtn = document.getElementById('send-btn');

function send() {
  var text = inputEl.value.trim();
  if (!text) return;
  hideSuggestions();
  if (text.startsWith('/')) {
    inputEl.value = ''; inputEl.style.height = '';
    runCommand(text);
    return;
  }
  if (!ws || ws.readyState !== WebSocket.OPEN) { appendSystem('Not connected.'); return; }
  ws.send(JSON.stringify({type:'message', content:text}));
  appendUser(text);
  inputEl.value = ''; inputEl.style.height = '';
}

sendBtn.addEventListener('click', send);

inputEl.addEventListener('keydown', function(e) {
  if (suggestEl.style.display !== 'none') {
    if (e.key === 'ArrowDown') { e.preventDefault(); moveSugg(1); return; }
    if (e.key === 'ArrowUp')   { e.preventDefault(); moveSugg(-1); return; }
    if (e.key === 'Tab' || (e.key === 'Enter' && cmdSuggIdx >= 0)) {
      var active = suggestEl.querySelector('.cmd-item.active');
      if (active) {
        e.preventDefault();
        fillCommand(active.querySelector('.cmd-name').textContent);
        return;
      }
    }
    if (e.key === 'Escape') { hideSuggestions(); return; }
  }
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

inputEl.addEventListener('input', function() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
  var val = inputEl.value;
  if (val.startsWith('/')) showSuggestions(val.split(' ')[0]);
  else hideSuggestions();
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
    session = session_manager.create_session(user_id=user_id)
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
async def list_sessions(user_id: str | None = None):
    return {"sessions": session_manager.list_sessions(user_id=user_id)}


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
