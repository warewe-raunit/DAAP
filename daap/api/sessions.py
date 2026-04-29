"""
DAAP Session Management — in-memory session store with session-scoped toolkit.

Each session holds a live master agent, conversation history, and pending
topology state. The session-scoped ask_user closure fixes the concurrency
bug from Section 4's module-level state.
"""

import asyncio
import json
import logging
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from agentscope.message import TextBlock
from agentscope.tool import Toolkit, ToolResponse

from daap.master.tools import (
    _delegate_to_architect_impl,
    register_skill as register_skill_tool,
)
from daap.master.runtime import build_master_runtime_snapshot
from daap.retention import get_data_dir, get_session_ttl_hours
from daap.skills.manager import apply_configured_skills

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """A single user conversation session."""
    session_id: str
    created_at: float
    user_id: str = "default"

    master_agent: object | None = None          # ReActAgent (typed as object to avoid circular)
    conversation: list[dict] = field(default_factory=list)

    # Topology state — stored as raw dict after delegate_to_architect succeeds
    pending_topology: dict | None = None        # raw TopologySpec dict
    pending_estimate: dict | None = None        # estimate data dict
    topology_just_generated: bool = False       # set by delegate_to_architect wrapper; cleared by ws_handler

    # Execution state
    is_executing: bool = False
    execution_result: dict | None = None
    execution_progress: dict | None = None

    # Optional model selection (set at session creation)
    master_operator_config: dict | None = None
    subagent_operator_config: dict | None = None

    # Token usage tracking — reset per turn, read by ws_handler
    token_tracker: object | None = None         # TokenTracker instance

    # ask_user state — set by session-scoped closure, read by ws_handler
    pending_questions: list | None = None
    _resolve_answers: object | None = None      # callable injected by create_session_scoped_toolkit
    _questions_event: object | None = None      # asyncio.Event: set when pending_questions is ready

    # File permission state — set by _file_permission_fn, resolved by ws_handler
    pending_permission: dict | None = None      # {"filepath": str, "operation": str}
    _resolve_permission: object | None = None   # callable(granted: bool)
    _file_permission_fn: object | None = None   # async (filepath, op) -> bool

    # WebSocket send callback — set by ws_handler so agent tools can stream progress
    _ws_send: object | None = None              # async callable: (dict) -> None


# ---------------------------------------------------------------------------
# Session persistence (SQLite)
# ---------------------------------------------------------------------------

_DEFAULT_SESSION_TTL_HOURS = get_session_ttl_hours()
_SESSION_COMPACTION_TRIGGER_MESSAGES = 160
_SESSION_COMPACTION_KEEP_RECENT_MESSAGES = 100
_SESSION_COMPACTION_MAX_BULLETS = 8
_SESSION_SUMMARY_PREFIX = "[DAAP conversation summary]"

# Fields serialised to DB. Excludes live objects (agent, callables, asyncio state).
_PERSIST_FIELDS = (
    "session_id", "user_id", "created_at",
    "conversation",
    "pending_topology", "pending_estimate",
    "master_operator_config", "subagent_operator_config",
    "execution_result",
)


class SessionStore:
    """SQLite-backed persistence for sessions across server restarts.

    Persists serialisable session state so users don't lose work on deploy.
    TTL: sessions inactive for >24 h are purged automatically.
    Non-serialisable state (master_agent, asyncio events, ws callbacks) is
    recreated on reconnect.
    """

    def __init__(self, db_path: str | None = None):
        resolved = Path(db_path) if db_path else get_data_dir() / "daap_sessions.db"
        self.db_path = str(resolved)
        self.ttl_hours = _DEFAULT_SESSION_TTL_HOURS
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id              TEXT PRIMARY KEY,
                    user_id                 TEXT NOT NULL DEFAULT 'default',
                    created_at              REAL NOT NULL,
                    updated_at              REAL NOT NULL,
                    conversation            TEXT NOT NULL DEFAULT '[]',
                    pending_topology        TEXT,
                    pending_estimate        TEXT,
                    master_operator_config  TEXT,
                    subagent_operator_config TEXT,
                    execution_result        TEXT,
                    is_executing            INTEGER NOT NULL DEFAULT 0
                )
            """)
            # Migrate existing databases that pre-date the is_executing column.
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN is_executing INTEGER NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # column already exists
            conn.commit()

    def save(self, session: "Session") -> None:
        """Upsert serialisable session fields."""
        now = time.time()
        compacted_conversation = _compact_conversation_for_storage(session.conversation)
        if len(compacted_conversation) < len(session.conversation):
            logger.info(
                "SessionStore: compacted conversation for %s from %d to %d messages",
                session.session_id,
                len(session.conversation),
                len(compacted_conversation),
            )
            session.conversation = compacted_conversation
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions
                    (session_id, user_id, created_at, updated_at,
                     conversation, pending_topology, pending_estimate,
                     master_operator_config, subagent_operator_config,
                     execution_result, is_executing)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id                  = excluded.user_id,
                    updated_at               = excluded.updated_at,
                    conversation             = excluded.conversation,
                    pending_topology         = excluded.pending_topology,
                    pending_estimate         = excluded.pending_estimate,
                    master_operator_config   = excluded.master_operator_config,
                    subagent_operator_config = excluded.subagent_operator_config,
                    execution_result         = excluded.execution_result,
                    is_executing             = excluded.is_executing
            """, (
                session.session_id,
                session.user_id,
                session.created_at,
                now,
                json.dumps(compacted_conversation),
                json.dumps(session.pending_topology) if session.pending_topology else None,
                json.dumps(session.pending_estimate) if session.pending_estimate else None,
                json.dumps(session.master_operator_config) if session.master_operator_config else None,
                json.dumps(session.subagent_operator_config) if session.subagent_operator_config else None,
                json.dumps(session.execution_result) if session.execution_result else None,
                1 if session.is_executing else 0,
            ))
            conn.commit()

    def set_executing(self, session_id: str, value: bool) -> None:
        """Targeted write for is_executing — cheaper than a full save()."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE sessions SET is_executing = ?, updated_at = ? WHERE session_id = ?",
                (1 if value else 0, time.time(), session_id),
            )
            conn.commit()

    def load_one(self, session_id: str, ttl_hours: int | None = None) -> dict | None:
        """Return a single session row if it exists and is within TTL, else None."""
        ttl = self.ttl_hours if ttl_hours is None else ttl_hours
        cutoff = time.time() - ttl * 3600
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ? AND updated_at >= ?",
                (session_id, cutoff),
            ).fetchone()
        return dict(row) if row else None

    def load_active(self, ttl_hours: int | None = None) -> list[dict]:
        """Return all sessions updated within the TTL window."""
        ttl = self.ttl_hours if ttl_hours is None else ttl_hours
        cutoff = time.time() - ttl * 3600
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sessions WHERE updated_at >= ?", (cutoff,)
            ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, session_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()

    def purge_expired(self, ttl_hours: int | None = None) -> int:
        """Hard-delete sessions older than ttl_hours. Returns count deleted."""
        ttl = self.ttl_hours if ttl_hours is None else ttl_hours
        cutoff = time.time() - ttl * 3600
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM sessions WHERE updated_at < ?", (cutoff,)
            )
            conn.commit()
            return cur.rowcount


def _is_compaction_summary(message: dict) -> bool:
    metadata = message.get("metadata", {}) if isinstance(message, dict) else {}
    if not isinstance(metadata, dict):
        return False
    return bool(metadata.get("daap_compacted"))


def _extract_message_text(message: dict) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block.strip())
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"].strip())
        return " ".join(p for p in parts if p).strip()
    return ""


def _build_compaction_summary(messages: list[dict]) -> dict:
    user_samples: list[str] = []
    assistant_samples: list[str] = []
    prior_compacted = 0
    for msg in messages:
        if _is_compaction_summary(msg):
            metadata = msg.get("metadata", {}) if isinstance(msg, dict) else {}
            if isinstance(metadata, dict):
                prior_compacted += int(metadata.get("compacted_count", 0) or 0)
            continue

        role = str(msg.get("role", "unknown")) if isinstance(msg, dict) else "unknown"
        text = _extract_message_text(msg)
        if not text:
            continue
        clipped = text.replace("\n", " ")[:180]
        if role == "user" and len(user_samples) < _SESSION_COMPACTION_MAX_BULLETS:
            user_samples.append(clipped)
        elif role == "assistant" and len(assistant_samples) < _SESSION_COMPACTION_MAX_BULLETS:
            assistant_samples.append(clipped)

    compacted_count = len(messages) + prior_compacted
    lines = [
        _SESSION_SUMMARY_PREFIX,
        f"Compacted {compacted_count} historical messages.",
    ]
    if user_samples:
        lines.append("Recent user intents (compressed):")
        lines.extend(f"- {entry}" for entry in user_samples)
    if assistant_samples:
        lines.append("Recent assistant outputs (compressed):")
        lines.extend(f"- {entry}" for entry in assistant_samples)

    return {
        "role": "assistant",
        "content": "\n".join(lines),
        "metadata": {
            "daap_compacted": True,
            "compacted_count": compacted_count,
            "strategy": "deterministic-summary-v1",
        },
    }


def _compact_conversation_for_storage(conversation: list[dict]) -> list[dict]:
    """
    Deterministically compact long conversation history for persistence.

    Keeps the most recent window verbatim and replaces older history with a
    structured summary message so restart/recovery retains intent continuity.
    """
    if len(conversation) <= _SESSION_COMPACTION_TRIGGER_MESSAGES:
        return conversation

    recent = conversation[-_SESSION_COMPACTION_KEEP_RECENT_MESSAGES:]
    older = conversation[:-_SESSION_COMPACTION_KEEP_RECENT_MESSAGES]
    summary = _build_compaction_summary(older)
    recent_without_summary = [m for m in recent if not _is_compaction_summary(m)]
    return [summary, *recent_without_summary]


def _merge_from_row(session: "Session", row: dict) -> None:
    """Refresh serialisable fields on a live session from a fresh DB row.

    Called by get_session() on every lookup when a store is present so that
    state written by another worker is visible immediately. Live objects
    (master_agent, asyncio events, ws callbacks) are left untouched.
    """
    session.conversation = json.loads(row["conversation"] or "[]")
    session.pending_topology = json.loads(row["pending_topology"]) if row["pending_topology"] else None
    session.pending_estimate = json.loads(row["pending_estimate"]) if row["pending_estimate"] else None
    session.master_operator_config = json.loads(row["master_operator_config"]) if row["master_operator_config"] else None
    session.subagent_operator_config = json.loads(row["subagent_operator_config"]) if row["subagent_operator_config"] else None
    session.execution_result = json.loads(row["execution_result"]) if row["execution_result"] else None
    session.is_executing = bool(row.get("is_executing", 0))


def _restore_session(row: dict) -> "Session":
    """Reconstruct a Session from a persisted DB row (no live objects)."""
    s = Session(
        session_id=row["session_id"],
        user_id=row["user_id"],
        created_at=row["created_at"],
    )
    s.conversation = json.loads(row["conversation"] or "[]")
    s.pending_topology = json.loads(row["pending_topology"]) if row["pending_topology"] else None
    s.pending_estimate = json.loads(row["pending_estimate"]) if row["pending_estimate"] else None
    s.master_operator_config = json.loads(row["master_operator_config"]) if row["master_operator_config"] else None
    s.subagent_operator_config = json.loads(row["subagent_operator_config"]) if row["subagent_operator_config"] else None
    s.execution_result = json.loads(row["execution_result"]) if row["execution_result"] else None
    s.is_executing = bool(row.get("is_executing", 0))
    # master_agent is None — recreated on first WS reconnect (see routes.py)
    return s


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------

class SessionManager:
    """In-memory session index backed by optional SQLite persistence.

    On startup, loads active sessions from the store so users don't lose work
    across server restarts. master_agent is recreated lazily on WS reconnect.
    TTL: sessions inactive for >24 h are dropped from memory and DB.
    """

    def __init__(self, store: SessionStore | None = None):
        self._sessions: dict[str, Session] = {}
        self._store = store
        if store is not None:
            self._load_from_store(store)

    def _load_from_store(self, store: SessionStore) -> None:
        purged = store.purge_expired()
        if purged:
            logger.info("SessionManager: purged %d expired sessions from store", purged)
        for row in store.load_active():
            session = _restore_session(row)
            self._sessions[session.session_id] = session
        if self._sessions:
            logger.info(
                "SessionManager: restored %d sessions from store", len(self._sessions)
            )

    def create_session(self, user_id: str = "default") -> Session:
        session_id = secrets.token_urlsafe(32)
        session = Session(session_id=session_id, created_at=time.time(), user_id=user_id)
        self._sessions[session_id] = session
        if self._store is not None:
            self._store.save(session)
        return session

    def get_session(self, session_id: str) -> Session | None:
        if self._store is None:
            return self._sessions.get(session_id)

        # Always read from DB so writes by other workers are visible immediately.
        row = self._store.load_one(session_id)
        if row is None:
            # Expired or never existed — evict stale local entry if present.
            self._sessions.pop(session_id, None)
            return None

        local = self._sessions.get(session_id)
        if local is not None:
            # Merge fresh serialisable state; keep live objects (agent, events).
            _merge_from_row(local, row)
            return local

        # First time this worker sees this session — restore without live objects.
        session = _restore_session(row)
        self._sessions[session_id] = session
        return session

    def persist(self, session_id: str) -> None:
        """Flush serialisable session state to the store (no-op if no store)."""
        if self._store is None:
            return
        session = self._sessions.get(session_id)
        if session is not None:
            self._store.save(session)

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        if self._store is not None:
            self._store.delete(session_id)

    def list_sessions(self, user_id: str | None = None) -> list[dict]:
        if self._store is not None:
            rows = self._store.load_active()
            if user_id is not None:
                rows = [r for r in rows if r.get("user_id") == user_id]
            return [
                {
                    "session_id": r["session_id"],
                    "user_id": r.get("user_id", "default"),
                    "created_at": r["created_at"],
                    "message_count": len(json.loads(r["conversation"] or "[]")),
                    "has_pending_topology": r["pending_topology"] is not None,
                    "is_executing": bool(r.get("is_executing", 0)),
                }
                for r in rows
            ]
        sessions = self._sessions.values()
        if user_id is not None:
            sessions = [s for s in sessions if s.user_id == user_id]
        return [
            {
                "session_id": s.session_id,
                "user_id": s.user_id,
                "created_at": s.created_at,
                "message_count": len(s.conversation),
                "has_pending_topology": s.pending_topology is not None,
                "is_executing": s.is_executing,
            }
            for s in sessions
        ]


# ---------------------------------------------------------------------------
# Session-scoped toolkit factory
# ---------------------------------------------------------------------------

def create_session_scoped_toolkit(
    session: Session,
    topology_store=None,
    daap_memory=None,
    rl_optimizer=None,
    session_store: "SessionStore | None" = None,
) -> Toolkit:
    """
    Build a Toolkit with ask_user scoped to this specific session.

    The session-scoped ask_user closure:
    - Stores pending questions on the Session object (not module-level state)
    - Uses a per-session asyncio.Event — no cross-session interference
    - Attaches session._resolve_answers for the WebSocket handler to call

    delegate_to_architect is wrapped per session so we can inject user-selected
    subagent model/operator settings after the architect returns the topology.
    """
    toolkit = Toolkit()
    apply_configured_skills(toolkit, target="master")

    # Per-session architect grounding context — set by the agent factory
    # (create_master_agent_with_toolkit) after the runtime snapshot is built.
    # Forwarded into the architect so it designs against today's date,
    # connected MCP servers, known capability gaps, and user memory.
    _architect_ctx: dict = {
        "user_context": None,
        "runtime_context": None,
        "operator_config": None,
    }

    # Per-session topology handoff slot — replaces the prior ContextVar.
    _last_topology: dict = {"topology": None, "estimate": None}

    def set_architect_context(
        *,
        user_context: object = None,
        runtime_context: dict | None = None,
        operator_config: dict | None = None,
    ) -> None:
        _architect_ctx["user_context"] = user_context
        _architect_ctx["runtime_context"] = runtime_context
        _architect_ctx["operator_config"] = operator_config

    def _merge_operator_config(existing: dict | None, selected: dict) -> dict:
        existing_cfg = existing if isinstance(existing, dict) else {}
        merged = dict(existing_cfg)

        for key in ("provider", "base_url", "api_key_env"):
            selected_val = selected.get(key)
            if selected_val is not None:
                merged[key] = selected_val

        existing_model_map = (
            existing_cfg.get("model_map")
            if isinstance(existing_cfg.get("model_map"), dict)
            else {}
        )
        selected_model_map = (
            selected.get("model_map")
            if isinstance(selected.get("model_map"), dict)
            else {}
        )

        model_map = dict(existing_model_map)
        model_map.update({k: v for k, v in selected_model_map.items() if v})
        merged["model_map"] = model_map
        return merged

    async def delegate_to_architect(
        current_topology_json: str,
        user_feedback: str,
    ) -> ToolResponse:
        """Session-scoped topology architect delegation with optional operator/model injection."""
        result = await _delegate_to_architect_impl(
            current_topology_json,
            user_feedback,
            user_context=_architect_ctx.get("user_context"),
            runtime_context=_architect_ctx.get("runtime_context"),
            operator_config=_architect_ctx.get("operator_config"),
        )

        # Read topology directly from the impl's structured ToolResponse.metadata.
        meta = result.metadata or {}
        topo_dict = meta.get("topology")
        estimate = meta.get("estimate")
        if topo_dict is not None:
            selected = session.subagent_operator_config
            if selected:
                topo_dict["operator_config"] = _merge_operator_config(
                    topo_dict.get("operator_config"),
                    selected,
                )
                for node in topo_dict.get("nodes", []):
                    if not isinstance(node, dict):
                        continue
                    if node.get("operator_override") is not None:
                        node["operator_override"] = _merge_operator_config(
                            node.get("operator_override"),
                            selected,
                        )

            session.pending_topology = topo_dict
            session.pending_estimate = estimate
            session.topology_just_generated = True
            # Mirror into per-session slot for parse_turn_result. Cleared by
            # the agent layer after consumption — not here.
            _last_topology["topology"] = topo_dict
            _last_topology["estimate"] = estimate

        return result

    def get_last_topology_result() -> dict:
        return {
            "topology": _last_topology["topology"],
            "estimate": _last_topology["estimate"],
        }

    def clear_last_topology_result() -> None:
        _last_topology["topology"] = None
        _last_topology["estimate"] = None

    toolkit.register_tool_function(delegate_to_architect)

    # Per-session file permission state
    _perm_event: asyncio.Event = asyncio.Event()
    _perm_state: dict = {"granted": None}

    async def _file_permission_fn(filepath: str, operation: str) -> bool:
        """Pause agent, ask user via WS to approve out-of-cwd file access."""
        session.pending_permission = {"filepath": filepath, "operation": operation}
        _perm_event.clear()
        _perm_state["granted"] = None
        await _perm_event.wait()
        session.pending_permission = None
        return bool(_perm_state["granted"])

    def _resolve_permission(granted: bool) -> None:
        _perm_state["granted"] = granted
        _perm_event.set()

    session._resolve_permission = _resolve_permission
    session._file_permission_fn = _file_permission_fn

    # Per-session state captured by the closure
    _answer_event: asyncio.Event = asyncio.Event()
    _questions_event: asyncio.Event = asyncio.Event()
    _state: dict = {"user_answers": None}
    session._questions_event = _questions_event

    async def ask_user(questions_json: str) -> ToolResponse:
        """Present structured options to the user and wait for their choice.

        Use ONLY for structured option-picker flows where the user must choose
        between labeled alternatives — not for open-ended clarification questions.

        Primary use cases:
        - Topology approval: "Proceed / Make it cheaper / Cancel"
        - Any other explicit multiple-choice decision

        For open-ended clarification (missing product, audience, format, etc.),
        ask in plain text instead and let the user reply in the next turn.

        Args:
            questions_json: JSON array of question objects. Each has:
                - "question": the question text
                - "options": array of {"label": str, "description": str}
                - "multi_select": boolean

                Keep to 1-4 questions. Mark recommended option with "(Recommended)".

        Returns:
            The user's selected answers.
        """
        try:
            questions = json.loads(questions_json)
        except json.JSONDecodeError as exc:
            return ToolResponse(content=[TextBlock(
                type="text", text=f"Invalid questions JSON: {exc}",
            )])

        if not isinstance(questions, list) or len(questions) == 0:
            return ToolResponse(content=[TextBlock(
                type="text",
                text="questions_json must be a non-empty JSON array.",
            )])

        # Store on session and signal ws_handler (no polling needed)
        session.pending_questions = questions
        _answer_event.clear()
        _state["user_answers"] = None
        _questions_event.set()

        # Pause until WebSocket handler calls _resolve_answers()
        # Timeout avoids hanging forever if the client disconnects mid-question.
        _ASK_TIMEOUT = 300.0
        try:
            await asyncio.wait_for(_answer_event.wait(), timeout=_ASK_TIMEOUT)
        except asyncio.TimeoutError:
            session.pending_questions = None
            _questions_event.clear()
            logger.warning(
                "ask_user: no answer received in %.0fs for session %s — client may have disconnected",
                _ASK_TIMEOUT,
                session.session_id,
            )
            return ToolResponse(content=[TextBlock(
                type="text",
                text="User did not respond in time. Proceed with what you know or ask again.",
            )])

        # Clear pending state
        session.pending_questions = None
        _questions_event.clear()
        answers = _state["user_answers"]

        if answers is None:
            return ToolResponse(content=[TextBlock(
                type="text",
                text="User did not provide answers. Proceed with what you know or ask again.",
            )])
            
        # Direct execution safeguard: if the user's answer implies approval,
        # execute directly to prevent the master agent from hallucinating or 
        # forgetting to call execute_pending_topology.
        if session.pending_topology is not None and answers:
            ans_lower = " ".join(answers[0].strip().lower().split())
            approve_phrases = {
                "yes", "yes execute", "approve", "proceed", "run", "execute", 
                "run it", "execute it", "proceed with this topology and continue."
            }
            if ans_lower in approve_phrases:
                logger.info("ask_user tool detected approval. Directly executing topology %s", session.pending_topology.get("topology_id"))
                session.topology_just_generated = False
                return await execute_pending_topology()

            cancel_phrases = {
                "cancel", "abort", "stop"
            }
            if ans_lower in cancel_phrases:
                logger.info("ask_user tool detected cancel.")
                session.pending_topology = None
                session.pending_estimate = None
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text="User cancelled the topology. Acknowledge and ask what to do next."
                )])

        lines = []
        for i, q in enumerate(questions):
            a = answers[i] if i < len(answers) else "(no answer)"
            lines.append(f"Q: {q.get('question', '')}\nA: {a}")

        return ToolResponse(content=[TextBlock(
            type="text",
            text="User's answers:\n\n" + "\n\n".join(lines),
        )])

    def resolve_answers(answers: list[str]) -> None:
        _state["user_answers"] = answers
        _answer_event.set()

    # Attach resolver to session so ws_handler can call it
    session._resolve_answers = resolve_answers

    toolkit.register_tool_function(ask_user)
    toolkit.register_tool_function(register_skill_tool)

    async def get_runtime_context() -> ToolResponse:
        """Return current runtime capabilities and infrastructure for this session."""
        snapshot = build_master_runtime_snapshot(
            toolkit,
            execution_mode="api-session",
            memory_enabled=bool(daap_memory),
            optimizer_enabled=bool(rl_optimizer),
            topology_store_enabled=topology_store is not None,
            feedback_store_enabled=True,
            session_store_enabled=session_store is not None,
            extra={
                "session": {
                    "has_pending_topology": session.pending_topology is not None,
                    "is_executing": bool(session.is_executing),
                }
            },
        )
        return ToolResponse(content=[TextBlock(
            type="text",
            text=json.dumps(snapshot, indent=2),
        )])

    toolkit.register_tool_function(get_runtime_context)

    async def get_execution_status() -> ToolResponse:
        """Check the status of the current or most recent topology execution.

        Call this when the user asks about execution progress, whether the
        topology is running, or what the results were. This is the ONLY way
        to check execution state — do NOT say you cannot check it.

        Returns:
            Current execution status and results (if available).
        """
        progress = session.execution_progress or {}

        if session.is_executing:
            completed = int(progress.get("completed_nodes", 0))
            total = int(progress.get("total_nodes", 0))
            remaining = int(progress.get("remaining_nodes", max(total - completed, 0)))
            percent = int(progress.get("percent_complete", 0))
            current_node = progress.get("current_node") or "initializing"
            return ToolResponse(content=[TextBlock(
                type="text",
                text=(
                    "Topology is currently executing.\n"
                    f"Progress: {completed}/{total} nodes complete ({percent}%).\n"
                    f"Current node: {current_node}\n"
                    f"Remaining nodes: {remaining}"
                ),
            )])

        result = session.execution_result
        if result is None:
            pending = session.pending_topology
            if pending:
                nodes = [n.get("node_id") for n in pending.get("nodes", [])]
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text=f"No execution yet. A topology is pending approval: nodes={nodes}. Ask the user if they want to proceed, then call execute_pending_topology.",
                )])
            return ToolResponse(content=[TextBlock(
                type="text",
                text="No topology has been executed in this session yet.",
            )])

        success = result.get("success", False)
        if success:
            completed = int(progress.get("completed_nodes", 0))
            total = int(progress.get("total_nodes", 0))
            progress_line = f"Progress: {completed}/{total} nodes completed\n" if total else ""
            return ToolResponse(content=[TextBlock(
                type="text",
                text=(
                    f"Last execution: SUCCESS\n"
                    f"Topology: {result.get('topology_id', 'unknown')}\n"
                    f"Latency: {result.get('latency_seconds', 0):.1f}s\n"
                    f"{progress_line}"
                    f"Output:\n{result.get('final_output', '')}"
                ),
            )])
        else:
            return ToolResponse(content=[TextBlock(
                type="text",
                text=(
                    f"Last execution: FAILED\n"
                    f"Error: {result.get('error', 'unknown error')}"
                ),
            )])

    toolkit.register_tool_function(get_execution_status)

    # ------------------------------------------------------------------
    # Topology persistence tools
    # ------------------------------------------------------------------

    if topology_store is not None:

        async def list_saved_topologies() -> ToolResponse:
            """List all saved topologies with their IDs and names.

            Call this FIRST before rerun_topology or load_topology when the user
            asks to rerun, edit, or reference a saved topology. Use the exact
            topology_id from this list — never guess an ID.

            Returns:
                A list of saved topologies with id, name, and creation date.
            """
            from datetime import datetime as _dt
            topologies = topology_store.list_topologies(user_id=session.user_id)
            if not topologies:
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text="No saved topologies found.",
                )])
            lines = []
            for t in topologies:
                created = _dt.fromtimestamp(t.created_at).strftime("%Y-%m-%d %H:%M")
                lines.append(
                    f"- topology_id: {t.topology_id!r} | name: {t.name!r} | "
                    f"created: {created} | version: {t.version}"
                )
            return ToolResponse(content=[TextBlock(
                type="text",
                text="Saved topologies:\n" + "\n".join(lines),
            )])

        async def load_topology(topology_id: str, version: int | None = None) -> ToolResponse:
            """Load a saved topology into session.pending_topology for editing or rerun."""
            stored = topology_store.get_topology(topology_id, version=version)
            if stored is None:
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text=f"Topology '{topology_id}' not found in store.",
                )])
            session.pending_topology = stored.spec
            session.pending_estimate = None
            return ToolResponse(content=[TextBlock(
                type="text",
                text=(
                    f"Loaded topology '{topology_id}' (v{stored.version}) into session. "
                    f"Name: '{stored.name}'. Ready to edit or rerun."
                ),
            )])

        async def persist_topology(topology_id: str, save_mode: str) -> ToolResponse:
            """Persist session.pending_topology to the topology store."""
            if session.pending_topology is None:
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text="No pending topology to save. Generate or load one first.",
                )])

            if save_mode not in {"overwrite", "new_version"}:
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text="Invalid save_mode. Use 'overwrite' or 'new_version'.",
                )])

            overwrite = save_mode == "overwrite"
            spec_to_save = dict(session.pending_topology)
            spec_to_save["topology_id"] = topology_id

            stored = topology_store.save_topology(
                spec=spec_to_save,
                user_id=session.user_id,
                overwrite=overwrite,
            )
            session.pending_topology = stored.spec
            return ToolResponse(content=[TextBlock(
                type="text",
                text=(
                    f"Topology saved: id='{stored.topology_id}', "
                    f"version={stored.version}, name='{stored.name}'."
                ),
            )])

        async def rerun_topology(
            topology_id: str,
            user_prompt: str | None = None,
        ) -> ToolResponse:
            """Load and execute a saved topology with optional prompt override."""
            from daap.executor.engine import execute_topology
            from daap.spec.resolver import resolve_topology
            from daap.spec.schema import TopologySpec

            stored = topology_store.get_topology(topology_id)
            if stored is None:
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text=f"Topology '{topology_id}' not found.",
                )])
            if stored.deleted_at is not None:
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text=f"Topology '{topology_id}' has been deleted and cannot be rerun.",
                )])

            prompt = user_prompt or stored.spec.get("user_prompt", "")

            try:
                spec = TopologySpec.model_validate(stored.spec)
                resolved = resolve_topology(spec)
                if isinstance(resolved, list):
                    errors = "; ".join(error.message for error in resolved)
                    return ToolResponse(content=[TextBlock(
                        type="text",
                        text=f"Topology resolution failed: {errors}",
                    )])

                result = await execute_topology(
                    resolved=resolved,
                    user_prompt=prompt,
                    tracker=session.token_tracker,
                    daap_memory=daap_memory,
                    user_id=session.user_id,
                    permission_fn=session._file_permission_fn,
                )

                result_payload = {
                    "topology_id": result.topology_id,
                    "final_output": result.final_output,
                    "success": result.success,
                    "error": result.error,
                    "latency_seconds": result.total_latency_seconds,
                    "total_input_tokens": result.total_input_tokens,
                    "total_output_tokens": result.total_output_tokens,
                }
                session.execution_result = result_payload

                topology_store.save_run(
                    topology_id=topology_id,
                    topology_version=stored.version,
                    user_id=session.user_id,
                    result=result_payload,
                    user_prompt=prompt,
                )

                if result.success:
                    return ToolResponse(content=[TextBlock(
                        type="text",
                        text=(
                            f"Rerun complete. Latency: {result.total_latency_seconds:.1f}s. "
                            f"Tokens: {result.total_input_tokens} in / {result.total_output_tokens} out.\n"
                            f"Output:\n{result.final_output}"
                        ),
                    )])

                return ToolResponse(content=[TextBlock(
                    type="text",
                    text=f"Rerun failed: {result.error}",
                )])
            except Exception as exc:
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text=f"Rerun error: {exc}",
                )])

        toolkit.register_tool_function(list_saved_topologies)
        toolkit.register_tool_function(load_topology)
        toolkit.register_tool_function(persist_topology)
        toolkit.register_tool_function(rerun_topology)

    # ------------------------------------------------------------------
    # execute_pending_topology — always registered
    # Lets the agent execute a newly-generated topology after getting
    # user approval via ask_user, without any hardcoded keyword detection.
    # ------------------------------------------------------------------

    async def execute_pending_topology() -> ToolResponse:
        """Execute the pending topology that was just approved by the user.

        Call this immediately after the user confirms they want to proceed
        via ask_user. Do NOT call this without user approval.

        Streams progress to the client (if connected via WebSocket) and
        returns the full execution result so you can summarise it.

        Returns:
            Execution result including output, latency, and token usage.
        """
        if session.pending_topology is None:
            return ToolResponse(content=[TextBlock(
                type="text",
                text="No pending topology to execute. Call delegate_to_architect first.",
            )])

        if session.is_executing:
            return ToolResponse(content=[TextBlock(
                type="text",
                text="A topology is already executing. Wait for it to finish.",
            )])

        from daap.spec.schema import TopologySpec
        from daap.spec.resolver import resolve_topology
        from daap.executor.engine import execute_topology as _execute_topology

        topo_dict = session.pending_topology
        topo_id = topo_dict.get("topology_id", "unknown")
        user_prompt = topo_dict.get("user_prompt", "")
        ws_send = session._ws_send  # None in CLI, set by ws_handler for WebSocket
        logger.info(
            "execute_pending_topology: requested session=%s topology=%s",
            session.session_id,
            topo_id,
        )

        try:
            # RL: recommend model tier per role via LinTS (non-fatal)
            if rl_optimizer is not None:
                try:
                    from daap.optimizer.integration import get_tier_recommendations
                    nodes = topo_dict.get("nodes", []) or []
                    roles = [n.get("role", "") for n in nodes if isinstance(n, dict)]
                    _has_parallel = any(
                        isinstance(n, dict) and
                        n.get("instance_config", {}).get("parallel_instances", 1) > 1
                        for n in nodes
                    )
                    recs = get_tier_recommendations(
                        user_id=session.user_id,
                        user_prompt=user_prompt,
                        proposed_roles=roles,
                        node_count=len(roles),
                        has_parallel=_has_parallel,
                    )
                    for node in nodes:
                        if not isinstance(node, dict):
                            continue
                        role = node.get("role", "")
                        if role in recs:
                            op_override = node.get("operator_override", {})
                            if not (isinstance(op_override, dict) and op_override.get("model_map")):
                                node["model_tier"] = recs[role]
                except Exception as _rl_exc:
                    logger.warning("RL override failed (non-fatal): %s", _rl_exc)

            spec = TopologySpec.model_validate(topo_dict)
            resolved = resolve_topology(spec)
            if isinstance(resolved, list):
                errors = "; ".join(e.message for e in resolved)
                logger.warning(
                    "execute_pending_topology: resolution failed topology=%s errors=%s",
                    topo_id,
                    errors,
                )
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text=f"Topology resolution failed: {errors}",
                )])

            total_nodes = len(resolved.nodes)
            logger.info(
                "execute_pending_topology: starting topology=%s nodes=%d order=%s",
                topo_id,
                total_nodes,
                resolved.execution_order,
            )
            session.is_executing = True
            if session_store is not None:
                session_store.set_executing(session.session_id, True)
            session.execution_progress = {
                "topology_id": topo_id,
                "status": "running",
                "total_nodes": total_nodes,
                "completed_nodes": 0,
                "remaining_nodes": total_nodes,
                "percent_complete": 0,
                "current_node": None,
            }

            if ws_send:
                await ws_send({
                    "type": "executing",
                    "topology_id": topo_id,
                    "total_nodes": total_nodes,
                })

            completed_nodes = 0

            def _on_node_start(node_id: str, model_id: str, step_num: int, total: int) -> None:
                logger.info(
                    "execute_pending_topology: node_start topology=%s node=%s model=%s step=%d/%d",
                    topo_id,
                    node_id,
                    model_id,
                    step_num,
                    total,
                )
                session.execution_progress.update({
                    "current_node": node_id,
                    "completed_nodes": completed_nodes,
                    "remaining_nodes": max(total_nodes - completed_nodes, 0),
                    "percent_complete": int((completed_nodes / total_nodes) * 100) if total_nodes else 0,
                })
                if ws_send:
                    import asyncio
                    asyncio.get_running_loop().create_task(ws_send({
                        "type": "progress",
                        "event": "node_start",
                        "topology_id": topo_id,
                        "node_id": node_id,
                        "model_id": model_id,
                        "step_num": step_num,
                        "total_steps": total,
                        "completed_nodes": completed_nodes,
                        "total_nodes": total_nodes,
                    }))

            def _on_node_complete(nr) -> None:
                nonlocal completed_nodes
                completed_nodes += 1
                logger.info(
                    "execute_pending_topology: node_complete topology=%s node=%s latency=%.3fs",
                    topo_id,
                    nr.node_id,
                    nr.latency_seconds,
                )
                pct = int((completed_nodes / total_nodes) * 100) if total_nodes else 100
                session.execution_progress.update({
                    "completed_nodes": completed_nodes,
                    "remaining_nodes": max(total_nodes - completed_nodes, 0),
                    "percent_complete": pct,
                    "current_node": nr.node_id,
                })
                # Memory: write agent diary entry (fire-and-forget)
                if daap_memory is not None:
                    node_role = next(
                        (n.get("role", "unknown") for n in topo_dict.get("nodes", [])
                         if isinstance(n, dict) and n.get("node_id") == nr.node_id),
                        "unknown",
                    )
                    
                    def _write_mem(r, out_text, lat, mod):
                        try:
                            daap_memory.remember_node_output(
                                user_id=session.user_id,
                                role=r,
                                node_output=out_text,
                                latency_seconds=lat,
                                model_used=mod,
                                success=True,
                            )
                        except Exception as exc:
                            logger.warning("Node memory write failed (non-fatal): %s", exc, exc_info=True)
                    
                    import asyncio
                    asyncio.get_running_loop().run_in_executor(
                        None, 
                        _write_mem,
                        node_role,
                        getattr(nr, "output_text", ""),
                        getattr(nr, "latency_seconds", 0.0),
                        getattr(nr, "model_id", "unknown")
                    )
                if ws_send:
                    import asyncio
                    asyncio.get_running_loop().create_task(ws_send({
                        "type": "progress",
                        "event": "node_complete",
                        "topology_id": topo_id,
                        "node_id": nr.node_id,
                        "completed_nodes": completed_nodes,
                        "total_nodes": total_nodes,
                        "percent_complete": pct,
                        "node_latency_seconds": nr.latency_seconds,
                    }))

            result = await _execute_topology(
                resolved=resolved,
                user_prompt=user_prompt,
                tracker=session.token_tracker,
                on_node_start=_on_node_start,
                on_node_complete=_on_node_complete,
                daap_memory=daap_memory,
                user_id=session.user_id,
                permission_fn=session._file_permission_fn,
            )
            logger.info(
                "execute_pending_topology: finished topology=%s success=%s latency=%.3fs error=%s",
                topo_id,
                result.success,
                result.total_latency_seconds,
                result.error,
            )

            session.execution_result = {
                "topology_id": result.topology_id,
                "final_output": result.final_output,
                "success": result.success,
                "error": result.error,
                "latency_seconds": result.total_latency_seconds,
                "total_input_tokens": result.total_input_tokens,
                "total_output_tokens": result.total_output_tokens,
            }

            # Write run summary to memory (non-fatal, always — not gated on success)
            if daap_memory is not None:
                def _write_run_mem(exec_res, topo_nodes):
                    try:
                        from daap.memory.writer import (
                            write_run_to_memory,
                            write_agent_learnings_from_run,
                        )
                        topology_summary = topo_dict.get("user_prompt", "topology run")[:120]
                        write_run_to_memory(
                            memory=daap_memory,
                            user_id=session.user_id,
                            topology_summary=topology_summary,
                            execution_result=exec_res,
                        )
                        write_agent_learnings_from_run(
                            memory=daap_memory,
                            execution_result=exec_res,
                            topology_nodes=topo_nodes,
                        )
                    except Exception as _mem_exc:
                        logger.warning("Memory write failed (non-fatal): %s", _mem_exc)
                
                import asyncio
                asyncio.get_running_loop().run_in_executor(None, _write_run_mem, result, topo_dict.get("nodes", []))

            # Auto-save topology + run
            if topology_store is not None:
                def _save_store(exec_res):
                    try:
                        saved = topology_store.save_topology(
                            spec=topo_dict,
                            user_id=session.user_id,
                            overwrite=True,
                        )
                        topology_store.save_run(
                            topology_id=saved.topology_id,
                            topology_version=saved.version,
                            user_id=session.user_id,
                            result=exec_res,
                            user_prompt=user_prompt,
                        )
                    except Exception as _save_exc:
                        logger.warning("Auto-save failed (non-fatal): %s", _save_exc)
                
                import asyncio
                asyncio.get_running_loop().run_in_executor(None, _save_store, session.execution_result)

            session.pending_topology = None
            session.pending_estimate = None

            if ws_send:
                await ws_send({
                    "type": "result",
                    "success": result.success,
                    "output": result.final_output,
                    "error": result.error,        # markdown trace on failure, None on success
                    "latency_seconds": result.total_latency_seconds,
                    "models_used": result.models_used,
                    "usage": {
                        "input_tokens": result.total_input_tokens,
                        "output_tokens": result.total_output_tokens,
                        "total_tokens": result.total_input_tokens + result.total_output_tokens,
                    },
                })

            if result.success:
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text=(
                        f"Execution complete. Ran in {result.total_latency_seconds:.1f}s. "
                        f"Tokens: {result.total_input_tokens} in / {result.total_output_tokens} out.\n"
                        f"Output:\n{result.final_output}"
                    ),
                )])
            else:
                return ToolResponse(content=[TextBlock(
                    type="text",
                    text=f"Execution failed: {result.error}",
                )])

        except asyncio.CancelledError:
            logger.info("execute_pending_topology cancelled for %s", topo_id)
            if ws_send:
                try:
                    import asyncio
                    asyncio.get_running_loop().create_task(ws_send({
                        "type": "result",
                        "success": False,
                        "output": "",
                        "error": "Execution was cancelled by the user.",
                    }))
                except Exception:
                    pass
            raise  # Let it bubble up to ws_handler
        except Exception as exc:
            logger.exception("execute_pending_topology tool failed for %s", topo_id)
            # Send failure to UI so it stops showing "executing"
            if ws_send:
                try:
                    await ws_send({
                        "type": "result",
                        "success": False,
                        "output": "",
                        "error": f"Internal error: {exc}",
                    })
                except Exception:
                    pass
            return ToolResponse(content=[TextBlock(
                type="text",
                text=f"Execution error: {exc}",
            )])
        finally:
            session.is_executing = False
            if session_store is not None:
                session_store.set_executing(session.session_id, False)

    toolkit.register_tool_function(execute_pending_topology)

    toolkit.set_architect_context = set_architect_context
    toolkit.get_last_topology_result = get_last_topology_result
    toolkit.clear_last_topology_result = clear_last_topology_result

    return toolkit
