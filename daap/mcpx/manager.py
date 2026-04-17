"""
DAAP MCP Manager — loads and manages MCP servers via AgentScope built-in clients.

This module integrates AgentScope's MCP client classes:
  - StdIOStatefulClient
  - HttpStatefulClient
  - HttpStatelessClient

It discovers MCP tools at startup and exposes them as DAAP tool IDs:
  - Preferred: mcp://server_name/tool_name
  - Alias:     mcp://server_name (only when a default tool is configured,
               or the server exposes exactly one tool)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentscope.mcp import (
    HttpStatefulClient,
    HttpStatelessClient,
    StdIOStatefulClient,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".daap" / "mcpx.json"
CONFIG_ENV_VAR = "DAAP_MCP_CONFIG_PATH"


@dataclass
class MCPServerSpec:
    """Normalized MCP server configuration."""

    name: str
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    cwd: str | None = None
    url: str | None = None
    transport: str = "streamable_http"
    headers: dict[str, str] | None = None
    timeout: float = 30.0
    sse_read_timeout: float = 300.0
    stateful: bool = True
    default_tool: str | None = None


class MCPManager:
    """Manages MCP server lifecycle and exposes discovered tool functions."""

    def __init__(self, config_path: str | None = None):
        cfg = config_path or os.environ.get(CONFIG_ENV_VAR, "")
        self._config_path = Path(cfg).expanduser() if cfg else DEFAULT_CONFIG_PATH

        self._clients: dict[str, Any] = {}
        self._connected_order: list[str] = []
        self._tool_registry: dict[str, Any] = {}
        self._tool_metadata: list[dict[str, Any]] = []
        self._service_aliases: dict[str, str] = {}

        self._started = False
        self._lock = asyncio.Lock()

    async def start_all(self) -> None:
        """Start all configured MCP servers and discover their tools."""
        async with self._lock:
            if self._started:
                return

            specs = self._load_specs(self._config_path)
            if not specs:
                self._started = True
                return

            for spec in specs:
                await self._start_one(spec)

            self._started = True

    async def stop_all(self) -> None:
        """Stop all started MCP servers (LIFO for stateful stdio safety)."""
        async with self._lock:
            if not self._started and not self._connected_order:
                return

            for name in reversed(self._connected_order):
                client = self._clients.get(name)
                if client is None:
                    continue

                close = getattr(client, "close", None)
                if close is None:
                    continue

                try:
                    await self._maybe_await(close(ignore_errors=True))
                except Exception as exc:
                    logger.warning("MCP close failed for server '%s': %s", name, exc)

            self._clients.clear()
            self._connected_order.clear()
            self._tool_registry.clear()
            self._tool_metadata.clear()
            self._service_aliases.clear()
            self._started = False

    def list_connected(self) -> list[str]:
        """Return currently connected/active server names."""
        return list(self._connected_order)

    async def list_all_tools(self) -> list[dict[str, Any]]:
        """Return discovered MCP tools with DAAP IDs and descriptions."""
        # API kept async for compatibility with existing chat command path.
        return sorted(
            [dict(item) for item in self._tool_metadata],
            key=lambda t: t["name"],
        )

    def get_tool_registry_entries(self) -> dict[str, Any]:
        """Return DAAP tool ID -> callable mapping for discovered MCP tools."""
        return dict(self._tool_registry)

    def get_available_tool_names(self) -> set[str]:
        """Return all discovered MCP tool IDs (including aliases)."""
        return set(self._tool_registry.keys())

    async def _start_one(self, spec: MCPServerSpec) -> None:
        """Start one MCP server and load its tools."""
        try:
            client = self._build_client(spec)
            self._clients[spec.name] = client

            connect = getattr(client, "connect", None)
            if connect is not None:
                await self._maybe_await(connect())

            tools = await self._maybe_await(client.list_tools())
            self._connected_order.append(spec.name)

            discovered_ids: list[str] = []
            for tool in tools:
                tool_name = str(getattr(tool, "name", "") or "").strip()
                if not tool_name:
                    continue

                func = await self._maybe_await(
                    client.get_callable_function(tool_name, wrap_tool_result=True),
                )

                tool_id = f"mcp://{spec.name}/{tool_name}"
                self._tool_registry[tool_id] = func
                discovered_ids.append(tool_id)

                description = str(getattr(tool, "description", "") or "")
                self._tool_metadata.append(
                    {
                        "name": tool_id,
                        "server": spec.name,
                        "tool_name": tool_name,
                        "description": description,
                    },
                )

            self._maybe_register_service_alias(spec, discovered_ids)
            logger.info(
                "MCP server '%s' connected with %d tools",
                spec.name,
                len(discovered_ids),
            )

        except Exception as exc:
            logger.warning("MCP server '%s' failed to start: %s", spec.name, exc)

    def _maybe_register_service_alias(
        self,
        spec: MCPServerSpec,
        discovered_ids: list[str],
    ) -> None:
        """
        Register service-level alias (mcp://server) when unambiguous.

        Alias rules:
        - If default_tool is configured and present, alias points to it.
        - Else if exactly one tool exists, alias points to that single tool.
        """
        alias_id = f"mcp://{spec.name}"

        target_id: str | None = None
        if spec.default_tool:
            candidate = f"mcp://{spec.name}/{spec.default_tool}"
            if candidate in self._tool_registry:
                target_id = candidate

        if target_id is None and len(discovered_ids) == 1:
            target_id = discovered_ids[0]

        if target_id is None:
            return

        self._tool_registry[alias_id] = self._tool_registry[target_id]
        self._service_aliases[alias_id] = target_id

        self._tool_metadata.append(
            {
                "name": alias_id,
                "server": spec.name,
                "tool_name": spec.default_tool or discovered_ids[0].split("/", 3)[-1],
                "description": f"Alias of {target_id}",
            },
        )

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        """Await values when needed, pass through plain values unchanged."""
        if inspect.isawaitable(value):
            return await value
        return value

    def _build_client(self, spec: MCPServerSpec) -> Any:
        """Build an AgentScope MCP client from normalized spec."""
        if spec.command:
            return StdIOStatefulClient(
                name=spec.name,
                command=spec.command,
                args=spec.args or [],
                env=spec.env,
                cwd=spec.cwd,
            )

        if spec.url:
            shared_kwargs = {
                "name": spec.name,
                "transport": spec.transport,
                "url": spec.url,
                "headers": spec.headers,
                "timeout": spec.timeout,
                "sse_read_timeout": spec.sse_read_timeout,
            }
            if spec.stateful:
                return HttpStatefulClient(**shared_kwargs)
            return HttpStatelessClient(**shared_kwargs)

        raise ValueError(
            f"Server '{spec.name}' is invalid. Expected stdio (command) or http (url) config.",
        )

    @classmethod
    def _load_specs(cls, path: Path) -> list[MCPServerSpec]:
        """
        Load config from ~/.daap/mcpx.json (or DAAP_MCP_CONFIG_PATH).

        Supported formats:
        1) Official MCP style:
           {
             "mcpServers": {
               "linkedin": {"command": "...", "args": [...], "env": {...}}
             }
           }

        2) DAAP explicit list:
           {
             "servers": [
               {
                 "name": "linkedin",
                 "command": "...",
                 "args": [...],
                 "default_tool": "search_people"
               }
             ]
           }
        """
        if not path.exists():
            return []

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed reading MCP config '%s': %s", path, exc)
            return []

        specs: list[MCPServerSpec] = []

        # Format 1: mcpServers object
        mcp_servers = raw.get("mcpServers") if isinstance(raw, dict) else None
        if isinstance(mcp_servers, dict):
            for name, cfg in mcp_servers.items():
                if not isinstance(cfg, dict):
                    continue
                spec = cls._normalize_server_dict({"name": name, **cfg})
                if spec:
                    specs.append(spec)
            return specs

        # Format 2: servers array
        servers = raw.get("servers") if isinstance(raw, dict) else None
        if isinstance(servers, list):
            for cfg in servers:
                if not isinstance(cfg, dict):
                    continue
                spec = cls._normalize_server_dict(cfg)
                if spec:
                    specs.append(spec)
            return specs

        logger.warning(
            "MCP config '%s' has no supported format. Expected 'mcpServers' or 'servers'.",
            path,
        )
        return []

    @staticmethod
    def _normalize_server_dict(cfg: dict[str, Any]) -> MCPServerSpec | None:
        """Normalize one server dictionary into MCPServerSpec."""
        name = str(cfg.get("name", "") or "").strip()
        if not name:
            return None

        command = cfg.get("command")
        url = cfg.get("url")

        args = cfg.get("args")
        if not isinstance(args, list):
            args = []

        env = cfg.get("env")
        if isinstance(env, dict):
            env = {str(k): os.path.expandvars(str(v)) for k, v in env.items()}
        else:
            env = None

        cwd = cfg.get("cwd")
        if cwd is not None:
            cwd = os.path.expandvars(str(cwd))

        headers = cfg.get("headers")
        if isinstance(headers, dict):
            headers = {str(k): str(v) for k, v in headers.items()}
        else:
            headers = None

        transport = str(cfg.get("transport", "streamable_http") or "streamable_http")
        stateful = bool(cfg.get("stateful", True))
        timeout = float(cfg.get("timeout", 30.0) or 30.0)
        sse_read_timeout = float(cfg.get("sse_read_timeout", 300.0) or 300.0)
        default_tool = cfg.get("default_tool")
        if default_tool is not None:
            default_tool = str(default_tool).strip() or None

        if command is not None:
            command = os.path.expandvars(str(command))
        if url is not None:
            url = str(url)

        return MCPServerSpec(
            name=name,
            command=command,
            args=[str(a) for a in args],
            env=env,
            cwd=cwd,
            url=url,
            transport=transport,
            headers=headers,
            timeout=timeout,
            sse_read_timeout=sse_read_timeout,
            stateful=stateful,
            default_tool=default_tool,
        )


_MCP_MANAGER: MCPManager | None = None


def get_mcp_manager(config_path: str | None = None) -> MCPManager:
    """Return singleton MCPManager instance."""
    global _MCP_MANAGER

    if _MCP_MANAGER is None:
        _MCP_MANAGER = MCPManager(config_path=config_path)
    return _MCP_MANAGER
