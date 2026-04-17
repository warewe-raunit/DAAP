"""Tests for DAAP MCP manager and registry integration."""

import json

import pytest


class _FakeTool:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description


class _FakeStdIOClient:
    def __init__(self, name, command, args=None, env=None, cwd=None, **kwargs):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = cwd
        self._connected = False

    async def connect(self):
        self._connected = True

    async def close(self, ignore_errors=True):
        self._connected = False

    async def list_tools(self):
        return [_FakeTool("search_people", "Search LinkedIn profiles")]

    async def get_callable_function(self, func_name, wrap_tool_result=True):
        async def _tool(**kwargs):
            return {"tool": func_name, "kwargs": kwargs}

        _tool.__name__ = func_name
        return _tool


@pytest.mark.asyncio
async def test_mcp_manager_loads_stdio_servers_from_mcpservers_config(monkeypatch, tmp_path):
    from daap.mcpx.manager import MCPManager
    import daap.mcpx.manager as mcp_manager_module

    config_path = tmp_path / "mcpx.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "linkedin": {
                        "command": "fake-mcp-server",
                        "args": ["--stdio"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mcp_manager_module, "StdIOStatefulClient", _FakeStdIOClient)

    manager = MCPManager(config_path=str(config_path))
    await manager.start_all()
    try:
        assert manager.list_connected() == ["linkedin"]

        names = manager.get_available_tool_names()
        assert "mcp://linkedin/search_people" in names
        # Alias is registered because exactly one tool is available.
        assert "mcp://linkedin" in names

        tools = await manager.list_all_tools()
        assert any(t["name"] == "mcp://linkedin/search_people" for t in tools)
    finally:
        await manager.stop_all()


@pytest.mark.asyncio
async def test_tool_registry_includes_dynamic_mcp_tools(monkeypatch):
    from daap.tools.registry import get_tool_registry, get_available_tool_names
    import daap.tools.registry as registry_module

    async def _fake_tool(**kwargs):
        return {"ok": True, "kwargs": kwargs}

    class _FakeManager:
        def get_tool_registry_entries(self):
            return {"mcp://linkedin/search_people": _fake_tool}

        def get_available_tool_names(self):
            return {"mcp://linkedin/search_people"}

    monkeypatch.setattr(registry_module, "_get_mcp_manager_safe", lambda: _FakeManager())

    registry = get_tool_registry()
    assert "agentscope.tools.WebSearch" in registry
    assert "mcp://linkedin/search_people" in registry

    available = get_available_tool_names()
    assert "WebSearch" in available
    assert "mcp://linkedin/search_people" in available
