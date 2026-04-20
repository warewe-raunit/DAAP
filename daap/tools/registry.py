"""
DAAP Tool Registry — maps abstract tool names to concrete async tool functions.

web_search: DuckDuckGo (free, no API key required)
web_fetch:  httpx + BeautifulSoup (text extraction from HTML)
read_file, write_file, code_execution: local ops

This is the SINGLE SOURCE OF TRUTH for available tools.
Validator uses keys to check tool availability.
Node builder uses values to register tools with AgentScope.
"""

import asyncio
import logging
import re
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)
from bs4 import BeautifulSoup
from agentscope.tool import ToolResponse
from agentscope.message import TextBlock


def _is_inside_cwd(filepath: str, cwd: Path) -> tuple[bool, Path]:
    """Resolve filepath and check if it lives inside cwd. Returns (inside, resolved)."""
    try:
        resolved = Path(filepath).resolve()
    except Exception:
        return False, Path(filepath)
    try:
        resolved.relative_to(cwd.resolve())
        return True, resolved
    except ValueError:
        return False, resolved


# ---------------------------------------------------------------------------
# web_search — DuckDuckGo, no API key
# ---------------------------------------------------------------------------

async def web_search(query: str, max_results: int = 8) -> ToolResponse:
    """Search the web using DuckDuckGo and return structured results.

    USE FOR: any web search — Reddit posts, news, company research, LinkedIn
    profiles, product pages, job listings, competitor analysis.

    DO NOT USE FOR: reading the content of a page (use WebFetch for that).

    REDDIT PATTERN: include "site:reddit.com" in the query.
      Recency filter: append "after:YYYY-MM-DD" (e.g. "after:2026-04-13" for
      the past week). Today is injected into node system prompts by the master.

    ARGS:
        query:       full search string including any site: or after: operators
        max_results: number of results (default 8, max ~20)

    RETURNS: numbered list of results with title, URL, and snippet.
    """
    try:
        from ddgs import DDGS
        # DDGS.text() is synchronous — run in thread to avoid blocking
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: list(DDGS().text(query, max_results=max_results)),
        )

        if not results:
            return ToolResponse(content=[TextBlock(
                type="text",
                text=f"No results found for: {query}",
            )])

        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "").strip()
            href  = r.get("href", "").strip()
            body  = r.get("body", "").strip()[:200]
            lines.append(f"{i}. **{title}**\n   URL: {href}\n   {body}\n")

        return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])

    except Exception as exc:
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Search failed: {exc}",
        )])


# ---------------------------------------------------------------------------
# web_fetch — httpx + BeautifulSoup
# ---------------------------------------------------------------------------

_FETCH_TIMEOUT = 15  # seconds
_MAX_CONTENT_CHARS = 8000


def _html_to_text(html: str) -> str:
    """Strip HTML to readable plain text."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, nav, footer noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse whitespace
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


async def web_fetch(url: str) -> ToolResponse:
    """Fetch the full readable text content of a web page.

    USE FOR: reading a specific URL found via WebSearch — scraping a Reddit
    thread, reading a blog post, extracting data from a landing page.

    DO NOT USE FOR: searching (use WebSearch instead). Do not guess URLs;
    only fetch URLs you have already retrieved from WebSearch or the user.

    ARGS:
        url: full URL starting with http:// or https://

    RETURNS: plain text of the page, stripped of HTML, truncated to 8000 chars.
    """
    if not url.startswith(("http://", "https://")):
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Invalid URL (must start with http:// or https://): {url}",
        )])

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=_FETCH_TIMEOUT,
            verify=False,  # Windows cert store issues; acceptable for scraping
            headers={"User-Agent": "Mozilla/5.0 (compatible; DAAP/1.0)"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "html" in content_type:
            text = _html_to_text(response.text)
        else:
            text = response.text

        if len(text) > _MAX_CONTENT_CHARS:
            text = text[:_MAX_CONTENT_CHARS] + f"\n\n[Truncated — {len(text)} total chars]"

        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Content from {url}:\n\n{text}",
        )])

    except httpx.HTTPStatusError as exc:
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"HTTP {exc.response.status_code} fetching {url}",
        )])
    except Exception as exc:
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Fetch failed for {url}: {exc}",
        )])


# ---------------------------------------------------------------------------
# read_file / write_file / code_execution (local ops)
# ---------------------------------------------------------------------------

async def read_file(filepath: str) -> ToolResponse:
    """Read a local file and return its full text content.

    USE FOR: reading CSVs, JSON files, text inputs that the user has placed
    on disk — e.g. a lead list, a prompt template, configuration data.

    DO NOT USE FOR: fetching web pages (use WebFetch) or running computations
    (use CodeExecution).

    ARGS:
        filepath: absolute or relative path. Paths outside the working
                  directory require explicit user permission.

    RETURNS: raw file contents as text.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return ToolResponse(content=[TextBlock(type="text", text=content)])
    except Exception as exc:
        return ToolResponse(content=[TextBlock(type="text", text=f"Error reading file: {exc}")])


async def write_file(filepath: str, content: str) -> ToolResponse:
    """Write text content to a local file (creates or overwrites).

    USE FOR: persisting final output to disk — saving a CSV of leads, writing
    a report, exporting formatted emails or JSON results for downstream use.

    DO NOT USE FOR: reading files (use ReadFile), web requests, or temp
    scratch work that isn't meant to survive the node's execution.

    ARGS:
        filepath: absolute or relative path. Paths outside the working
                  directory require explicit user permission.
        content:  full text to write; overwrites existing file if present.

    RETURNS: confirmation message with the path written.
    """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResponse(content=[TextBlock(type="text", text=f"Written to {filepath}")])
    except Exception as exc:
        return ToolResponse(content=[TextBlock(type="text", text=f"Error writing file: {exc}")])


async def code_execution(code: str, language: str = "python") -> ToolResponse:
    """Execute Python code in a LOCAL subprocess sandbox.

    USE FOR: pure data-processing tasks — parsing JSON/CSV, deduplicating
    lists, sorting/filtering records, string transformations, arithmetic,
    reformatting structured data between nodes.

    DO NOT USE FOR: anything requiring network access. There is NO internet,
    NO requests/httpx/urllib, NO external APIs, NO Reddit/LinkedIn/web access
    inside this sandbox. Using it for web calls will always raise NameError or
    ImportError. Use WebSearch or WebFetch for any network operation.

    AVAILABLE: Python stdlib only (json, csv, re, datetime, collections, etc.).
    NOT AVAILABLE: requests, httpx, pandas, numpy, openai, or any third-party lib.

    TIMEOUT: 10 seconds. Keep code fast; no long loops.

    ARGS:
        code:     Python source to execute (print() to return output)
        language: must be "python" (only supported language)

    RETURNS: stdout of the script (up to 2000 chars), or stderr on error.
    """
    if language != "python":
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Only Python is supported in Phase 1. Got: {language}",
        )])

    try:
        proc = await asyncio.create_subprocess_exec(
            "python", "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        out = stdout.decode()[:2000]
        err = stderr.decode()[:500]

        if proc.returncode == 0:
            return ToolResponse(content=[TextBlock(type="text", text=out or "(no output)")])
        else:
            return ToolResponse(content=[TextBlock(
                type="text",
                text=f"Execution error:\n{err}",
            )])
    except asyncio.TimeoutError:
        return ToolResponse(content=[TextBlock(type="text", text="Code execution timed out (10s limit)")])
    except Exception as exc:
        return ToolResponse(content=[TextBlock(type="text", text=f"Execution failed: {exc}")])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TOOL_REGISTRY: dict[str, callable] = {
    "agentscope.tools.WebSearch":     web_search,
    "agentscope.tools.WebFetch":      web_fetch,
    "agentscope.tools.ReadFile":      read_file,
    "agentscope.tools.WriteFile":     write_file,
    "agentscope.tools.CodeExecution": code_execution,
}

_ABSTRACT_TO_RESOLVED: dict[str, str] = {
    "WebSearch":     "agentscope.tools.WebSearch",
    "WebFetch":      "agentscope.tools.WebFetch",
    "ReadFile":      "agentscope.tools.ReadFile",
    "WriteFile":     "agentscope.tools.WriteFile",
    "CodeExecution": "agentscope.tools.CodeExecution",
}

# Backward-compatible MCP placeholders so topology generation remains stable
# even when no MCP servers are connected at startup.
_MCP_PLACEHOLDER_TOOLS: set[str] = {"mcp://linkedin", "mcp://crunchbase"}


def _get_mcp_manager_safe():
    """Return MCP manager when available, otherwise None (never raises)."""
    try:
        from daap.mcpx.manager import get_mcp_manager
        return get_mcp_manager()
    except Exception as exc:
        logger.debug("MCP manager unavailable: %s", exc)
        return None


def get_tool_registry(
    cwd: Path | None = None,
    permission_fn=None,
) -> dict[str, callable]:
    """Returns resolved_tool_id → async function mapping.

    cwd:           working directory; file ops inside it are auto-allowed.
    permission_fn: async (filepath: str, operation: str) -> bool
                   called for paths outside cwd. If None, outside-cwd ops are blocked.
    """
    registry = dict(_TOOL_REGISTRY)

    mcp_manager = _get_mcp_manager_safe()
    if mcp_manager is not None:
        try:
            registry.update(mcp_manager.get_tool_registry_entries())
        except Exception as exc:
            logger.warning("MCP tool registry entries unavailable (non-fatal): %s", exc, exc_info=True)

    if cwd is not None:
        _cwd = cwd.resolve()

        async def _guarded_read(filepath: str) -> ToolResponse:
            inside, resolved = _is_inside_cwd(filepath, _cwd)
            if not inside:
                if permission_fn is None:
                    return ToolResponse(content=[TextBlock(
                        type="text",
                        text=f"Access denied: '{resolved}' is outside the working directory.",
                    )])
                granted = await permission_fn(str(resolved), "read")
                if not granted:
                    return ToolResponse(content=[TextBlock(
                        type="text",
                        text=f"Permission denied by user for read: '{resolved}'.",
                    )])
            return await read_file(filepath)

        async def _guarded_write(filepath: str, content: str) -> ToolResponse:
            inside, resolved = _is_inside_cwd(filepath, _cwd)
            if not inside:
                if permission_fn is None:
                    return ToolResponse(content=[TextBlock(
                        type="text",
                        text=f"Access denied: '{resolved}' is outside the working directory.",
                    )])
                granted = await permission_fn(str(resolved), "write")
                if not granted:
                    return ToolResponse(content=[TextBlock(
                        type="text",
                        text=f"Permission denied by user for write: '{resolved}'.",
                    )])
            return await write_file(filepath, content)

        registry["agentscope.tools.ReadFile"] = _guarded_read
        registry["agentscope.tools.WriteFile"] = _guarded_write

    return registry


def get_tool_descriptions() -> str:
    """Return a formatted description block for all built-in tools.

    Built from each tool function's docstring so descriptions stay in sync
    with the implementation. Used by the master agent system prompt.
    """
    # abstract name → implementation function
    _BUILTIN_FUNCS = {
        "WebSearch":     web_search,
        "WebFetch":      web_fetch,
        "ReadFile":      read_file,
        "WriteFile":     write_file,
        "CodeExecution": code_execution,
    }
    lines = []
    for abstract_name, fn in _BUILTIN_FUNCS.items():
        doc = (fn.__doc__ or "").strip()
        lines.append(f"**{abstract_name}**\n{doc}")
    return "\n\n".join(lines)


def get_available_tool_names() -> set[str]:
    """Returns abstract tool names for the validator and master agent prompt."""
    names = set(_ABSTRACT_TO_RESOLVED.keys()) | set(_MCP_PLACEHOLDER_TOOLS)

    mcp_manager = _get_mcp_manager_safe()
    if mcp_manager is not None:
        try:
            names.update(mcp_manager.get_available_tool_names())
        except Exception:
            pass

    return names
