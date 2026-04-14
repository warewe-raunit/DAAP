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
import re

import httpx
from bs4 import BeautifulSoup
from agentscope.tool import ToolResponse
from agentscope.message import TextBlock


# ---------------------------------------------------------------------------
# web_search — DuckDuckGo, no API key
# ---------------------------------------------------------------------------

async def web_search(query: str, max_results: int = 8) -> ToolResponse:
    """Search the web for the given query and return structured results.

    Args:
        query:       the search query string
        max_results: maximum number of results to return (default 8)

    Returns:
        Numbered list of results with title, URL, and snippet.
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
    """Fetch the content of a web page and return readable plain text.

    Args:
        url: the full URL to fetch (must start with http:// or https://)

    Returns:
        Plain text content of the page, truncated to 8000 characters.
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
    """Read a local file and return its content.

    Args:
        filepath: absolute or relative path to the file
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return ToolResponse(content=[TextBlock(type="text", text=content)])
    except Exception as exc:
        return ToolResponse(content=[TextBlock(type="text", text=f"Error reading file: {exc}")])


async def write_file(filepath: str, content: str) -> ToolResponse:
    """Write content to a local file.

    Args:
        filepath: absolute or relative path to write to
        content:  text content to write
    """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResponse(content=[TextBlock(type="text", text=f"Written to {filepath}")])
    except Exception as exc:
        return ToolResponse(content=[TextBlock(type="text", text=f"Error writing file: {exc}")])


async def code_execution(code: str, language: str = "python") -> ToolResponse:
    """Execute code in a subprocess sandbox.

    Args:
        code:     the code to execute
        language: programming language (currently only 'python' supported)
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


def get_tool_registry() -> dict[str, callable]:
    """Returns resolved_tool_id → async function mapping."""
    return dict(_TOOL_REGISTRY)


def get_available_tool_names() -> set[str]:
    """Returns abstract tool names for the validator and master agent prompt."""
    abstract = set(_ABSTRACT_TO_RESOLVED.keys())
    mcp_tools = {"mcp://linkedin", "mcp://crunchbase", "mcp://twitter"}
    return abstract | mcp_tools
