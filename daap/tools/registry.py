"""
DAAP Tool Registry — maps abstract tool names to concrete async tool functions.

web_search:  DuckDuckGo (free, no API key required)
web_fetch:   Jina Reader primary (JS-capable), httpx+BS4 fallback
deep_crawl:  Crawl4AI multi-page extraction (optional dep)
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
# web_fetch — Jina Reader primary, httpx + BeautifulSoup fallback
# ---------------------------------------------------------------------------

_FETCH_TIMEOUT = 20  # seconds
_MAX_CONTENT_CHARS = 20_000
_JINA_BASE = "https://r.jina.ai/"


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


async def _fetch_via_jina(url: str) -> str | None:
    """Try Jina Reader; return clean markdown or None on failure."""
    import os
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; DAAP/1.0)",
        "X-Return-Format": "markdown",
        "X-Timeout": str(_FETCH_TIMEOUT),
    }
    api_key = os.environ.get("JINA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=_FETCH_TIMEOUT + 5,
            verify=False,
        ) as client:
            resp = await client.get(f"{_JINA_BASE}{url}", headers=headers)
            if resp.status_code == 200 and resp.text.strip():
                return resp.text
    except Exception:
        pass
    return None


async def _fetch_via_httpx(url: str) -> str:
    """Raw httpx fetch with BeautifulSoup text extraction."""
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=_FETCH_TIMEOUT,
        verify=False,
        headers={"User-Agent": "Mozilla/5.0 (compatible; DAAP/1.0)"},
    ) as client:
        response = await client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "html" in content_type:
        return _html_to_text(response.text)
    return response.text


async def web_fetch(url: str) -> ToolResponse:
    """Fetch the full readable text content of a web page.

    USE FOR: reading a specific URL found via WebSearch — scraping a Reddit
    thread, reading a blog post, extracting data from a landing page,
    reading JavaScript-heavy sites, fetching PDFs.

    DO NOT USE FOR: searching (use WebSearch instead). Do not guess URLs;
    only fetch URLs you have already retrieved from WebSearch or the user.
    For crawling multiple linked pages use DeepCrawl instead.

    ARGS:
        url: full URL starting with http:// or https://

    RETURNS: clean markdown text of the page, truncated to 20000 chars.
             Uses Jina Reader for JS-rendered pages; falls back to direct fetch.
    """
    if not url.startswith(("http://", "https://")):
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Invalid URL (must start with http:// or https://): {url}",
        )])

    try:
        # Primary: Jina Reader handles JS-rendered pages, paywalls, PDFs
        text = await _fetch_via_jina(url)
        source = "Jina Reader"

        if not text:
            # Fallback: direct httpx + BeautifulSoup
            text = await _fetch_via_httpx(url)
            source = "direct fetch"

        if len(text) > _MAX_CONTENT_CHARS:
            text = text[:_MAX_CONTENT_CHARS] + f"\n\n[Truncated — {len(text)} total chars]"

        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Content from {url} (via {source}):\n\n{text}",
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
# deep_crawl — Crawl4AI multi-page extraction
# ---------------------------------------------------------------------------

_DEEP_CRAWL_MAX_PAGES = 10
_DEEP_CRAWL_MAX_CHARS = 40_000


async def deep_crawl(start_url: str, max_pages: int = 5, query: str = "") -> ToolResponse:
    """Crawl a website and extract clean content from multiple linked pages.

    USE FOR: extracting structured content from documentation sites, company
    websites, news sites, or any multi-page resource where WebFetch covers
    only one page. Handles JavaScript-rendered pages, shadow DOM, anti-bot
    measures, and consent popups automatically.

    DO NOT USE FOR: single-page reads (use WebFetch). Do not use for general
    search (use WebSearch). Only call with URLs you have from WebSearch or the user.

    REQUIRES: crawl4ai installed (`pip install crawl4ai && crawl4ai-setup`).
    Falls back gracefully with install instructions if not available.

    ARGS:
        start_url: root URL to begin crawling (http:// or https://)
        max_pages: max pages to crawl from start_url (default 5, max 10)
        query:     optional topic hint to filter crawled content (e.g. "pricing")

    RETURNS: LLM-ready markdown from all crawled pages, separated by dividers.
    """
    if not start_url.startswith(("http://", "https://")):
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Invalid URL (must start with http:// or https://): {start_url}",
        )])

    max_pages = min(max_pages, _DEEP_CRAWL_MAX_PAGES)

    try:
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig  # type: ignore
    except ImportError:
        return ToolResponse(content=[TextBlock(
            type="text",
            text=(
                "crawl4ai not installed. Install it with:\n"
                "  pip install crawl4ai\n"
                "  crawl4ai-setup\n\n"
                "Falling back to WebFetch for single-page extraction."
            ),
        )])

    try:
        sections: list[str] = []

        async with AsyncWebCrawler() as crawler:
            # Single-page mode — one reliable crawl of start_url
            config = CrawlerRunConfig(
                word_count_threshold=10,
                exclude_external_links=False,
                remove_overlay_elements=True,
                process_iframes=True,
            )
            result = await crawler.arun(url=start_url, config=config)
            if result.success and result.markdown:
                md = result.markdown.strip()
                if query:
                    # Filter to paragraphs/sections mentioning the query
                    lines = md.splitlines()
                    relevant = [
                        ln for ln in lines
                        if query.lower() in ln.lower() or ln.startswith("#")
                    ]
                    md = "\n".join(relevant) if relevant else md
                sections.append(f"## {start_url}\n\n{md}")

            # Follow internal links up to max_pages - 1
            if max_pages > 1 and result.success and result.links:
                internal = [
                    lnk.get("href", "") for lnk in result.links.get("internal", [])
                    if lnk.get("href", "").startswith(("http://", "https://"))
                ][:max_pages - 1]

                for link_url in internal:
                    try:
                        sub = await crawler.arun(url=link_url, config=config)
                        if sub.success and sub.markdown:
                            md = sub.markdown.strip()
                            if query:
                                lines = md.splitlines()
                                relevant = [
                                    ln for ln in lines
                                    if query.lower() in ln.lower() or ln.startswith("#")
                                ]
                                md = "\n".join(relevant) if relevant else md
                            sections.append(f"## {link_url}\n\n{md}")
                    except Exception as link_exc:
                        logger.debug("deep_crawl sub-page failed %s: %s", link_url, link_exc)

        if not sections:
            return ToolResponse(content=[TextBlock(
                type="text",
                text=f"No content extracted from {start_url}. Try WebFetch for this URL instead.",
            )])

        combined = "\n\n---\n\n".join(sections)
        if len(combined) > _DEEP_CRAWL_MAX_CHARS:
            combined = combined[:_DEEP_CRAWL_MAX_CHARS] + f"\n\n[Truncated — {len(combined)} total chars]"

        page_word = "page" if len(sections) == 1 else "pages"
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Deep crawl of {start_url} ({len(sections)} {page_word}):\n\n{combined}",
        )])

    except Exception as exc:
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Deep crawl failed for {start_url}: {exc}",
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
    "agentscope.tools.DeepCrawl":     deep_crawl,
    "agentscope.tools.ReadFile":      read_file,
    "agentscope.tools.WriteFile":     write_file,
    "agentscope.tools.CodeExecution": code_execution,
}

_ABSTRACT_TO_RESOLVED: dict[str, str] = {
    "WebSearch":     "agentscope.tools.WebSearch",
    "WebFetch":      "agentscope.tools.WebFetch",
    "DeepCrawl":     "agentscope.tools.DeepCrawl",
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
        "DeepCrawl":     deep_crawl,
        "ReadFile":      read_file,
        "WriteFile":     write_file,
        "CodeExecution": code_execution,
    }
    lines = []
    for abstract_name, fn in _BUILTIN_FUNCS.items():
        doc = (fn.__doc__ or "").strip()
        lines.append(f"**{abstract_name}**\n{doc}")
    return "\n\n".join(lines)


def get_available_tool_names(include_mcp_placeholders: bool = True) -> set[str]:
    """Returns abstract tool names for validation and master-agent prompting."""
    names = set(_ABSTRACT_TO_RESOLVED.keys())
    if include_mcp_placeholders:
        names.update(_MCP_PLACEHOLDER_TOOLS)

    mcp_manager = _get_mcp_manager_safe()
    if mcp_manager is not None:
        try:
            names.update(mcp_manager.get_available_tool_names())
        except Exception:
            pass

    return names
