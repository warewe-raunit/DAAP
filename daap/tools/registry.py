"""
DAAP Tool Registry — maps abstract tool names to concrete async tool functions.

web_search:  DuckDuckGo via duckduckgo_search (free, no API key required)
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
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
if hasattr(sys.stderr, 'reconfigure'):
    try: sys.stderr.reconfigure(encoding='utf-8')
    except Exception: pass

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

async def web_search(
    query: str,
    max_results: int = 8,
    date_from: str | None = None,
) -> ToolResponse:
    """Search the web using DuckDuckGo and return structured results.

    USE FOR: general web search — news, company research, LinkedIn profiles,
    product pages, job listings, competitor analysis.

    DO NOT USE FOR: Reddit posts (use RedditSearch instead).
    DO NOT USE FOR: reading the content of a page (use WebFetch for that).

    TEMPORAL QUERIES — mandatory for any time-scoped request:
      If the task asks for "recent", "last N days/weeks/months", "this week",
      "posted after [date]", or any other recency constraint, you MUST pass
      date_from. Never rely on model knowledge for recent events — always search.

      Example: task says "posts from last 2 months", today is 2026-04-22
        → date_from="2026-02-22"

    ARGS:
        query:       full search string.
        max_results: number of results (default 8, max ~20)
        date_from:   ISO date string YYYY-MM-DD. Filters results to content
                     published on or after this date. Always pass for recency
                     queries — do not omit and then guess at result freshness.

    RETURNS: numbered list of results with title, URL, snippet, and the
             active date filter so you can verify recency.
    """
    from datetime import date as _date

    # Build date-aware query and DDG timelimit
    timelimit = None
    date_note = ""
    if date_from:
        # Append Google-style date operator — DDG passes it through as a filter
        query = f"{query} after:{date_from}"
        date_note = f" [date_from={date_from}]"
        # Map delta to DDG timelimit for a second layer of recency enforcement
        try:
            cutoff = _date.fromisoformat(date_from)
            delta = (_date.today() - cutoff).days
            if delta <= 1:
                timelimit = "d"
            elif delta <= 7:
                timelimit = "w"
            elif delta <= 31:
                timelimit = "m"
            elif delta <= 365:
                timelimit = "y"
            # > 1 year: no DDG timelimit available; after: operator still applies
        except ValueError:
            pass

    try:
        from duckduckgo_search import DDGS
        # DDGS.text() is synchronous — run in thread to avoid blocking
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: list(DDGS().text(query, max_results=max_results, timelimit=timelimit)),
        )

        if not results:
            no_result_msg = f"No results found for: {query}"
            if date_from:
                no_result_msg += (
                    f"\nDate filter active (after:{date_from}) — try a broader query "
                    f"or an earlier date_from if the topic is niche."
                )
            return ToolResponse(content=[TextBlock(type="text", text=no_result_msg)])

        lines = [f"Search results for: {query}{date_note}\n"]
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
_MAX_CONTENT_CHARS = 6_000   # ~1,500 tokens per fetch — prevents "lost in middle" context flood
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

    # Playwright requires create_subprocess_exec; SelectorEventLoop (Windows default)
    # doesn't support it. Run in a dedicated thread with ProactorEventLoop.
    def _run_crawl() -> list[str]:
        import asyncio as _asyncio
        loop = (
            _asyncio.ProactorEventLoop()
            if hasattr(_asyncio, "ProactorEventLoop")
            else _asyncio.new_event_loop()
        )
        _asyncio.set_event_loop(loop)
        try:
            async def _fetch() -> list[str]:
                sects: list[str] = []
                config = CrawlerRunConfig(
                    word_count_threshold=10,
                    exclude_external_links=False,
                    remove_overlay_elements=True,
                    process_iframes=True,
                )
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url=start_url, config=config)
                    if result.success and result.markdown:
                        md = result.markdown.strip()
                        if query:
                            lines = md.splitlines()
                            relevant = [
                                ln for ln in lines
                                if query.lower() in ln.lower() or ln.startswith("#")
                            ]
                            md = "\n".join(relevant) if relevant else md
                        sects.append(f"## {start_url}\n\n{md}")

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
                                    sects.append(f"## {link_url}\n\n{md}")
                            except Exception as link_exc:
                                logger.debug("deep_crawl sub-page failed %s: %s", link_url, link_exc)
                return sects

            return loop.run_until_complete(_fetch())
        finally:
            loop.close()
            _asyncio.set_event_loop(None)

    try:
        sections = await asyncio.get_event_loop().run_in_executor(None, _run_crawl)

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
# Reddit cross-call session state
# ---------------------------------------------------------------------------
# Why module-level: a single topology run typically calls RedditSearch from one
# node and BatchRedditFetch / RedditFetch from a downstream node. Without shared
# state, the same URL gets re-fetched, and overlapping search queries return the
# same posts again. These sets give the LLM a concrete "already seen / already
# fetched" signal it can read off the tool result, and feed the dedup_guard's
# rule of thumb ("if you have it, don't re-fetch").
#
# Lifetime: process. The engine clears them at the start of each topology run
# via `reset_reddit_session_state()` so cross-run leaks don't happen.

_REDDIT_SEARCH_RETURNED_URLS: set[str] = set()
# _REDDIT_CONTENT_CACHE is defined further down (next to reddit_batch_fetch);
# both share this same reset lifecycle.


def reset_reddit_session_state() -> None:
    """Clear cross-call Reddit state so a new topology run starts fresh.

    Called by the engine at the start of each execute_topology run.
    """
    _REDDIT_SEARCH_RETURNED_URLS.clear()
    _REDDIT_CONTENT_CACHE.clear()


# ---------------------------------------------------------------------------
# reddit_search — public Reddit JSON search, no auth
# ---------------------------------------------------------------------------

async def _reddit_search_via_crawl4ai(base: str, params: dict) -> dict | None:
    """Fallback search using Playwright to bypass Reddit blocking."""
    import json as _json
    from urllib.parse import urlencode
    try:
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig  # type: ignore
    except ImportError:
        return None

    search_url = f"{base}?{urlencode(params)}"
    run_cfg = CrawlerRunConfig(word_count_threshold=5, remove_overlay_elements=True)

    # Playwright needs create_subprocess_exec which SelectorEventLoop (Windows default) doesn't
    # support. Run in a thread with its own ProactorEventLoop to avoid NotImplementedError.
    def _run_in_thread():
        import asyncio as _asyncio
        loop = (
            _asyncio.ProactorEventLoop()
            if hasattr(_asyncio, "ProactorEventLoop")
            else _asyncio.new_event_loop()
        )
        _asyncio.set_event_loop(loop)
        try:
            async def _fetch():
                async with AsyncWebCrawler(verbose=False) as crawler:
                    return await crawler.arun(url=search_url, config=run_cfg)
            return loop.run_until_complete(_fetch())
        finally:
            loop.close()
            _asyncio.set_event_loop(None)

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _run_in_thread)
        raw = result.markdown or result.html or ""
        if raw:
            if "```" in raw:
                import re
                raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
            try:
                return _json.loads(raw)
            except (_json.JSONDecodeError, KeyError, IndexError):
                pass
    except Exception as exc:
        logger.debug("crawl4ai reddit search failed for %s: %s", search_url, exc)
    return None

async def reddit_search(
    query: str,
    subreddit: str | None = None,
    sort: str = "relevance",
    time_filter: str = "month",
    limit: int = 15,
) -> ToolResponse:
    """Search Reddit posts using Reddit's public JSON endpoint.

    USE FOR: finding Reddit post URLs that may be relevant to a product,
    niche, pain point, or promotion opportunity. This returns post URLs,
    titles, subreddit, score/comment counts, created_utc, and body previews.

    DO NOT USE FOR: reading full post content and comments. Use RedditFetch
    on each candidate URL before deciding whether a product can be promoted.

    ARGS:
        query: search terms, e.g. "email verification tool" or "SaaS feedback".
        subreddit: optional subreddit name without r/, e.g. "SaaS".
        sort: relevance, hot, top, new, or comments.
        time_filter: hour, day, week, month, year, or all.
        limit: number of posts to return, max 25.

    RETURNS: numbered candidate Reddit posts with canonical URL and preview.
    """
    if not query.strip():
        return ToolResponse(content=[TextBlock(type="text", text="query is required.")])

    sort = sort if sort in {"relevance", "hot", "top", "new", "comments"} else "relevance"
    time_filter = time_filter if time_filter in {"hour", "day", "week", "month", "year", "all"} else "month"
    limit = max(1, min(int(limit), 25))

    base = (
        f"https://www.reddit.com/r/{subreddit.strip().strip('/')}/search.json"
        if subreddit
        else "https://www.reddit.com/search.json"
    )
    params = {
        "q": query,
        "restrict_sr": "1" if subreddit else "0",
        "type": "link",
        "sort": sort,
        "t": time_filter,
        "limit": str(limit),
    }

    # Try crawl4ai first as primary (higher success rate for Reddit JSON API)
    data = await _reddit_search_via_crawl4ai(base, params)

    if not data:
        # Fall back to direct httpx if crawl4ai is unavailable or fails
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=_REDDIT_FETCH_TIMEOUT,
                verify=False,
                headers=_REDDIT_HEADERS,
            ) as client:
                resp = await client.get(base, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.debug("direct reddit search fallback failed: %s", exc)

    if not data:
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"RedditSearch failed (blocked or unreachable). Please try a different query.",
        )])

    children = data.get("data", {}).get("children", [])
    posts = [c.get("data", {}) for c in children if c.get("kind") == "t3"]
    if not posts:
        scope = f" in r/{subreddit}" if subreddit else ""
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"No Reddit posts found for '{query}'{scope}. Try broader terms.",
        )])

    new_urls: list[str] = []
    repeat_count = 0
    lines = [f"RedditSearch results for: {query}"]
    for idx, post in enumerate(posts, 1):
        permalink = post.get("permalink") or ""
        url = "https://www.reddit.com" + permalink if permalink.startswith("/") else post.get("url", "")
        preview = (post.get("selftext") or post.get("selftext_html") or "").strip().replace("\n", " ")
        if len(preview) > 500:
            preview = preview[:500] + "..."

        # Per-URL dedup tags so the LLM can see what it already has.
        tags: list[str] = []
        if url and url in _REDDIT_SEARCH_RETURNED_URLS:
            tags.append("ALREADY_SEEN_IN_PRIOR_SEARCH")
            repeat_count += 1
        if url and url in _REDDIT_CONTENT_CACHE:
            tags.append("ALREADY_FETCHED")
        if url:
            _REDDIT_SEARCH_RETURNED_URLS.add(url)
            new_urls.append(url)
        tag_str = f" [{' | '.join(tags)}]" if tags else ""

        lines.append(
            f"{idx}. {post.get('title', '').strip()}{tag_str}\n"
            f"   URL: {url}\n"
            f"   r/{post.get('subreddit', '')} | score: {post.get('score', 0)} | "
            f"comments: {post.get('num_comments', 0)} | created_utc: {int(post.get('created_utc', 0) or 0)}\n"
            f"   preview: {preview or '(no post body preview)'}\n"
        )

    # Hard signal: if half-or-more of returned URLs are repeats, tell the LLM
    # explicitly to stop searching and move to the fetch / consolidation step.
    if new_urls and repeat_count / len(new_urls) >= 0.5:
        lines.append(
            f"\n[STOP_SEARCHING_SIGNAL] {repeat_count}/{len(new_urls)} results were already returned "
            f"by a prior RedditSearch in this run. Do NOT issue another RedditSearch with a similar query — "
            f"call BatchRedditFetch / generate_response on the URLs you already have."
        )

    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


# ---------------------------------------------------------------------------
# reddit_fetch — direct Reddit .json (primary) → crawl4ai old.reddit (fallback)
# ---------------------------------------------------------------------------

_REDDIT_HEADERS = {
    "User-Agent": "DAAP:research:1.0 (research tool, non-commercial)",
    "Accept": "application/json",
}
_REDDIT_FETCH_TIMEOUT = 20
_REDDIT_MAX_COMMENTS = 5

def _extract_post_id(url: str) -> str | None:
    """Extract Reddit post ID from a reddit.com URL. Returns base36 ID or None."""
    m = re.search(r"/comments/([a-z0-9]+)/", url)
    return m.group(1) if m else None


def _to_reddit_json_url(url: str) -> str:
    """Convert any reddit.com post URL to its /.json form."""
    clean = re.sub(r"https?://(old\.|www\.)?reddit\.com", "https://www.reddit.com", url)
    clean = clean.split("?")[0].rstrip("/")
    return clean + "/.json"


def _to_old_reddit_url(url: str) -> str:
    """Convert any reddit.com post URL to old.reddit.com."""
    return re.sub(r"https?://(www\.)?reddit\.com", "https://old.reddit.com", url)


def _parse_reddit_json(data: list) -> str | None:
    """Parse Reddit .json API response into formatted text. Returns None if post removed."""
    try:
        post = data[0]["data"]["children"][0]["data"]
        selftext = post.get("selftext") or ""
        if selftext in ("[removed]", "[deleted]"):
            return None

        lines = [
            f"# {post.get('title', '')}",
            f"r/{post.get('subreddit', '')} | score: {post.get('score', 0)} | "
            f"comments: {post.get('num_comments', 0)} | created_utc: {int(post.get('created_utc', 0))}",
            "",
        ]
        if selftext:
            lines += ["## Post Body", selftext[:2000], ""]

        comments = data[1]["data"]["children"]
        valid = [
            c["data"] for c in comments
            if c.get("kind") == "t1"
            and c["data"].get("body") not in (None, "[removed]", "[deleted]")
        ]
        if valid:
            lines.append("## Top Comments")
            for c in valid[:_REDDIT_MAX_COMMENTS]:
                lines.append(f"[score:{c.get('score', 0)}] {c.get('body', '')[:500]}")

        return "\n".join(lines)
    except Exception:
        return None


async def _reddit_fetch_via_crawl4ai(url: str) -> str | None:
    """Primary: fetch Reddit .json endpoint through crawl4ai's Playwright browser.

    Using a real browser bypasses Reddit's rate-limiting and anti-bot checks
    that block plain httpx requests. Parses the JSON response for clean output.
    Falls back to old.reddit.com markdown if JSON parsing fails.
    Returns formatted text or None on failure / crawl4ai not installed.
    """
    import json as _json
    try:
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig  # type: ignore
    except ImportError:
        return None

    json_url = _to_reddit_json_url(url)
    old_url = _to_old_reddit_url(url)
    run_cfg = CrawlerRunConfig(word_count_threshold=5, remove_overlay_elements=True)

    # Playwright requires create_subprocess_exec; SelectorEventLoop (Windows default)
    # doesn't support it. Run in a dedicated thread with ProactorEventLoop.
    def _run_in_thread():
        import asyncio as _asyncio
        loop = (
            _asyncio.ProactorEventLoop()
            if hasattr(_asyncio, "ProactorEventLoop")
            else _asyncio.new_event_loop()
        )
        _asyncio.set_event_loop(loop)
        try:
            async def _fetch():
                async with AsyncWebCrawler(verbose=False) as crawler:
                    result = await crawler.arun(url=json_url, config=run_cfg)

                raw = result.markdown or result.html or ""
                if raw:
                    if "```" in raw:
                        raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
                    try:
                        data = _json.loads(raw)
                        parsed = _parse_reddit_json(data)
                        if parsed:
                            return parsed
                    except (_json.JSONDecodeError, KeyError, IndexError):
                        pass

                # JSON parse failed — fall back to old.reddit.com markdown
                async with AsyncWebCrawler(verbose=False) as crawler2:
                    r2 = await crawler2.arun(url=old_url, config=run_cfg)
                md = r2.markdown or ""
                if len(md) > 200:
                    return md[:_DEEP_CRAWL_MAX_CHARS]
                return None

            return loop.run_until_complete(_fetch())
        finally:
            loop.close()
            _asyncio.set_event_loop(None)

    try:
        return await asyncio.get_event_loop().run_in_executor(None, _run_in_thread)
    except Exception as exc:
        logger.debug("crawl4ai reddit fetch failed for %s: %s", url, exc)
    return None


async def _reddit_fetch_via_json(url: str) -> str | None:
    """Fetch Reddit .json directly with httpx and parse post body + comments."""
    json_url = _to_reddit_json_url(url)
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=_REDDIT_FETCH_TIMEOUT,
            verify=False,
            headers=_REDDIT_HEADERS,
        ) as client:
            resp = await client.get(json_url)
            resp.raise_for_status()
            data = resp.json()
        return _parse_reddit_json(data)
    except Exception as exc:
        logger.debug("direct reddit json fetch failed for %s: %s", url, exc)
    return None


async def reddit_fetch(url: str) -> ToolResponse:
    """Fetch a Reddit post's full content (body + top comments).

    USE FOR: reading the actual content of a specific Reddit post — the post
    body and top comments. Does NOT require a Reddit API key.
    ALWAYS use this instead of WebFetch for Reddit post URLs — WebFetch hits
    Reddit's JS login wall and returns only navigation markup.

    Fetch strategy:
      direct Reddit .json endpoint (structured, clean)
      └─ if blocked → crawl4ai browser → Reddit .json / old.reddit markdown

    DO NOT USE FOR: searching Reddit (use RedditSearch first, WebSearch fallback).
    Do not use for non-Reddit URLs (use WebFetch instead).

    ARGS:
        url: a reddit.com post URL (e.g. https://www.reddit.com/r/SaaS/comments/abc123/...)

    RETURNS: post title, body text, and top comments with scores.
    """
    if "reddit.com" not in url:
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Not a Reddit URL. Use WebFetch for non-Reddit URLs: {url}",
        )])

    if not _extract_post_id(url):
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Could not extract post ID from URL: {url}. URL must contain /comments/<id>/",
        )])

    cached = _REDDIT_CONTENT_CACHE.get(url)
    if cached:
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Reddit post content from {url} (via cache):\n\n{cached}",
        )])

    text = await _reddit_fetch_via_json(url)
    source = "reddit .json"

    if not text:
        text = await _reddit_fetch_via_crawl4ai(url)
        source = "crawl4ai"

    if not text:
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"POST_NOT_FOUND: Could not fetch post from any source. "
                 f"Post may be removed/private or all sources unavailable. URL: {url}",
        )])

    _REDDIT_CONTENT_CACHE[url] = text
    return ToolResponse(content=[TextBlock(
        type="text",
        text=f"Reddit post content from {url} (via {source}):\n\n{text}",
    )])


# Module-level Reddit content cache.
# Why module-level (not per-call): reddit_batch_fetch and reddit_fetch are called
# across multiple nodes in the same execution. Caching content by URL prevents
# re-fetching the same post when an upstream node already pulled it.
# The cache is process-lifetime; fresh runs of the API server start empty.
_REDDIT_CONTENT_CACHE: dict[str, str] = {}

# Per-section budget for batch fetch output. Total output stays under
# BATCH_REDDIT_MAX_TOTAL_CHARS by trimming each section AND skipping later
# URLs into a SKIPPED list rather than a silent truncation suffix.
_BATCH_REDDIT_PER_SECTION_CHARS = 1_500
_BATCH_REDDIT_MAX_TOTAL_CHARS = 60_000


async def reddit_batch_fetch(urls: str, max_posts: int = 30, slim_manifest: bool = True) -> ToolResponse:
    """Fetch multiple Reddit post URLs and return post metadata + content per post.

    USE FOR: turning a RedditSearch result list into grounded post content for
    downstream reasoning. Prefer this over many individual RedditFetch calls
    when the node receives multiple candidate URLs.

    DO NOT USE FOR: finding posts. Use RedditSearch first.

    ARGS:
        urls: raw text or JSON/list-like text containing reddit.com post URLs.
        max_posts: maximum URLs to fetch, capped at 40.
        slim_manifest: if True (default), emit a compact JSON manifest with
            metadata + 200-char body_preview per post instead of full bodies.
            Use slim_manifest=False only when the downstream node genuinely
            needs full post text (e.g. a summariser node). For filter/rank/
            promote topologies always use slim_manifest=True — it cuts
            downstream token cost by ~85% with no loss of evaluation signal.

    RETURNS: when slim_manifest=True, a JSON array of rows with fields:
             url, status, title, subreddit, score, num_comments, created_utc,
             body_preview (200 chars). status ∈ {ACTIVE, REJECTED}.
             when slim_manifest=False, the original full-body manifest with
             ACTIVE sections, REJECTED sections, and SKIPPED_DUE_TO_SIZE list.
             Downstream nodes MUST treat REJECTED / SKIPPED URLs as unread —
             never invent body text for them.
    """
    if not urls or not str(urls).strip():
        return ToolResponse(content=[TextBlock(type="text", text="No URLs provided.")])

    max_posts = max(1, min(int(max_posts), 40))
    found = []
    seen: set[str] = set()
    for match in re.finditer(r"https?://(?:www\.|old\.)?reddit\.com/r/[^\s<>\]\)\"']+/comments/[a-z0-9]+/[^\s<>\]\)\"']*", str(urls), re.I):
        url = match.group(0).rstrip(".,;")
        if url not in seen:
            seen.add(url)
            found.append(url)
        if len(found) >= max_posts:
            break

    if not found:
        return ToolResponse(content=[TextBlock(
            type="text",
            text="No Reddit post URLs found. Input must contain reddit.com/r/.../comments/<id>/ URLs.",
        )])

    sem = asyncio.Semaphore(4)

    async def _fetch_single(url: str) -> tuple[str, str | None, str]:
        """Returns (status, content_or_none, source). status ∈ {ACTIVE, REJECTED}."""
        if not _extract_post_id(url):
            return "REJECTED", None, "invalid_url"

        cached = _REDDIT_CONTENT_CACHE.get(url)
        if cached:
            return "ACTIVE", cached, "cache"

        async with sem:
            text = await _reddit_fetch_via_json(url)
            source = "reddit .json"
            if not text:
                text = await _reddit_fetch_via_crawl4ai(url)
                source = "crawl4ai"

        if not text:
            return "REJECTED", None, "post_not_found_or_unusable"

        _REDDIT_CONTENT_CACHE[url] = text
        return "ACTIVE", text, source

    tasks = [_fetch_single(url) for url in found]
    fetch_results = await asyncio.gather(*tasks)

    # -----------------------------------------------------------------------
    # slim_manifest=True path: compact JSON rows, one per post.
    # Each row: url, status, title, subreddit, score, num_comments,
    #           created_utc, body_preview (200 chars max).
    # For 30 posts: ~6 KB output vs ~45 KB for full-body mode — 85% reduction.
    # body_preview is sufficient for:
    #   • [removed] / [deleted] detection
    #   • bot/automod template detection ("I am a bot", "automatically performed")
    #   • metadata-based ranking (score, num_comments, created_utc)
    # -----------------------------------------------------------------------
    if slim_manifest:
        import re as _re
        import json as _json

        def _extract_meta(raw: str) -> dict:
            """Extract structured fields from fetched post text."""
            meta: dict = {}
            # Title (line starting with #)
            m = _re.search(r"^#\s+(.+)$", raw, _re.M)
            meta["title"] = m.group(1).strip() if m else ""
            # Subreddit
            m = _re.search(r"\br/([\w]+)\b", raw)
            meta["subreddit"] = m.group(1) if m else ""
            # score
            m = _re.search(r"score:\s*([0-9,]+)", raw)
            meta["score"] = int(m.group(1).replace(",", "")) if m else 0
            # num_comments
            m = _re.search(r"comments:\s*([0-9,]+)", raw)
            meta["num_comments"] = int(m.group(1).replace(",", "")) if m else 0
            # created_utc
            m = _re.search(r"created_utc:\s*([0-9]+)", raw)
            meta["created_utc"] = int(m.group(1)) if m else 0
            # body_preview: first 200 chars of Post Body section
            m = _re.search(r"##\s*Post Body\s*\n(.{1,200})", raw, _re.S)
            if m:
                preview_text = m.group(1).strip()[:200]
            else:
                # fallback: grab first non-header, non-metadata text block
                lines = [l for l in raw.splitlines()
                         if l.strip() and not l.startswith("#") and "score:" not in l]
                preview_text = " ".join(lines)[:200]
            meta["body_preview"] = preview_text
            return meta

        rows: list[dict] = []
        skipped_slim: list[str] = []

        for url, (status, content, source) in zip(found, fetch_results):
            if status == "REJECTED":
                rows.append({
                    "url": url,
                    "status": "REJECTED",
                    "title": "",
                    "subreddit": "",
                    "score": 0,
                    "num_comments": 0,
                    "created_utc": 0,
                    "body_preview": "",
                    "reject_reason": source,
                })
                continue

            meta = _extract_meta(content or "")
            rows.append({
                "url": url,
                "status": "ACTIVE",
                "title": meta.get("title", ""),
                "subreddit": meta.get("subreddit", ""),
                "score": meta.get("score", 0),
                "num_comments": meta.get("num_comments", 0),
                "created_utc": meta.get("created_utc", 0),
                "body_preview": meta.get("body_preview", ""),
            })

        manifest_json = _json.dumps(rows, ensure_ascii=False, indent=2)
        footer = ""
        if skipped_slim:
            footer = (
                "\n\nSKIPPED_DUE_TO_SIZE (not fetched — mark as POST_NOT_READ):\n"
                + "\n".join(f"- {u}" for u in skipped_slim)
            )
        output = (
            "BatchRedditFetch slim manifest (slim_manifest=True).\n"
            "Fields: url | status | title | subreddit | score | num_comments | created_utc | body_preview\n"
            "status ∈ {ACTIVE, REJECTED}. Evaluate ACTIVE rows only. "
            "REJECTED rows must be excluded. body_preview is ≤200 chars — "
            "sufficient for [removed]/[deleted]/bot-template detection.\n\n"
            f"{manifest_json}{footer}"
        )
        return ToolResponse(content=[TextBlock(type="text", text=output)])

    # -----------------------------------------------------------------------
    # Full body path (slim_manifest=False): original behaviour preserved.
    # Use only when the downstream node needs complete post text.
    # -----------------------------------------------------------------------
    sections: list[str] = []
    skipped_due_to_size: list[str] = []
    rejected_sections: list[str] = []
    total_chars = 0
    fixed_overhead = 200  # header + footer manifest text
    budget = _BATCH_REDDIT_MAX_TOTAL_CHARS - fixed_overhead

    for idx, (url, (status, content, source)) in enumerate(zip(found, fetch_results), 1):
        if status == "REJECTED":
            section = (
                f"## {idx}. REJECTED\n"
                f"URL: {url}\n"
                f"Reason: {source.upper()}. "
                f"Post may be removed, deleted, private, inaccessible, or empty."
            )
            if total_chars + len(section) <= budget:
                rejected_sections.append(section)
                total_chars += len(section)
            else:
                skipped_due_to_size.append(url)
            continue

        body = content or ""
        if len(body) > _BATCH_REDDIT_PER_SECTION_CHARS:
            body = body[:_BATCH_REDDIT_PER_SECTION_CHARS] + f"\n\n[Section trimmed — {len(content or '')} total chars]"
        section = (
            f"## {idx}. ACTIVE\n"
            f"URL: {url}\n"
            f"Source: {source}\n\n"
            f"{body}"
        )
        if total_chars + len(section) <= budget:
            sections.append(section)
            total_chars += len(section)
        else:
            skipped_due_to_size.append(url)

    parts = ["Batch Reddit fetch results:\n"]
    if sections:
        parts.append("\n\n---\n\n".join(sections))
    if rejected_sections:
        parts.append("\n\n---\n\n".join(rejected_sections))
    if skipped_due_to_size:
        parts.append(
            "## SKIPPED_DUE_TO_SIZE\n"
            "These URLs were NOT fetched into this response because the total "
            "content exceeded the output budget. Downstream nodes MUST mark these "
            "as 'POST_NOT_READ' — do NOT invent or summarize their content.\n"
            + "\n".join(f"- {u}" for u in skipped_due_to_size)
        )

    combined = "\n\n".join(parts)
    return ToolResponse(content=[TextBlock(type="text", text=combined)])




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


async def keywords_everywhere(keywords: str, country: str = "us", currency: str = "USD") -> ToolResponse:
    """Get search volume, CPC, and competition data for keywords via Keywords Everywhere API.

    USE FOR: ranking which keywords/topics have real search demand before deciding
    which Reddit posts are commercially valuable. Call this with product-related
    keywords to find which ones have highest monthly search volume, then use those
    high-volume keywords as RedditSearch queries for Category 3 posts.

    DO NOT USE FOR: discovering Reddit posts directly (use RedditSearch for that).
    DO NOT USE FOR: more than 100 keywords per call (API hard limit).

    ARGS:
        keywords: comma-separated keywords to look up, e.g. "email tool,saas feedback,crm software".
                  Max 100 keywords. Each keyword costs 1 credit.
        country:  2-letter country code for search volume data, e.g. "us", "gb", "in". Default "us".
        currency: 3-letter currency code for CPC data, e.g. "USD", "GBP". Default "USD".

    RETURNS: table of keyword | monthly_search_volume | cpc | competition score (0–1),
             sorted descending by search volume.
    """
    import os as _os
    api_key = _os.environ.get("KEYWORDS_EVERYWHERE_API_KEY", "").strip()
    if not api_key:
        return ToolResponse(content=[TextBlock(
            type="text",
            text="KEYWORDS_EVERYWHERE_API_KEY not set. Cannot call Keywords Everywhere API.",
        )])

    kw_list = [k.strip() for k in keywords.split(",") if k.strip()][:100]
    if not kw_list:
        return ToolResponse(content=[TextBlock(type="text", text="No keywords provided.")])

    def _call_ke_api() -> dict:
        import httpx as _httpx
        import urllib.parse as _urlparse
        pairs = [
            ("country", country),
            ("currency", currency),
            ("dataSource", "cli"),
        ] + [("kw[]", kw) for kw in kw_list]
        resp = _httpx.post(
            "https://api.keywordseverywhere.com/v1/get_keyword_data",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            content=_urlparse.urlencode(pairs),
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _call_ke_api)
    except Exception as exc:
        return ToolResponse(content=[TextBlock(type="text", text=f"Keywords Everywhere API error: {exc}")])

    data = result.get("data", [])
    if not data:
        return ToolResponse(content=[TextBlock(type="text", text="No data returned from Keywords Everywhere.")])

    rows = []
    for item in data:
        kw = item.get("keyword", "")
        vol = item.get("vol", 0) or 0
        cpc_val = (item.get("cpc") or {}).get("value", "0")
        comp = item.get("competition", 0) or 0
        rows.append((kw, int(vol), str(cpc_val), float(comp)))

    rows.sort(key=lambda x: x[1], reverse=True)

    lines = ["Keywords Everywhere results (sorted by search volume):"]
    lines.append(f"{'Keyword':<40} {'Vol/mo':>8} {'CPC':>8} {'Comp':>6}")
    lines.append("-" * 68)
    for kw, vol, cpc_val, comp in rows:
        lines.append(f"{kw:<40} {vol:>8,} {cpc_val:>8} {comp:>6.2f}")
    lines.append(f"\nCredits used: {result.get('credits_used', len(rows))}")

    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


async def keywords_everywhere_url_traffic(urls: str, country: str = "") -> ToolResponse:
    """Check how much organic Google traffic a list of URLs gets — estimates Google ranking strength.

    USE FOR: identifying which Reddit post URLs already rank on Google (organic search traffic).
    A Reddit thread with totalTraffic > 0 means Google has indexed and ranked it — it reaches
    audiences BEYOND Reddit. Higher totalTraffic = stronger Google presence = higher-value post.
    Use this after BatchRedditFetch to score posts by Google visibility.

    DO NOT USE FOR: keyword volume research (use KeywordsEverywhere for that).
    DO NOT USE FOR: more than 20 URLs per call — each URL costs credits.

    ARGS:
        urls: comma-separated Reddit post URLs to check, e.g.
              "https://reddit.com/r/SaaS/comments/abc/...,https://reddit.com/r/..."
              Max 20 URLs per call.
        country: 2-letter country code for traffic data, e.g. "us", "gb". Default "" = global.

    RETURNS: table of URL | total_keywords_ranked | monthly_organic_traffic | traffic_value,
             sorted descending by monthly_organic_traffic.
             Posts with traffic > 0 rank on Google. Use traffic score to prioritize posts.
    """
    import os as _os
    api_key = _os.environ.get("KEYWORDS_EVERYWHERE_API_KEY", "").strip()
    if not api_key:
        return ToolResponse(content=[TextBlock(
            type="text",
            text="KEYWORDS_EVERYWHERE_API_KEY not set. Cannot call Keywords Everywhere API.",
        )])

    url_list = [u.strip() for u in urls.split(",") if u.strip()][:20]
    if not url_list:
        return ToolResponse(content=[TextBlock(type="text", text="No URLs provided.")])

    def _fetch_url_traffic(url: str) -> dict:
        import httpx as _httpx
        resp = _httpx.post(
            "https://api.keywordseverywhere.com/v1/get_url_traffic",
            headers={"Authorization": f"Bearer {api_key}"},
            data={"url": url, "country": country},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()

    results = []
    loop = asyncio.get_event_loop()
    for url in url_list:
        try:
            data = await loop.run_in_executor(None, _fetch_url_traffic, url)
            traffic_data = data.get("data", {})
            results.append({
                "url": url,
                "total_keywords": int(traffic_data.get("totalKeywords", 0) or 0),
                "monthly_traffic": int(traffic_data.get("totalTraffic", 0) or 0),
                "traffic_cost": str(traffic_data.get("trafficCost", "0")),
            })
        except Exception as exc:
            results.append({
                "url": url,
                "total_keywords": 0,
                "monthly_traffic": 0,
                "traffic_cost": "0",
                "error": str(exc),
            })

    results.sort(key=lambda x: x["monthly_traffic"], reverse=True)

    lines = ["URL Traffic Metrics (Google organic traffic estimate, sorted by monthly traffic):"]
    lines.append(f"{'URL':<70} {'Keywords':>10} {'Traffic/mo':>12} {'Value':>10}")
    lines.append("-" * 106)
    for r in results:
        short_url = r["url"][-67:] if len(r["url"]) > 70 else r["url"]
        err = f" [ERROR: {r.get('error', '')}]" if r.get("error") else ""
        lines.append(
            f"{short_url:<70} {r['total_keywords']:>10,} {r['monthly_traffic']:>12,} {r['traffic_cost']:>10}{err}"
        )
    lines.append(
        "\nInterpretation: traffic > 0 = post ranks on Google. "
        "Higher traffic = more Google-visible = higher-value outreach target."
    )

    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


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
    "agentscope.tools.WebSearch":            web_search,
    "agentscope.tools.WebFetch":             web_fetch,
    "agentscope.tools.DeepCrawl":            deep_crawl,
    "agentscope.tools.RedditSearch":         reddit_search,
    "agentscope.tools.RedditFetch":          reddit_fetch,
    "agentscope.tools.BatchRedditFetch":     reddit_batch_fetch,
    "agentscope.tools.KeywordsEverywhere":        keywords_everywhere,
    "agentscope.tools.KeywordsEverywhereTraffic": keywords_everywhere_url_traffic,
    "agentscope.tools.ReadFile":                  read_file,
    "agentscope.tools.WriteFile":            write_file,
    "agentscope.tools.CodeExecution":        code_execution,
}

_ABSTRACT_TO_RESOLVED: dict[str, str] = {
    "WebSearch":          "agentscope.tools.WebSearch",
    "WebFetch":           "agentscope.tools.WebFetch",
    "DeepCrawl":          "agentscope.tools.DeepCrawl",
    "RedditSearch":       "agentscope.tools.RedditSearch",
    "RedditFetch":        "agentscope.tools.RedditFetch",
    "BatchRedditFetch":   "agentscope.tools.BatchRedditFetch",
    "KeywordsEverywhere":        "agentscope.tools.KeywordsEverywhere",
    "KeywordsEverywhereTraffic": "agentscope.tools.KeywordsEverywhereTraffic",
    "ReadFile":                  "agentscope.tools.ReadFile",
    "WriteFile":          "agentscope.tools.WriteFile",
    "CodeExecution":      "agentscope.tools.CodeExecution",
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
        "WebSearch":          web_search,
        "WebFetch":           web_fetch,
        "DeepCrawl":          deep_crawl,
        "RedditSearch":       reddit_search,
        "RedditFetch":        reddit_fetch,
        "BatchRedditFetch":   reddit_batch_fetch,
        "KeywordsEverywhere":        keywords_everywhere,
        "KeywordsEverywhereTraffic": keywords_everywhere_url_traffic,
        "ReadFile":                  read_file,
        "WriteFile":          write_file,
        "CodeExecution":      code_execution,
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
