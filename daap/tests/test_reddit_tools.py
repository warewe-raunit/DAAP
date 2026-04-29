"""Tests for Reddit search/fetch tool wiring."""

import pytest

from daap.tools import registry
from daap.tools.registry import get_available_tool_names, get_tool_registry


def _block_text(block) -> str:
    if hasattr(block, "text"):
        return block.text
    if isinstance(block, dict):
        return block.get("text", "")
    return str(block)


def test_reddit_tools_are_runtime_available():
    names = get_available_tool_names(include_mcp_placeholders=False)
    tools = get_tool_registry()

    assert "RedditSearch" in names
    assert "RedditFetch" in names
    assert "BatchRedditFetch" in names
    assert "PullpushSearch" not in names
    assert "agentscope.tools.RedditSearch" in tools
    assert "agentscope.tools.RedditFetch" in tools
    assert "agentscope.tools.BatchRedditFetch" in tools
    assert "agentscope.tools.PullpushSearch" not in tools


@pytest.mark.asyncio
async def test_reddit_search_formats_public_json_results(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": {
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "title": "Where should I promote my SaaS?",
                                "permalink": "/r/SaaS/comments/abc123/post/",
                                "subreddit": "SaaS",
                                "score": 42,
                                "num_comments": 7,
                                "created_utc": 1760000000,
                                "selftext": "I need feedback on launch channels.",
                            },
                        }
                    ]
                }
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(registry.httpx, "AsyncClient", FakeClient)

    # crawl4ai is the primary path; force it to miss so the httpx fallback
    # (which the FakeClient/FakeResponse stubs above target) is exercised.
    async def _crawl_returns_none(base, params):
        return None
    monkeypatch.setattr(registry, "_reddit_search_via_crawl4ai", _crawl_returns_none)

    # Fresh-state guarantee: dedup tags depend on session state. Reset so
    # this assertion is independent of any prior search in the same process.
    registry.reset_reddit_session_state()

    result = await registry.reddit_search("promote saas", subreddit="SaaS", limit=1)
    text = _block_text(result.content[0])

    assert "Where should I promote my SaaS?" in text
    assert "https://www.reddit.com/r/SaaS/comments/abc123/post/" in text
    assert "I need feedback on launch channels." in text


@pytest.mark.asyncio
async def test_reddit_fetch_uses_direct_json_before_browser(monkeypatch):
    async def fake_json(url: str):
        return "# Title\n\n## Post Body\nBody\n\n## Top Comments\n[score:1] Comment"

    async def fail_crawl4ai(url: str):
        raise AssertionError("crawl4ai fallback should not run when direct json works")

    monkeypatch.setattr(registry, "_reddit_fetch_via_json", fake_json)
    monkeypatch.setattr(registry, "_reddit_fetch_via_crawl4ai", fail_crawl4ai)

    # Clear cache so a prior test that seeded this URL doesn't short-circuit
    # the fetch via cache and skip fake_json entirely.
    registry.reset_reddit_session_state()

    result = await registry.reddit_fetch("https://www.reddit.com/r/SaaS/comments/abc123/post/")
    text = _block_text(result.content[0])

    assert "via reddit .json" in text
    assert "## Post Body" in text
    assert "## Top Comments" in text


@pytest.mark.asyncio
async def test_reddit_batch_fetch_fetches_each_url(monkeypatch):
    calls = []

    async def fake_json(url: str):
        calls.append(url)
        return "# Title\n\n## Post Body\nBody\n\n## Top Comments\n[score:1] Comment"

    async def fail_crawl4ai(url: str):
        raise AssertionError("crawl4ai fallback should not run when direct json works")

    monkeypatch.setattr(registry, "_reddit_fetch_via_json", fake_json)
    monkeypatch.setattr(registry, "_reddit_fetch_via_crawl4ai", fail_crawl4ai)

    # Module-level _REDDIT_CONTENT_CACHE persists across tests; reset so a
    # previous test's cached body does not short-circuit the fetch we want
    # to count here.
    registry.reset_reddit_session_state()

    result = await registry.reddit_batch_fetch(
        "1. URL: https://www.reddit.com/r/SaaS/comments/abc123/post/\n"
        "2. URL: https://reddit.com/r/emailmarketing/comments/def456/post/"
    )
    text = _block_text(result.content[0])

    assert len(calls) == 2
    assert "## 1. ACTIVE" in text
    assert "## 2. ACTIVE" in text
    assert "## Top Comments" in text
