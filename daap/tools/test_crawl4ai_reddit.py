import asyncio
import json
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from urllib.parse import urlencode

async def _reddit_search_via_crawl4ai(base: str, params: dict) -> dict | None:
    search_url = f"{base}?{urlencode(params)}"
    run_cfg = CrawlerRunConfig(word_count_threshold=5, remove_overlay_elements=True)

    def _run_in_thread():
        import sys
        if hasattr(sys.stdout, 'reconfigure'):
            try: sys.stdout.reconfigure(encoding='utf-8')
            except: pass
        if hasattr(sys.stderr, 'reconfigure'):
            try: sys.stderr.reconfigure(encoding='utf-8')
            except: pass
        loop = (
            asyncio.ProactorEventLoop()
            if hasattr(asyncio, "ProactorEventLoop")
            else asyncio.new_event_loop()
        )
        asyncio.set_event_loop(loop)
        try:
            async def _fetch():
                async with AsyncWebCrawler(verbose=False) as crawler:
                    return await crawler.arun(url=search_url, config=run_cfg)
            return loop.run_until_complete(_fetch())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _run_in_thread)
        raw = result.markdown or result.html or ""
        print("RAW OUTPUT:", repr(raw[:500]))
        if raw:
            if "```" in raw:
                import re
                raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print("JSON Decode Error:", e)
                pass
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print("crawl4ai reddit search failed for", search_url, exc)
    return None

async def main():
    base = "https://www.reddit.com/r/SaaS/search.json"
    params = {
        "q": "email verification tool",
        "restrict_sr": "1",
        "type": "link",
        "sort": "relevance",
        "t": "month",
        "limit": "10",
    }
    res = await _reddit_search_via_crawl4ai(base, params)
    print("FINAL RESULT:", res is not None)

if __name__ == "__main__":
    asyncio.run(main())
