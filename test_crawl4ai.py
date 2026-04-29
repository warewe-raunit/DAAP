import asyncio
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

POST_ID = "1stu0jy"
BASE = "https://www.reddit.com/r/AITAH/comments/1stu0jy/aitah_for_not_giving_my_exs_wife_equal_say_and"

urls = {
    "old.reddit": f"https://old.reddit.com/r/AITAH/comments/{POST_ID}/",
    "json_api":   f"{BASE}/.json",
}

async def test(label, url):
    print(f"\n{'='*50}")
    print(f"Testing: {label}")
    print(f"URL: {url}")
    cfg = BrowserConfig(headers={"User-Agent": "Mozilla/5.0 (compatible; test/1.0)"})
    run_cfg = CrawlerRunConfig(word_count_threshold=10)
    async with AsyncWebCrawler(verbose=False, config=cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)
        if result.success:
            md = result.markdown or ""
            print(f"SUCCESS — {len(md)} chars")
            # Check for actual post keywords
            has_post = any(w in md.lower() for w in ["aitah", "ex", "wife", "kids", "decision"])
            print(f"Contains post content: {has_post}")
            print("--- Sample ---")
            print(md[:600])
        else:
            print(f"FAILED: {result.error_message}")

async def main():
    for label, url in urls.items():
        await test(label, url)

asyncio.run(main())
