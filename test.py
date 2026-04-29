import asyncio
from crawl4ai import AsyncWebCrawler

async def main():
    url = 'https://www.reddit.com/r/AITAH/comments/1stu0jy/aitah_for_not_giving_my_exs_wife_equal_say_and/'
    crawler = AsyncWebCrawler()
    result = await crawler.crawl(url, max_pages=1)
    print(result[:1500])

asyncio.run(main())