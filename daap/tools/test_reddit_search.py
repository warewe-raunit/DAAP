import asyncio
from registry import reddit_search

async def main():
    print("Running reddit search...")
    res = await reddit_search(query="email verification tool")
    print(res)

if __name__ == "__main__":
    asyncio.run(main())
