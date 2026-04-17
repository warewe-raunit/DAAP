"""CLI chat utility regression tests."""


def test_extract_text_dedupes_adjacent_duplicate_blocks():
    from scripts.chat import _extract_text

    payload = [
        {"type": "text", "text": "Hello from DAAP"},
        {"type": "text", "text": "Hello from DAAP"},
    ]

    assert _extract_text(payload) == "Hello from DAAP"


def test_extract_text_keeps_distinct_blocks():
    from scripts.chat import _extract_text

    payload = [
        {"type": "text", "text": "First line"},
        {"type": "text", "text": "Second line"},
    ]

    assert _extract_text(payload) == "First line\nSecond line"


def test_suppress_agentscope_stdout_replaces_print():
    import asyncio

    from scripts.chat import _suppress_agentscope_stdout

    class _FakeAgent:
        async def print(self, *_args, **_kwargs):
            return "original"

    agent = _FakeAgent()
    _suppress_agentscope_stdout(agent)

    result = asyncio.run(agent.print("hello"))
    assert result is None
