"""Unit tests for auto_name_from_prompt."""

from daap.topology.naming import auto_name_from_prompt


def test_basic_slug():
    assert auto_name_from_prompt("Find B2B leads in fintech") == "find-b2b-leads-in-fintech"


def test_strips_special_chars():
    assert auto_name_from_prompt("Find leads! @2024 #top") == "find-leads-2024-top"


def test_collapses_spaces():
    assert auto_name_from_prompt("find   lots   of   leads") == "find-lots-of-leads"


def test_max_60_chars():
    long_prompt = "a" * 200
    result = auto_name_from_prompt(long_prompt)
    assert len(result) <= 60


def test_empty_prompt_fallback():
    assert auto_name_from_prompt("") == "unnamed-topology"


def test_whitespace_only_fallback():
    assert auto_name_from_prompt("   ") == "unnamed-topology"


def test_no_trailing_dash():
    result = auto_name_from_prompt("a" * 59 + " b")
    assert not result.endswith("-")
