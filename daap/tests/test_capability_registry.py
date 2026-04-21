"""Tests for daap.master.capability_registry."""
import pytest
from daap.master.capability_registry import (
    build_functional_capabilities,
    build_known_gaps,
    CAPABILITY_REGISTRY,
)


def test_builtin_tool_marked_available_when_installed():
    installed = {"WebSearch", "WebFetch", "CodeExecution"}
    caps = build_functional_capabilities(installed)
    web_search = next(c for c in caps if c["label"] == "Web search")
    assert web_search["available"] is True


def test_builtin_tool_marked_unavailable_when_missing():
    installed = set()
    caps = build_functional_capabilities(installed)
    web_search = next(c for c in caps if c["label"] == "Web search")
    assert web_search["available"] is False


def test_mcp_tool_marked_available_by_server_prefix():
    installed = {"WebSearch", "mcp://linkedin/search_people"}
    caps = build_functional_capabilities(installed)
    linkedin = next(c for c in caps if c["label"] == "LinkedIn")
    assert linkedin["available"] is True


def test_mcp_tool_marked_available_by_bare_server_name():
    installed = {"mcp://linkedin"}
    caps = build_functional_capabilities(installed)
    linkedin = next(c for c in caps if c["label"] == "LinkedIn")
    assert linkedin["available"] is True


def test_mcp_tool_marked_unavailable_when_not_installed():
    installed = {"WebSearch", "WebFetch"}
    caps = build_functional_capabilities(installed)
    linkedin = next(c for c in caps if c["label"] == "LinkedIn")
    assert linkedin["available"] is False


def test_all_entries_present_in_functional_capabilities():
    installed = set()
    caps = build_functional_capabilities(installed)
    assert len(caps) == len(CAPABILITY_REGISTRY)


def test_known_gaps_excludes_builtin_tools():
    installed = set()
    gaps = build_known_gaps(installed)
    gap_labels = {g["label"] for g in gaps}
    assert "Web search" not in gap_labels
    assert "Web page reading" not in gap_labels
    assert "Code execution" not in gap_labels


def test_known_gaps_includes_missing_mcp_with_install_cmd():
    installed = {"WebSearch"}
    gaps = build_known_gaps(installed)
    linkedin_gap = next((g for g in gaps if g["label"] == "LinkedIn"), None)
    assert linkedin_gap is not None
    assert "install_cmd" in linkedin_gap
    assert "daap mcp add linkedin" in linkedin_gap["install_cmd"]


def test_known_gaps_empty_when_all_mcps_installed():
    installed = {
        "mcp://linkedin",
        "mcp://crunchbase",
        "mcp://gmail",
        "mcp://slack",
        "mcp://github",
        "mcp://hubspot",
    }
    gaps = build_known_gaps(installed)
    assert gaps == []


def test_known_gaps_has_task_keywords():
    installed = set()
    gaps = build_known_gaps(installed)
    for gap in gaps:
        assert "keywords" in gap
        assert len(gap["keywords"]) > 0
