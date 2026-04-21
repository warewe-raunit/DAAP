"""DAAP Capability Registry.

Maps capability categories to installed tools/MCPs and surfaces gaps with
exact install commands. Used by runtime.py to enrich the snapshot injected
into the master agent system prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CapabilityEntry:
    label: str
    task_keywords: list[str]
    builtin_tool: str | None = None
    mcp_server: str | None = None
    install_cmd: str | None = None
    docs_url: str | None = None


CAPABILITY_REGISTRY: list[CapabilityEntry] = [
    CapabilityEntry(
        label="Web search",
        task_keywords=["search", "find", "research", "look up"],
        builtin_tool="WebSearch",
    ),
    CapabilityEntry(
        label="Web page reading",
        task_keywords=["read", "fetch", "scrape", "page content", "website"],
        builtin_tool="WebFetch",
    ),
    CapabilityEntry(
        label="Deep web crawling",
        task_keywords=["crawl", "multi-page", "site crawl", "documentation"],
        builtin_tool="DeepCrawl",
    ),
    CapabilityEntry(
        label="File read/write",
        task_keywords=["read file", "write file", "csv", "save", "load file"],
        builtin_tool="ReadFile",
    ),
    CapabilityEntry(
        label="Code execution",
        task_keywords=["run code", "execute", "compute", "parse data", "script"],
        builtin_tool="CodeExecution",
    ),
    CapabilityEntry(
        label="LinkedIn",
        task_keywords=["linkedin", "people search", "profile", "prospect", "sales navigator"],
        mcp_server="linkedin",
        install_cmd="daap mcp add linkedin npx @daap/linkedin-mcp",
        docs_url="https://mcp.so/server/linkedin",
    ),
    CapabilityEntry(
        label="Crunchbase",
        task_keywords=["crunchbase", "funding", "investors", "startup data", "company data"],
        mcp_server="crunchbase",
        install_cmd="daap mcp add crunchbase npx @daap/crunchbase-mcp",
        docs_url="https://mcp.so/server/crunchbase",
    ),
    CapabilityEntry(
        label="Email sending",
        task_keywords=["send email", "gmail", "smtp", "email outreach", "send outreach"],
        mcp_server="gmail",
        install_cmd="daap mcp add gmail npx @modelcontextprotocol/server-gmail",
        docs_url="https://mcp.so/server/gmail",
    ),
    CapabilityEntry(
        label="Slack",
        task_keywords=["slack", "send message", "notify team", "post to channel"],
        mcp_server="slack",
        install_cmd="daap mcp add slack npx @modelcontextprotocol/server-slack",
        docs_url="https://mcp.so/server/slack",
    ),
    CapabilityEntry(
        label="GitHub",
        task_keywords=["github", "repository", "pull request", "issues", "code review"],
        mcp_server="github",
        install_cmd="daap mcp add github npx @modelcontextprotocol/server-github",
        docs_url="https://mcp.so/server/github",
    ),
    CapabilityEntry(
        label="HubSpot CRM",
        task_keywords=["hubspot", "crm", "contacts", "deals", "pipeline"],
        mcp_server="hubspot",
        install_cmd="daap mcp add hubspot npx @daap/hubspot-mcp",
        docs_url="https://mcp.so/server/hubspot",
    ),
]


def build_functional_capabilities(installed_tool_names: set[str]) -> list[dict]:
    """Return list of {label, available} for all registered capabilities.

    Args:
        installed_tool_names: full set of available tool names (builtins + MCP IDs).
    """
    results = []
    for entry in CAPABILITY_REGISTRY:
        if entry.builtin_tool is not None:
            available = entry.builtin_tool in installed_tool_names
        elif entry.mcp_server is not None:
            prefix = f"mcp://{entry.mcp_server}"
            available = any(
                t == prefix or t.startswith(prefix + "/")
                for t in installed_tool_names
            )
        else:
            available = False
        results.append({"label": entry.label, "available": available})
    return results


def build_known_gaps(installed_tool_names: set[str]) -> list[dict]:
    """Return gaps: MCP capabilities not yet installed, with install info.

    Builtin tools are excluded — they are always present when DAAP runs.
    """
    gaps = []
    for entry in CAPABILITY_REGISTRY:
        if entry.builtin_tool is not None:
            continue
        if entry.mcp_server is None:
            continue
        prefix = f"mcp://{entry.mcp_server}"
        available = any(
            t == prefix or t.startswith(prefix + "/")
            for t in installed_tool_names
        )
        if not available:
            gap: dict = {
                "label": entry.label,
                "keywords": entry.task_keywords,
            }
            if entry.install_cmd:
                gap["install_cmd"] = entry.install_cmd
            elif entry.docs_url:
                gap["docs_url"] = entry.docs_url
            gaps.append(gap)
    return gaps
