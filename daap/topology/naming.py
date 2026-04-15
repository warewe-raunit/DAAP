"""Auto-generate stable, URL-safe names for stored topologies."""

import re


def auto_name_from_prompt(prompt: str) -> str:
    """Create a readable slug from a user prompt with a 60-char cap."""
    slug = (prompt or "").lower()
    slug = re.sub(r"[^a-z0-9\s]", "", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug)
    slug = slug[:60].rstrip("-")
    return slug or "unnamed-topology"
