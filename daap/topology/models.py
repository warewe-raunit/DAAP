"""DAAP Topology Persistence Models — dataclasses for stored topologies and runs."""
from dataclasses import dataclass, field


@dataclass
class StoredTopology:
    """A topology that has been saved to persistent storage."""
    topology_id: str
    version: int
    user_id: str
    name: str
    spec: dict
    created_at: float
    updated_at: float
    deleted_at: float | None
    max_runs: int = 10


@dataclass
class TopologyRun:
    """One execution run of a stored topology."""
    run_id: str
    topology_id: str
    topology_version: int
    user_id: str
    ran_at: float
    user_prompt: str | None
    result: dict | None
    success: bool
    latency_seconds: float
    input_tokens: int
    output_tokens: int
