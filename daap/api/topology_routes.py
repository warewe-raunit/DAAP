"""
DAAP Topology Routes — REST endpoints for saved topology management.

Mounted by daap.api.routes and configured via set_store/set_session_manager.
"""

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from daap.executor.engine import execute_topology
from daap.spec.resolver import resolve_topology
from daap.spec.schema import TopologySpec
from daap.topology.store import TopologyStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/topologies", tags=["topologies"])

_store: TopologyStore | None = None
_session_manager = None


def set_store(store: TopologyStore) -> None:
    """Inject TopologyStore instance from routes.py."""
    global _store
    _store = store


def set_session_manager(session_manager) -> None:
    """Inject SessionManager instance from routes.py."""
    global _session_manager
    _session_manager = session_manager


def _get_store() -> TopologyStore:
    if _store is None:
        raise RuntimeError("TopologyStore not initialized")
    return _store


def _get_session_manager():
    if _session_manager is None:
        raise RuntimeError("SessionManager not initialized")
    return _session_manager


def _topology_to_dict(topology) -> dict:
    return {
        "topology_id": topology.topology_id,
        "version": topology.version,
        "user_id": topology.user_id,
        "name": topology.name,
        "spec": topology.spec,
        "created_at": topology.created_at,
        "updated_at": topology.updated_at,
        "deleted_at": topology.deleted_at,
        "max_runs": topology.max_runs,
    }


def _run_to_dict(run) -> dict:
    return {
        "run_id": run.run_id,
        "topology_id": run.topology_id,
        "topology_version": run.topology_version,
        "user_id": run.user_id,
        "ran_at": run.ran_at,
        "user_prompt": run.user_prompt,
        "result": run.result,
        "success": run.success,
        "latency_seconds": run.latency_seconds,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
    }


class TopologyPatchRequest(BaseModel):
    spec: dict
    save_mode: Literal["overwrite", "new_version"]


class RerunRequest(BaseModel):
    user_prompt: str | None = None
    session_id: str


class RenameRequest(BaseModel):
    name: str


class MaxRunsRequest(BaseModel):
    max_runs: int


@router.get("")
async def list_topologies(
    user_id: str = Query(default="default"),
    include_deleted: bool = Query(default=False),
):
    """List saved topologies for a user (latest version only)."""
    store = _get_store()
    topologies = store.list_topologies(user_id=user_id, include_deleted=include_deleted)
    return {"topologies": [_topology_to_dict(topology) for topology in topologies]}


@router.get("/{topology_id}")
async def get_topology(topology_id: str):
    """Get latest saved version of a topology."""
    store = _get_store()
    topology = store.get_topology(topology_id)
    if topology is None:
        raise HTTPException(status_code=404, detail="Topology not found")
    return _topology_to_dict(topology)


@router.get("/{topology_id}/versions")
async def list_versions(topology_id: str):
    """List all saved versions for a topology."""
    store = _get_store()
    versions = store.list_versions(topology_id)
    if not versions:
        raise HTTPException(status_code=404, detail="Topology not found")
    # If the topology is soft-deleted, all versions will have deleted_at set.
    # Return 410 rather than leaking deleted topology data.
    if all(t.deleted_at is not None for t in versions):
        raise HTTPException(status_code=410, detail="Topology has been deleted")
    return {"versions": [_topology_to_dict(topology) for topology in versions]}


@router.get("/{topology_id}/v/{version}")
async def get_topology_version(topology_id: str, version: int):
    """Get a specific topology version."""
    store = _get_store()
    topology = store.get_topology(topology_id, version=version)
    if topology is None:
        raise HTTPException(status_code=404, detail="Topology version not found")
    return _topology_to_dict(topology)


@router.get("/{topology_id}/runs")
async def get_runs(topology_id: str, limit: int | None = Query(default=None)):
    """Get run history across all versions of a topology."""
    store = _get_store()
    topology = store.get_topology(topology_id)
    if topology is None:
        raise HTTPException(status_code=404, detail="Topology not found")
    runs = store.get_runs(topology_id, limit=limit)
    return {"runs": [_run_to_dict(run) for run in runs]}


@router.patch("/{topology_id}")
async def patch_topology(topology_id: str, req: TopologyPatchRequest):
    """
    Update topology spec directly.

    save_mode='overwrite': replace latest version.
    save_mode='new_version': save as incremented version.
    """
    store = _get_store()
    existing = store.get_topology(topology_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Topology not found")

    overwrite = req.save_mode == "overwrite"
    spec = {**req.spec, "topology_id": topology_id}
    saved = store.save_topology(
        spec=spec,
        user_id=existing.user_id,
        name=existing.name,
        overwrite=overwrite,
    )
    return _topology_to_dict(saved)


@router.patch("/{topology_id}/rename")
async def rename_topology(topology_id: str, req: RenameRequest):
    """Rename a topology."""
    store = _get_store()
    if store.get_topology(topology_id) is None:
        raise HTTPException(status_code=404, detail="Topology not found")
    store.rename_topology(topology_id, req.name)
    return {"status": "renamed", "name": req.name}


@router.patch("/{topology_id}/max-runs")
async def set_max_runs(topology_id: str, req: MaxRunsRequest):
    """Set run-history cap for a topology."""
    store = _get_store()
    if store.get_topology(topology_id) is None:
        raise HTTPException(status_code=404, detail="Topology not found")
    if not (1 <= req.max_runs <= 100):
        raise HTTPException(status_code=400, detail="max_runs must be between 1 and 100")
    store.set_max_runs(topology_id, req.max_runs)
    return {"status": "updated", "max_runs": req.max_runs}


@router.post("/{topology_id}/rerun")
async def rerun_topology(topology_id: str, req: RerunRequest):
    """Rerun a saved topology with an optional prompt override."""
    store = _get_store()
    session_manager = _get_session_manager()

    stored = store.get_topology(topology_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Topology not found")
    if stored.deleted_at is not None:
        raise HTTPException(status_code=410, detail="Topology has been deleted")

    session = session_manager.get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != stored.user_id:
        raise HTTPException(status_code=403, detail="Session user does not own this topology")

    user_prompt = req.user_prompt or stored.spec.get("user_prompt", "")

    try:
        spec = TopologySpec.model_validate(stored.spec)
        resolved = resolve_topology(spec)
        if isinstance(resolved, list):
            details = [error.message for error in resolved]
            raise HTTPException(status_code=400, detail={"errors": details})

        result = await execute_topology(
            resolved=resolved,
            user_prompt=user_prompt,
            tracker=session.token_tracker,
        )

        result_payload = {
            "topology_id": result.topology_id,
            "final_output": result.final_output,
            "success": result.success,
            "error": result.error,
            "latency_seconds": result.total_latency_seconds,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
        }
        store.save_run(
            topology_id=topology_id,
            topology_version=stored.version,
            user_id=session.user_id,
            result=result_payload,
            user_prompt=user_prompt,
        )

        return {
            "status": "success" if result.success else "failed",
            "topology_id": topology_id,
            "version": stored.version,
            "result": result_payload,
            "models_used": result.models_used,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Rerun failed for topology %s", topology_id)
        raise HTTPException(status_code=500, detail=f"Rerun failed: {exc}")


@router.delete("/{topology_id}")
async def delete_topology(topology_id: str, ttl_days: int = Query(default=30)):
    """Soft-delete a topology and mark purge TTL in days."""
    store = _get_store()
    if store.get_topology(topology_id) is None:
        raise HTTPException(status_code=404, detail="Topology not found")
    store.delete_topology(topology_id, ttl_days=ttl_days)
    return {"status": "deleted", "topology_id": topology_id, "ttl_days": ttl_days}


@router.post("/{topology_id}/restore")
async def restore_topology(topology_id: str):
    """Restore a soft-deleted topology."""
    store = _get_store()
    if store.get_topology(topology_id) is None:
        raise HTTPException(status_code=404, detail="Topology not found")
    store.restore_topology(topology_id)
    return {"status": "restored", "topology_id": topology_id}
