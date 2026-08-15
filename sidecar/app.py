"""FastAPI app — VoltMem HTTP sidecar."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated, Any, Union

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel

from .auth import require_api_key
from .memory_pool import MemoryPool
from .profiles import build_profile

AddData = Union[str, dict[str, str], list[dict[str, str]]]


class AddBody(BaseModel):
    data: AddData
    source: str = "explicit_statement"
    extract: bool | None = None
    event_id: str | None = None
    modality: str | None = None
    expires_at: float | None = None
    ttl_seconds: float | None = None


class AddEventBody(BaseModel):
    event_id: str
    facets: list[dict[str, Any]]
    source: str = "explicit_statement"


class MaintenanceTriggerBody(BaseModel):
    task: str | None = None
    dry_run: bool = False


class MaintenanceRollbackBody(BaseModel):
    run_id: str


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def create_app() -> FastAPI:
    """Build the sidecar app (used by uvicorn and tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        profile = os.environ.get("VOLTMEM_PROFILE", "stylens").strip() or "stylens"
        domains, classifier = build_profile(profile)
        restore = domains.install()

        db_path = os.environ.get("VOLTMEM_DB_PATH", "voltmem_sidecar.db")
        embeddings = _env_bool("VOLTMEM_EMBEDDINGS", True)
        verify_on_write = _env_bool("VOLTMEM_VERIFY_ON_WRITE", False)
        pool = MemoryPool(
            db_path, embeddings=embeddings, classifier=classifier,
            verify_on_write=verify_on_write,
        )
        app.state.pool = pool
        app.state.domain_restore = restore
        from .maintenance_scheduler import SidecarMaintenanceScheduler
        scheduler = SidecarMaintenanceScheduler.maybe_start(pool)
        app.state.maintenance_scheduler = scheduler
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.stop()
            pool.close()
            restore()

    app = FastAPI(
        title="VoltMem Sidecar",
        version="0.1.0",
        description="HTTP surface over VoltMem create_memory (add/search/domain_stats).",
        lifespan=lifespan,
    )

    def get_pool(request: Request) -> MemoryPool:
        return request.app.state.pool

    authed = [Depends(require_api_key)]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/users/{user_id}/memories", dependencies=authed)
    def add_memory(
        user_id: str,
        body: AddBody,
        mem_pool: MemoryPool = Depends(get_pool),
    ) -> Any:
        mem = mem_pool.for_user(user_id)
        kwargs: dict[str, Any] = {"source": body.source}
        if body.extract is not None:
            kwargs["extract"] = body.extract
        # Pass event_id / modality / expires_at when present
        if body.event_id is not None:
            kwargs["event_id"] = body.event_id
        if body.modality is not None:
            kwargs["modality"] = body.modality
        if body.expires_at is not None:
            kwargs["expires_at"] = body.expires_at
        if body.ttl_seconds is not None:
            kwargs["ttl_seconds"] = body.ttl_seconds
        return mem.add(body.data, **kwargs)

    @app.post("/v1/users/{user_id}/events", dependencies=authed)
    def add_event(
        user_id: str,
        body: AddEventBody,
        mem_pool: MemoryPool = Depends(get_pool),
    ) -> Any:
        mem = mem_pool.for_user(user_id)
        return mem.add_event(
            event_id=body.event_id,
            facets=body.facets,
            source=body.source,
        )

    @app.get("/v1/users/{user_id}/events/{event_id}", dependencies=authed)
    def get_event(
        user_id: str,
        event_id: str,
        mem_pool: MemoryPool = Depends(get_pool),
    ) -> list[dict[str, Any]]:
        return mem_pool.for_user(user_id).get_event(event_id)

    @app.get("/v1/users/{user_id}/memories/search", dependencies=authed)
    def search_memories(
        user_id: str,
        q: Annotated[str, Query(min_length=1)],
        limit: Annotated[int, Query(ge=1, le=100)] = 5,
        min_score: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
        mem_pool: MemoryPool = Depends(get_pool),
    ) -> list[dict[str, Any]]:
        return mem_pool.for_user(user_id).search(
            q, limit=limit, min_score=min_score
        )

    @app.get("/v1/users/{user_id}/memories", dependencies=authed)
    def list_memories(
        user_id: str,
        mem_pool: MemoryPool = Depends(get_pool),
    ) -> list[dict[str, Any]]:
        return mem_pool.for_user(user_id).get_all()

    @app.get("/v1/users/{user_id}/memories/{memory_id}", dependencies=authed)
    def get_memory(
        user_id: str,
        memory_id: str,
        mem_pool: MemoryPool = Depends(get_pool),
    ) -> dict[str, Any]:
        row = mem_pool.for_user(user_id).get(memory_id)
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="memory not found",
            )
        return row

    @app.delete(
        "/v1/users/{user_id}/memories/{memory_id}",
        dependencies=authed,
    )
    def delete_memory(
        user_id: str,
        memory_id: str,
        mem_pool: MemoryPool = Depends(get_pool),
    ) -> dict[str, bool]:
        ok = mem_pool.for_user(user_id).delete(memory_id)
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="memory not found",
            )
        return {"deleted": True}

    @app.delete("/v1/users/{user_id}/memories", dependencies=authed)
    def clear_memories(
        user_id: str,
        mem_pool: MemoryPool = Depends(get_pool),
    ) -> dict[str, bool]:
        mem_pool.for_user(user_id).clear()
        return {"cleared": True}

    @app.get("/v1/users/{user_id}/summary", dependencies=authed)
    def summary(
        user_id: str,
        mem_pool: MemoryPool = Depends(get_pool),
    ) -> dict[str, Any]:
        return mem_pool.for_user(user_id).summary()

    @app.get("/v1/users/{user_id}/domain_stats", dependencies=authed)
    def domain_stats(
        user_id: str,
        mem_pool: MemoryPool = Depends(get_pool),
    ) -> dict[str, Any]:
        return mem_pool.for_user(user_id).domain_stats()

    # ── maintenance endpoints ─────────────────────────────────────────────────

    @app.post("/v1/users/{user_id}/maintenance/trigger", dependencies=authed)
    def maintenance_trigger(
        user_id: str,
        body: MaintenanceTriggerBody,
        mem_pool: MemoryPool = Depends(get_pool),
    ) -> dict[str, Any]:
        """Run a maintenance task for a user.

        Pass ``task`` to run a specific task, or omit to run the default set
        (``expire_cleanup``, flag tasks, ``consolidate``, ``reconcile_twins``).

        ``dry_run`` (default ``false``) gates mutating tasks. Pass
        ``dry_run=true`` to preview. Returns a ``run_id`` for
        ``POST .../maintenance/rollback``.
        """
        mem = mem_pool.for_user(user_id)
        from voltmem import MaintenanceWindow
        mw = MaintenanceWindow(mem.layer)
        from voltmem.maintenance import (
            expire_cleanup,
            reclassify_ambiguous,
            pattern_audit,
            consolidate,
            reconcile_twins_default,
        )
        mw.register("expire_cleanup", expire_cleanup, interval=0)
        mw.register("reclassify_ambiguous", reclassify_ambiguous, interval=0)
        mw.register("pattern_audit", pattern_audit, interval=0)
        mw.register("consolidate", consolidate, interval=0)
        mw.register("reconcile_twins", reconcile_twins_default, interval=0)

        if body.task:
            try:
                out = mw.run_once(body.task, dry_run=body.dry_run)
                return out
            except KeyError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unknown task: {body.task}",
                ) from e

        return mw.run_all(dry_run=body.dry_run)

    @app.post("/v1/users/{user_id}/maintenance/rollback", dependencies=authed)
    def maintenance_rollback(
        user_id: str,
        body: MaintenanceRollbackBody,
        mem_pool: MemoryPool = Depends(get_pool),
    ) -> dict[str, Any]:
        """Undo a ledgered maintenance run (supersedes + purged snapshots)."""
        mem = mem_pool.for_user(user_id)
        try:
            return mem.layer.rollback_maintenance(body.run_id)
        except KeyError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
            ) from e
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
            ) from e

    @app.get("/v1/users/{user_id}/maintenance/tasks", dependencies=authed)
    def maintenance_tasks(
        user_id: str,
        mem_pool: MemoryPool = Depends(get_pool),
    ) -> list[dict[str, Any]]:
        """List available maintenance tasks."""
        return [
            {
                "name": "expire_cleanup",
                "description": "Purge expired rows (mutates unless dry_run; ledgered; scheduled)",
                "interval": 3600,
                "mutates": True,
                "default_run_all": True,
            },
            {
                "name": "reclassify_ambiguous",
                "description": "Flag domains with high mismatch rates (read-only; scheduled daily)",
                "interval": 86400,
                "mutates": False,
                "default_run_all": True,
            },
            {
                "name": "pattern_audit",
                "description": "Flag items with accumulated mismatches (read-only; scheduled)",
                "interval": 3600,
                "mutates": False,
                "default_run_all": True,
            },
            {
                "name": "consolidate",
                "description": (
                    "Merge mismatch evidence into updated tips "
                    "(mutates unless dry_run; ledgered; scheduled daily)"
                ),
                "interval": 86400,
                "mutates": True,
                "default_run_all": True,
            },
            {
                "name": "reconcile_twins",
                "description": (
                    "Pair-verify near-duplicate memories and retire the older "
                    "(mutates unless dry_run; ledgered; scheduled daily)"
                ),
                "interval": 86400,
                "mutates": True,
                "default_run_all": True,
            },
        ]

    return app


# Module-level app for ``uvicorn sidecar.app:app``
app = create_app()
