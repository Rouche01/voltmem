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
        pool = MemoryPool(db_path, embeddings=embeddings, classifier=classifier)
        app.state.pool = pool
        app.state.domain_restore = restore
        try:
            yield
        finally:
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
        return mem.add(body.data, **kwargs)

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

    return app


# Module-level app for ``uvicorn sidecar.app:app``
app = create_app()
