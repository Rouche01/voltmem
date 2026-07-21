"""Tests for the VoltMem HTTP sidecar."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Embeddings off for CI-fast runs (must be set before app lifespan).
os.environ["VOLTMEM_EMBEDDINGS"] = "0"
os.environ.setdefault("VOLTMEM_PROFILE", "stylens")

from fastapi.testclient import TestClient  # noqa: E402

from sidecar.app import create_app  # noqa: E402


def _client(**env: str) -> TestClient:
    """Fresh app with env applied for lifespan."""
    if "VOLTMEM_API_KEY" not in env:
        os.environ.pop("VOLTMEM_API_KEY", None)
    for key, value in env.items():
        os.environ[key] = value
    os.environ["VOLTMEM_EMBEDDINGS"] = "0"
    return TestClient(create_app())


def test_health():
    with _client(VOLTMEM_DB_PATH=":memory:") as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_add_search_domain_stats_delete():
    with _client(VOLTMEM_DB_PATH=":memory:") as client:
        add = client.post(
            "/v1/users/alice/memories",
            json={"data": "I prefer darker colors and minimal fits"},
        )
        assert add.status_code == 200, add.text
        body = add.json()
        assert body["action"] == "inserted"
        assert body["domain"] == "style_preference"
        mid = body["id"]

        search = client.get(
            "/v1/users/alice/memories/search",
            params={"q": "style preferences colors", "limit": 3},
        )
        assert search.status_code == 200
        hits = search.json()
        assert any(h["id"] == mid for h in hits)

        stats = client.get("/v1/users/alice/domain_stats")
        assert stats.status_code == 200
        assert "style_preference" in stats.json()

        got = client.get(f"/v1/users/alice/memories/{mid}")
        assert got.status_code == 200
        assert got.json()["id"] == mid

        deleted = client.delete(f"/v1/users/alice/memories/{mid}")
        assert deleted.status_code == 200
        assert deleted.json() == {"deleted": True}

        missing = client.get(f"/v1/users/alice/memories/{mid}")
        assert missing.status_code == 404


def test_namespace_isolation():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        with _client(VOLTMEM_DB_PATH=path) as client:
            client.post(
                "/v1/users/alice/memories",
                json={"data": "I prefer darker colors"},
            )
            client.post(
                "/v1/users/bob/memories",
                json={"data": "I prefer neon colors"},
            )
            alice = client.get("/v1/users/alice/memories").json()
            bob = client.get("/v1/users/bob/memories").json()
            assert all("neon" not in m["memory"].lower() for m in alice)
            assert any("neon" in m["memory"].lower() for m in bob)
            assert all("darker" not in m["memory"].lower() for m in bob)
    finally:
        os.unlink(path)


def test_api_key_required_when_set():
    previous = os.environ.get("VOLTMEM_API_KEY")
    try:
        with _client(
            VOLTMEM_DB_PATH=":memory:",
            VOLTMEM_API_KEY="test-secret",
        ) as client:
            # Health stays open
            assert client.get("/health").status_code == 200

            denied = client.post(
                "/v1/users/alice/memories",
                json={"data": "I prefer minimal fits"},
            )
            assert denied.status_code == 401

            wrong = client.post(
                "/v1/users/alice/memories",
                json={"data": "I prefer minimal fits"},
                headers={"X-API-Key": "wrong"},
            )
            assert wrong.status_code == 401

            ok = client.post(
                "/v1/users/alice/memories",
                json={"data": "I prefer minimal fits"},
                headers={"X-API-Key": "test-secret"},
            )
            assert ok.status_code == 200, ok.text
            assert ok.json()["domain"] == "style_preference"
    finally:
        if previous is None:
            os.environ.pop("VOLTMEM_API_KEY", None)
        else:
            os.environ["VOLTMEM_API_KEY"] = previous


def test_clear_and_summary():
    with _client(VOLTMEM_DB_PATH=":memory:") as client:
        client.post(
            "/v1/users/carol/memories",
            json={"data": "No wool — I'm allergic"},
        )
        summary = client.get("/v1/users/carol/summary")
        assert summary.status_code == 200
        assert isinstance(summary.json(), dict)

        cleared = client.delete("/v1/users/carol/memories")
        assert cleared.status_code == 200
        assert client.get("/v1/users/carol/memories").json() == []


def test_occasion_domain():
    with _client(VOLTMEM_DB_PATH=":memory:") as client:
        r = client.post(
            "/v1/users/dave/memories",
            json={"data": "I'm dressing for a summer wedding"},
        )
        assert r.status_code == 200
        assert r.json()["domain"] == "session_occasion"


if __name__ == "__main__":
    tests = [
        test_health,
        test_add_search_domain_stats_delete,
        test_namespace_isolation,
        test_api_key_required_when_set,
        test_clear_and_summary,
        test_occasion_domain,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    if failed:
        raise SystemExit(f"{failed} test(s) failed")
    print(f"{len(tests)} passed")
