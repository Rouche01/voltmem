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
os.environ["VOLTMEM_MAINTENANCE"] = "0"
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
            assert client.get("/v1/users/alice/graph").status_code == 401

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


def test_maintenance_dry_run_gates_expire_cleanup():
    """Default dry_run=false purges; dry_run=true previews only."""
    with _client(VOLTMEM_DB_PATH=":memory:") as client:
        add = client.post(
            "/v1/users/eve/memories",
            json={
                "data": "Conference badge code 9911",
                "expires_at": 1.0,
            },
        )
        assert add.status_code == 200, add.text
        mid = add.json()["id"]

        preview = client.post(
            "/v1/users/eve/maintenance/trigger",
            json={"task": "expire_cleanup", "dry_run": True},
        )
        assert preview.status_code == 200, preview.text
        body = preview.json()
        assert body["dry_run"] is True
        assert body["result"] == 1
        assert body["run_id"]

        still = client.get(f"/v1/users/eve/memories/{mid}")
        assert still.status_code == 200

        wet = client.post(
            "/v1/users/eve/maintenance/trigger",
            json={"task": "expire_cleanup"},  # dry_run defaults false
        )
        assert wet.status_code == 200, wet.text
        assert wet.json()["dry_run"] is False
        assert wet.json()["result"] == 1
        run_id = wet.json()["run_id"]

        gone = client.get(f"/v1/users/eve/memories/{mid}")
        assert gone.status_code == 404

        rb = client.post(
            "/v1/users/eve/maintenance/rollback",
            json={"run_id": run_id},
        )
        assert rb.status_code == 200, rb.text
        assert rb.json()["restored"] == 1
        back = client.get(f"/v1/users/eve/memories/{mid}")
        assert back.status_code == 200

        tasks = client.get("/v1/users/eve/maintenance/tasks")
        assert tasks.status_code == 200
        by_name = {t["name"]: t for t in tasks.json()}
        assert by_name["expire_cleanup"]["mutates"] is True
        assert by_name["consolidate"]["default_run_all"] is True
        assert by_name["reconcile_twins"]["mutates"] is True
        assert by_name["pattern_audit"]["mutates"] is False


def test_store_list_all_includes_superseded_rows():
    from voltmem.domains import MemoryItem
    from voltmem.store import MemoryStore

    store = MemoryStore(":memory:")
    try:
        store.insert(MemoryItem(
            id="a", content="old", domain="session_occasion",
            source="explicit_statement", namespace="alice",
            created_at=1.0, last_confirmed_at=1.0, superseded_by="b",
        ))
        store.insert(MemoryItem(
            id="b", content="new", domain="session_occasion",
            source="explicit_statement", namespace="alice",
            created_at=2.0, last_confirmed_at=2.0,
        ))
        store.insert(MemoryItem(
            id="c", content="other tenant", domain="session_occasion",
            source="explicit_statement", namespace="bob",
            created_at=3.0, last_confirmed_at=3.0,
        ))
        assert [row.id for row in store.list_all("alice")] == ["a", "b"]
        assert [row.id for row in store.all_active(namespace="alice")] == ["b"]
    finally:
        store.close()


def test_graph_supersedes_keeps_list_active_only():
    with _client(VOLTMEM_DB_PATH=":memory:") as client:
        added = client.post(
            "/v1/users/alice/memories",
            json={"data": "outfit for the wedding"},
        )
        assert added.status_code == 200, added.text
        old_id = added.json()["id"]

        mem = client.app.state.pool.for_user("alice")
        stored = mem.layer._store.get(old_id)
        assert stored is not None
        updated = mem.layer.observe(
            "outfit for the date night",
            domain=stored.domain,
            mismatch_magnitude=0.9,
            source="explicit_statement",
            force_update=True,
        )
        assert updated.action == "audited"
        new_id = updated.item.id

        listed = client.get("/v1/users/alice/memories").json()
        assert [row["id"] for row in listed] == [new_id]

        graph = client.get("/v1/users/alice/graph")
        assert graph.status_code == 200, graph.text
        body = graph.json()
        by_id = {node["id"]: node for node in body["nodes"]}
        assert set(by_id) == {old_id, new_id}
        assert by_id[old_id]["active"] is False
        assert by_id[old_id]["superseded_by"] == new_id
        assert by_id[new_id]["active"] is True
        assert by_id[old_id]["memory"] == "outfit for the wedding"
        supersedes = [edge for edge in body["edges"] if edge["kind"] == "supersedes"]
        assert supersedes == [{
            "source": old_id,
            "target": new_id,
            "kind": "supersedes",
        }]

        active_only = client.get(
            "/v1/users/alice/graph",
            params={"include_inactive": "false"},
        )
        assert active_only.status_code == 200
        assert [node["id"] for node in active_only.json()["nodes"]] == [new_id]
        assert active_only.json()["edges"] == []

        other = client.get("/v1/users/bob/graph")
        assert other.json() == {"nodes": [], "edges": []}


def test_graph_facet_edges_are_a_star():
    with _client(VOLTMEM_DB_PATH=":memory:") as client:
        added = client.post(
            "/v1/users/alice/events",
            json={
                "event_id": "look-1",
                "facets": [
                    {"content": "navy suit", "domain": "style_preference"},
                    {"content": "wedding", "domain": "session_occasion"},
                    {"content": "no wool", "domain": "style_constraint"},
                ],
            },
        )
        assert added.status_code == 200, added.text
        ids = [row["id"] for row in added.json()]
        assert len(ids) == 3

        graph = client.get("/v1/users/alice/graph")
        assert graph.status_code == 200, graph.text
        body = graph.json()
        nodes = body["nodes"]
        assert {node["id"] for node in nodes} == set(ids)
        assert all(node["event_id"] == "look-1" for node in nodes)
        hub = nodes[0]["id"]
        facets = [edge for edge in body["edges"] if edge["kind"] == "facet"]
        assert facets == [
            {"source": hub, "target": node["id"], "kind": "facet"}
            for node in nodes[1:]
        ]
        assert [edge for edge in body["edges"] if edge["kind"] == "supersedes"] == []

        listed = client.get("/v1/users/alice/memories").json()
        assert {row["id"] for row in listed} == set(ids)


def test_ui_shell_is_open_and_has_list_inspect():
    previous = os.environ.get("VOLTMEM_API_KEY")
    try:
        with _client(
            VOLTMEM_DB_PATH=":memory:",
            VOLTMEM_API_KEY="test-secret",
        ) as client:
            bare = client.get("/ui", follow_redirects=False)
            assert bare.status_code in (307, 308)
            assert bare.headers["location"].endswith("/ui/")

            page = client.get("/ui/")
            assert page.status_code == 200, page.text
            assert "text/html" in page.headers["content-type"]
            body = page.text
            assert 'id="memory-browser"' in body
            assert "sessionStorage" in body
            assert "/v1/users/" in body
            assert "filter-text" in body
            assert "filter-domain" in body
            assert "filter-source" in body
            assert "effective_volatility" in body
            assert "protection_weight" in body
            assert 'id="view-graph"' in body
            assert "/graph" in body
            assert "edge-supersedes" in body
            assert "edge-facet" in body
            assert client.get("/health").status_code == 200
            denied = client.get("/v1/users/alice/memories")
            assert denied.status_code == 401
    finally:
        if previous is None:
            os.environ.pop("VOLTMEM_API_KEY", None)
        else:
            os.environ["VOLTMEM_API_KEY"] = previous


if __name__ == "__main__":
    tests = [
        test_health,
        test_add_search_domain_stats_delete,
        test_namespace_isolation,
        test_api_key_required_when_set,
        test_clear_and_summary,
        test_occasion_domain,
        test_maintenance_dry_run_gates_expire_cleanup,
        test_store_list_all_includes_superseded_rows,
        test_graph_supersedes_keeps_list_active_only,
        test_graph_facet_edges_are_a_star,
        test_ui_shell_is_open_and_has_list_inspect,
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
