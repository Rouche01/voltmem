"""Tests for the VoltMem HTTP sidecar."""

from __future__ import annotations

import json
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
from sidecar.profiles import apply_domains_file, build_profile  # noqa: E402


def _client(**env: str) -> TestClient:
    """Fresh app with env applied for lifespan."""
    if "VOLTMEM_API_KEY" not in env:
        os.environ.pop("VOLTMEM_API_KEY", None)
    if "VOLTMEM_DOMAINS_FILE" not in env:
        os.environ.pop("VOLTMEM_DOMAINS_FILE", None)
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


def test_add_domain_skips_classifier():
    """A write-supplied domain is stored even when the text would classify as style."""
    with _client(VOLTMEM_DB_PATH=":memory:") as client:
        add = client.post(
            "/v1/users/alice/memories",
            json={
                "data": "I prefer darker colors and minimal fits",
                "domain": "community_outcome",
            },
        )
        assert add.status_code == 200, add.text
        body = add.json()
        assert body["domain"] == "community_outcome"

        listed = client.get("/v1/users/alice/memories")
        assert listed.status_code == 200
        assert [row["domain"] for row in listed.json()] == ["community_outcome"]

        graph = client.get("/v1/users/alice/graph")
        assert graph.status_code == 200, graph.text
        nodes = graph.json()["nodes"]
        assert [node["domain"] for node in nodes] == ["community_outcome"]


def test_domains_file_merges_kinds_and_searches():
    spec = {
        "domains": [
            {"name": "community_preference", "volatility": 0.20},
            {"name": "community_outcome", "volatility": 0.55},
        ],
        "keywords": {
            "community_outcome": ["[outcome]", "aborted"],
            "community_preference": ["[preference]", "allowlist"],
        },
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    Path(path).write_text(json.dumps(spec), encoding="utf-8")
    try:
        domains, classifier = apply_domains_file(*build_profile("stylens"), path)
        assert domains.volatility("community_outcome") == 0.55
        assert domains.volatility("style_preference") == 0.08
        restore = domains.install()
        try:
            assert classifier.classify_domain("[outcome] aborted the post") == "community_outcome"
            assert classifier.classify_domain("I prefer darker colors") == "style_preference"
        finally:
            restore()

        with _client(VOLTMEM_DB_PATH=":memory:", VOLTMEM_DOMAINS_FILE=path) as client:
            add = client.post(
                "/v1/users/relay-local/memories",
                json={"data": "[outcome] aborted the community post"},
            )
            assert add.status_code == 200, add.text
            body = add.json()
            assert body["domain"] == "community_outcome"

            stats = client.get("/v1/users/relay-local/domain_stats")
            assert stats.status_code == 200
            assert stats.json()["community_outcome"]["prior"] == 0.55

            search = client.get(
                "/v1/users/relay-local/memories/search",
                params={"q": "aborted community post", "limit": 3},
            )
            assert search.status_code == 200
            hits = search.json()
            assert any(
                hit["id"] == body["id"] and hit["domain"] == "community_outcome"
                for hit in hits
            )
    finally:
        os.environ.pop("VOLTMEM_DOMAINS_FILE", None)
        os.remove(path)


def test_domains_file_rejects_unregistered_keyword():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    Path(path).write_text(
        json.dumps({"keywords": {"community_outcome": ["[outcome]"]}}),
        encoding="utf-8",
    )
    try:
        try:
            apply_domains_file(*build_profile("stylens"), path)
        except ValueError as exc:
            assert "community_outcome" in str(exc)
        else:
            raise AssertionError("expected ValueError")
    finally:
        os.remove(path)


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
            assert client.get("/v1/tenants/alice/memories").status_code == 401

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


def test_clear_domain_keeps_other_kinds():
    with _client(VOLTMEM_DB_PATH=":memory:") as client:
        a = client.post(
            "/v1/tenants/domain-clear/memories",
            json={"data": "I prefer navy linen", "domain": "style_preference"},
        )
        b = client.post(
            "/v1/tenants/domain-clear/memories",
            json={"data": "Dressing for a beach wedding", "domain": "session_occasion"},
        )
        assert a.status_code == 200, a.text
        assert b.status_code == 200, b.text

        bad = client.delete("/v1/tenants/domain-clear/memories?domain=")
        assert bad.status_code == 400

        cleared = client.delete(
            "/v1/tenants/domain-clear/memories?domain=style_preference"
        )
        assert cleared.status_code == 200, cleared.text
        body = cleared.json()
        assert body["cleared"] is True
        assert body["domain"] == "style_preference"

        remaining = client.get("/v1/tenants/domain-clear/memories").json()
        assert len(remaining) == 1
        assert remaining[0]["domain"] == "session_occasion"
        assert remaining[0]["id"] == b.json()["id"]

        graph = client.get("/v1/tenants/domain-clear/graph").json()
        assert len(graph["nodes"]) == 1
        assert graph["nodes"][0]["domain"] == "session_occasion"


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


def test_tenants_path_aliases_users():
    with _client(VOLTMEM_DB_PATH=":memory:") as client:
        added = client.post(
            "/v1/tenants/alice/memories",
            json={"data": "I live in Berlin"},
        )
        assert added.status_code == 200, added.text
        mid = added.json()["id"]
        via_users = client.get("/v1/users/alice/memories")
        assert via_users.status_code == 200, via_users.text
        assert via_users.json()[0]["id"] == mid
        via_tenants = client.get("/v1/tenants/alice/memories")
        assert via_tenants.json()[0]["id"] == mid

        spec = client.get("/openapi.json")
        assert spec.status_code == 200, spec.text
        paths = spec.json()["paths"]
        users_get = paths["/v1/users/{tenant_id}/memories"]["get"]
        tenants_get = paths["/v1/tenants/{tenant_id}/memories"]["get"]
        assert users_get.get("deprecated") is True
        assert tenants_get.get("deprecated") is not True


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
            assert "/v1/tenants/" in body
            assert "filter-text" in body
            assert "filter-domain" in body
            assert "filter-source" in body
            assert "effective_volatility" in body
            assert "protection_weight" in body
            assert 'id="view-graph"' in body
            assert 'id="clear-tenant"' in body
            assert 'id="clear-domain"' in body
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
        test_add_domain_skips_classifier,
        test_domains_file_merges_kinds_and_searches,
        test_domains_file_rejects_unregistered_keyword,
        test_namespace_isolation,
        test_api_key_required_when_set,
        test_clear_and_summary,
        test_clear_domain_keeps_other_kinds,
        test_occasion_domain,
        test_maintenance_dry_run_gates_expire_cleanup,
        test_store_list_all_includes_superseded_rows,
        test_graph_supersedes_keeps_list_active_only,
        test_graph_facet_edges_are_a_star,
        test_tenants_path_aliases_users,
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
