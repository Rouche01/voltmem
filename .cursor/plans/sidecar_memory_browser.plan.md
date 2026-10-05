---
name: Sidecar memory browser + graph UI
overview: Ship a same-origin memory browser on the VoltMem sidecar — list + inspect first, then a small graph over supersedes/facet edges — so dogfood tenants (relay-os, stylens) can see what is stored without curling JSON.
todos:
  - id: graph-api
    content: Add GET /v1/users/{user_id}/graph — all rows (active + superseded), nodes + supersedes/facet edges; optional include_inactive query; keep list/search active-only
    status: completed
  - id: store-list-all
    content: Store/Memory helper to list every row for a namespace (not only superseded_by IS NULL); reuse for graph; thin tests in test_sidecar / store
    status: completed
  - id: static-ui-shell
    content: sidecar/static/index.html + FastAPI StaticFiles mount at /ui; sessionStorage for API key + user_id; same-origin fetch to /v1/*
    status: completed
  - id: list-inspect
    content: List pane from GET /memories (or graph nodes filtered active); filter domain/source/text; detail pane from GET /memories/{id} inspect fields
    status: completed
  - id: graph-view
    content: Second pane — SVG or vendored Cytoscape; color by domain; fade inactive; edges supersedes + facet only; no all-pairs similarity in v1
    status: completed
  - id: docs-smoke
    content: Document /ui + /graph in sidecar/README.md and docs/SIDECAR.md; one curl + browser smoke for local Docker
    status: completed
isProject: true
---

# Sidecar memory browser + graph UI

Canonical plan for an **operator / dogfood UI** inside the VoltMem sidecar. Open while the `voltmem` workspace is active.

**Related (do not merge):** public demo walkthrough lives in `[ts_playground_sibling_11142e17.plan.md](./ts_playground_sibling_11142e17.plan.md)` (sibling CF Worker). This plan is the **sidecar-local** inspector for real tenants (`relay-local`, stylens users, etc.).

## What this achieves

Today the only way to see stored items is HTTP JSON (`GET .../memories`, `.../search`, `.../memories/{id}`). Relay and other dogfood apps write facts through `@voltmem/client`; there is no screen that shows current-truth, replacement chains, or multi-facet events.

Ship a **quick browser on the sidecar itself**:

1. **List** — scan active memories, filter, open inspect detail
2. **Graph** — draw the edges that already exist in SQLite (`superseded_by`, shared `event_id`)

Not a product dashboard. Not a public playground. Same process, same API key, same DB as production sidecar.

## Non-goals


| Out                                         | Why                                                                               |
| ------------------------------------------- | --------------------------------------------------------------------------------- |
| Embedding / similarity edges in v1          | Not stored; all-pairs on every load is expensive and noisy                        |
| Exposing API key via a public Worker        | Operator UI only; key stays in browser sessionStorage on localhost / private host |
| Changing Relay write path                   | CommunityEngager keeps `add()`; graph fills as supersedes / `add_event` happen    |
| Replacing `/memories` active-only semantics | List/search stay current-truth; graph is the history-aware view                   |
| Full SPA build (Vite/React) in v1           | One static HTML (+ optional small vendored JS) keeps the Docker image Python-only |


## Architecture

```mermaid
flowchart LR
  browser[Browser /ui]
  sidecar[VoltMem sidecar]
  db[(SQLite)]

  browser -->|"X-API-Key + user_id"| sidecar
  sidecar --> db
  sidecar -->|"GET /v1/.../memories"| list[Active list]
  sidecar -->|"GET /v1/.../graph"| graph[Nodes + edges]
  sidecar -->|"GET /v1/.../memories/id"| inspect[Scoring inspect]
```



- **Same origin** — mount static files from FastAPI; UI calls `/v1/...` without CORS.
- **Auth** — existing `X-API-Key` on `/v1/`*; `/ui` and `/health` stay open (UI shell only; data still requires the key).
- **Tenant** — `{user_id}` path segment (Relay default: `relay-local` via `VOLTMEM_USER_ID`).

## Data model (edges that matter)

From `[voltmem/domains.py](../../voltmem/domains.py)` `MemoryItem`:


| Field                                 | Graph meaning                                  |
| ------------------------------------- | ---------------------------------------------- |
| `superseded_by`                       | Directed edge old → new (`kind: "supersedes"`) |
| `event_id`                            | Clique or star among facets (`kind: "facet"`)  |
| `is_active` / `superseded_by is None` | Active vs replaced (fade replaced nodes)       |


`GET /v1/users/{user_id}/memories` today returns **active only** via `get_all()` → `_layer._active()`. The graph endpoint must include superseded rows or replacement chains never appear.

Relay’s current writes are mostly flat `add()` facts (`[community] …`). Expect a useful **list** immediately; **graph density** grows when supersedes and `add_event` run (maintenance / multi-facet paths).

## Phase 1 — Graph API

### Store / Memory

Add a small helper (name TBD: `list_all` / `get_history`) on the store or `Memory` client:

- Scope: one `namespace` (== `user_id`)
- Include rows with `superseded_by` set
- Order by `created_at` (stable for UI)

Do **not** change `get_all()` active-only contract used by list/search.

### HTTP

```http
GET /v1/users/{user_id}/graph
Authorization: X-API-Key
```

Optional query: `include_inactive=1` (default true for graph; or always include inactive and let UI filter).

Response shape:

```json
{
  "nodes": [
    {
      "id": "…",
      "memory": "…",
      "domain": "…",
      "source": "…",
      "active": true,
      "superseded_by": null,
      "event_id": null,
      "created_at": 0,
      "last_confirmed_at": 0
    }
  ],
  "edges": [
    { "source": "old-id", "target": "new-id", "kind": "supersedes" },
    { "source": "a", "target": "b", "kind": "facet" }
  ]
}
```

Edge rules:

- For each node with `superseded_by`, emit one `supersedes` edge.
- For each `event_id` with ≥2 nodes, emit undirected (or ordered) `facet` edges between them (or hub-and-spoke to first id — pick one and document).

Later (not v1): `?similar=0.8` optional pairwise scores — only if measured and capped.

Tests: extend `[tests/test_sidecar.py](../../tests/test_sidecar.py)` — insert A, supersede with B, assert B active in `/memories`, both in `/graph`, one supersedes edge.

## Phase 2 — Static UI shell

```text
sidecar/
  static/
    index.html          # list + graph + inspect
    app.js              # optional; keep inline if tiny
    vendor/             # optional Cytoscape / force layout, vendored
  app.py                # mount StaticFiles at /ui
```

- Open `http://127.0.0.1:8080/ui`
- Prompt once for API key + user id; store in `sessionStorage`
- No npm build step in the VoltMem image

Wire in `[sidecar/app.py](../../sidecar/app.py)` with `StaticFiles` (or a single `FileResponse` for `/ui` → `index.html`). Prefer `/ui/` trailing-slash friendly mount.

## Phase 3 — List + inspect


| Pane          | Source                                                                                                                      |
| ------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Table / cards | `GET .../memories` or active filter on graph nodes                                                                          |
| Filters       | domain, source substring, text substring                                                                                    |
| Detail        | `GET .../memories/{id}` — volatility, protection, staleness, surprise, mismatch, age (`[inspect](../../voltmem/memory.py)`) |


Actions (optional v1.1, mark pending if timeboxed): delete one memory (existing DELETE). Do not ship clear-all in the UI without a typed confirm.

## Phase 4 — Graph view

- Toggle List | Graph (or split view on wide screens)
- Color nodes by `domain`
- Opacity for `active: false`
- Draw only `supersedes` + `facet` edges
- Click node → same inspect panel as list
- Empty-edge state: still show force/cluster by domain so Relay’s flat facts are scannable as a cloud, not a blank canvas

Implementation preference: plain SVG + simple force (or vendored Cytoscape under `static/vendor/`) — no CDN dependency in air-gapped / ThinkPad dogfood if avoidable; if CDN is used, document offline fallback.

## Docs + smoke

Update:

- `[sidecar/README.md](../../sidecar/README.md)` — API table row for `/graph`; “Memory browser” section (`/ui`)
- `[docs/SIDECAR.md](../../docs/SIDECAR.md)` — one paragraph for operators

Local smoke:

```bash
# sidecar up (Docker or python -m sidecar)
open http://127.0.0.1:8080/ui
# paste VOLTMEM_API_KEY + user_id (e.g. relay-local)
curl -s "$BASE/v1/users/relay-local/graph" -H "X-API-Key: $VOLTMEM_API_KEY" | jq '.nodes | length'
```

## Dogfood consumers


| Consumer                  | `user_id`                              | Expectation                                                                                             |
| ------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| relay-os CommunityEngager | `VOLTMEM_USER_ID` (e.g. `relay-local`) | List fills from learn/approve/abort/discovery facts; few edges until supersedes                         |
| stylens / other           | their tenant ids                       | Same UI                                                                                                 |
| ThinkPad `/opt/voltmem`   | bind localhost only                    | UI on `127.0.0.1:8080/ui` — do not expose `/ui` or API on public interfaces without auth/network policy |


## Complexity checklist

- Keep Docker image without Node build
- Fail closed on bad API key (existing auth)
- Graph payload size: fine for hundreds of rows; if thousands, add `limit` / cursor later
- Do not expand `@voltmem/client` until a second consumer needs typed `graph()` — curl/UI first is enough

## Success criteria

- [x] `/graph` returns active + superseded nodes and supersedes/facet edges; tests green
- [x] `/ui` loads against local sidecar; list + inspect work with Relay’s user id
- [x] Graph view renders replacement chain when a test/fixture supersedes a fact
- [x] SIDECAR + sidecar README document how to open the browser

## Suggested order of work

1. `store-list-all` + `graph-api` + sidecar tests
2. `static-ui-shell` + `list-inspect`
3. `graph-view`
4. `docs-smoke`

## Handoff notes

- Public marketing demo remains the **sibling playground** plan — different threat model (BFF hides key).
- Associative / co-occurrence graphs from sleeptime (`[docs/SCHEDULE.md](../../docs/SCHEDULE.md)` P3) are a future edge `kind`, not a blocker for this UI.

