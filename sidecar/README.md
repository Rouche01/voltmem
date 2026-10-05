# VoltMem HTTP sidecar

HTTP surface over [`create_memory`](../voltmem/client.py) for TypeScript / Cloudflare
Workers and other non-Python clients. The engine stays Python; callers use REST
or [`@voltmem/client`](../clients/typescript).

**Deploy guide (start here):** [docs/SIDECAR.md](../docs/SIDECAR.md)

Default domain profile: **stylens** (stable style prefs vs volatile occasion) —
same priors as [`examples/custom_classifier.py`](../examples/custom_classifier.py).

## Concepts

| Concept | Role | Example |
|---|---|---|
| **Tenant** | Isolation boundary (person or app bucket). Stored as SQLite `namespace`. | `relay-local`, `alice` |
| **Domain** | Fact kind plus a volatility prior, inside one tenant. | `style_preference` |
| **Memory item** | One stored fact. | approve/abort outcome text |
| **Profile** | Built-in registry and classifier (`VOLTMEM_PROFILE`). Extra kinds come from `VOLTMEM_DOMAINS_FILE`. | `stylens` |

HTTP addresses a tenant as `/v1/tenants/{tenant_id}`. `/v1/users/{tenant_id}/…` is the same handlers, marked deprecated. The profile classifier assigns `domain` on write. Full glossary: [docs/SIDECAR.md](../docs/SIDECAR.md#concepts).

## Quick start

### Docker (anyone)

```bash
# Build from this repo's Dockerfile
git clone https://github.com/Rouche01/voltmem.git && cd voltmem
docker build -t voltmem-sidecar .
docker run --rm -p 8080:8080 \
  -e VOLTMEM_API_KEY=dev-secret \
  -v voltmem-data:/data \
  voltmem-sidecar

# Or pull a release image (after GHCR publish):
# docker pull ghcr.io/rouche01/voltmem-sidecar:latest
```

### Local Python

```bash
pip install -e ".[sidecar]"
# production-quality search:
# pip install -e ".[sidecar,embeddings]"

export VOLTMEM_DB_PATH=./voltmem_sidecar.db
export VOLTMEM_EMBEDDINGS=0          # 1 when sentence-transformers is installed
export VOLTMEM_API_KEY=dev-secret    # optional locally; required in production
export PORT=8080

python -m sidecar
# or: uvicorn sidecar.app:app --host 0.0.0.0 --port 8080
```

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `VOLTMEM_DB_PATH` | `voltmem_sidecar.db` | SQLite path (use `/data/voltmem.db` in Docker). File DBs use WAL automatically. |
| `VOLTMEM_EMBEDDINGS` | `1` (truthy) | `0`/`false` disables embedder (hashing fallback) |
| `VOLTMEM_API_KEY` | _(empty)_ | When set, require matching `X-API-Key` on `/v1/*` |
| `VOLTMEM_PROFILE` | `stylens` | Built-in domain registry and classifier. One process, one profile. |
| `VOLTMEM_DOMAINS_FILE` | _(unset)_ | JSON file merged at startup: `domains` (`name`, `volatility`, optional `slot`) and optional `keywords`. Keyword hints run before the profile classifier. |
| `VOLTMEM_MAINTENANCE` | `1` | Background daemon for due tasks (`0` to disable) |
| `VOLTMEM_MAINTENANCE_CHECK_INTERVAL` | `60` | Seconds between daemon ticks |
| `VOLTMEM_EXPIRE_INTERVAL` | `3600` | Min seconds between `expire_cleanup` per tenant |
| `VOLTMEM_PATTERN_AUDIT_INTERVAL` | `3600` | Min seconds between `pattern_audit` |
| `VOLTMEM_RECLASSIFY_INTERVAL` | `86400` | Min seconds between `reclassify_ambiguous` |
| `VOLTMEM_CONSOLIDATE` | `1` | Include consolidate in the daemon (`0` to disable) |
| `VOLTMEM_CONSOLIDATE_INTERVAL` | `86400` | Min seconds between `consolidate` |
| `VOLTMEM_RECONCILE_TWINS` | `1` | Include twin reconciliation in the daemon (`0` to disable) |
| `VOLTMEM_RECONCILE_INTERVAL` | `86400` | Min seconds between `reconcile_twins` |
| `VOLTMEM_VERIFY_ON_WRITE` | `0` | `1` asks the 14B verifier on grey `add()` / `remember()`; default waits for sleeptime |
| `HOST` | `0.0.0.0` | Bind address (`python -m sidecar`) |
| `PORT` | `8080` | Listen port |

`GET /health` and `GET /ui` are always unauthenticated. `/v1/*` still requires `X-API-Key` when `VOLTMEM_API_KEY` is set.

## API

| Method | Path |
|---|---|
| GET | `/health` |
| POST | `/v1/tenants/{tenant_id}/memories` |
| GET | `/v1/tenants/{tenant_id}/memories/search?q=&limit=&min_score=` |
| GET | `/v1/tenants/{tenant_id}/memories` |
| GET | `/v1/tenants/{tenant_id}/graph?include_inactive=` |
| GET | `/v1/tenants/{tenant_id}/memories/{memory_id}` |
| GET | `/ui` |
| DELETE | `/v1/tenants/{tenant_id}/memories/{memory_id}` |
| DELETE | `/v1/tenants/{tenant_id}/memories` (clear) |
| GET | `/v1/tenants/{tenant_id}/summary` |
| GET | `/v1/tenants/{tenant_id}/domain_stats` |
| POST | `/v1/tenants/{tenant_id}/events` |
| GET | `/v1/tenants/{tenant_id}/events/{event_id}` |
| POST | `/v1/tenants/{tenant_id}/maintenance/trigger` |
| POST | `/v1/tenants/{tenant_id}/maintenance/rollback` |
| GET | `/v1/tenants/{tenant_id}/maintenance/tasks` |

### Maintenance

**Background daemon (default on):** a sidecar thread periodically runs
`run_due` per tenant for `expire_cleanup`, `pattern_audit`,
`reclassify_ambiguous`, `consolidate`, and `reconcile_twins`. Disable the
whole daemon with `VOLTMEM_MAINTENANCE=0`. Disable only consolidate with
`VOLTMEM_CONSOLIDATE=0`, or only twin reconciliation with
`VOLTMEM_RECONCILE_TWINS=0`.

Grey writes insert as twins until `reconcile_twins` runs (local 14B). Set
`VOLTMEM_VERIFY_ON_WRITE=1` to ask on the live `add()` path instead.

**Manual trigger:** `POST .../maintenance/trigger` body: `{ "task"?: string, "dry_run"?: bool }`.

- Default **`dry_run: false`** — mutating tasks apply (maintenance maintains).
- Pass **`dry_run: true`** to preview without changing memory.
- Omit ``task`` → default set including `consolidate` and `reconcile_twins`.
- Response includes **`run_id`**. Undo with:

`POST .../maintenance/rollback` body: `{ "run_id": "..." }`

- Flag-only tasks (`reclassify_ambiguous`, `pattern_audit`) are always read-only.

### Add

```bash
curl -s -X POST "http://127.0.0.1:8080/v1/tenants/alice/memories" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $VOLTMEM_API_KEY" \
  -d '{"data":"I prefer darker colors and minimal fits"}'
```

Body: `{ "data": <string | message | messages>, "source"?: "...", "extract"?: bool, "domain"?: "..." }`.

`domain`, when set, is the fact kind and skips the profile classifier. The same phrase that classifies as `style_preference` stays `community_outcome` if the body says so. Event facets already carry their own `domain` on `POST .../events`.

### Search

```bash
curl -s "http://127.0.0.1:8080/v1/tenants/alice/memories/search?q=style%20preferences&limit=5" \
  -H "X-API-Key: $VOLTMEM_API_KEY"
```

### Graph

Active and superseded rows for one tenant, plus the edges already stored in SQLite. List and search stay current-truth only. The `{tenant_id}` segment is that tenant id.

```bash
curl -s "http://127.0.0.1:8080/v1/tenants/relay-local/graph" \
  -H "X-API-Key: $VOLTMEM_API_KEY" | jq '.nodes | length'
```

`include_inactive` defaults to true. `include_inactive=false` returns the same rows as `GET .../memories`.

Each replaced row emits one `supersedes` edge to `superseded_by`. Memories that share an `event_id` form a star: the earliest row is the hub, and each other facet is a `facet` edge from that hub.

### Memory browser

Open [http://127.0.0.1:8080/ui](http://127.0.0.1:8080/ui) while the sidecar is running (Docker or `python -m sidecar`). Paste the same `VOLTMEM_API_KEY` you started the process with, and the tenant id (`relay-local` for Relay). Both stay in this tab’s `sessionStorage`. Leave the key blank when the sidecar was started without `VOLTMEM_API_KEY`.

**List** is `GET .../memories`, with text, domain (kind), and source filters. Domain is the fact kind from the profile classifier, or the `domain` sent on add. A row opens `GET .../memories/{id}` (volatility, protection, staleness, surprise, mismatch, age). **Graph** draws the `/graph` payload in the page: color by domain, faded replaced nodes, solid replacement lines, dashed shared-event lines. Memories with no links still show, grouped by domain.

For a laptop-only sidecar, start with `HOST=127.0.0.1` so `/ui` and `/v1` are not on other interfaces.

### Domain stats (prior calibration)

```bash
curl -s "http://127.0.0.1:8080/v1/tenants/alice/domain_stats" \
  -H "X-API-Key: $VOLTMEM_API_KEY"
```

## TypeScript client

```bash
npm install @voltmem/client
# or from this repo: cd clients/typescript && npm install && npm run build
```

See [clients/typescript/README.md](../clients/typescript/README.md).

## Multi-tenant

One process / one SQLite file. `{tenant_id}` is the tenant id and selects the
`namespace` column. Tenants never see each other's memories. `/v1/users/{tenant_id}/…` still reaches the same rows.
