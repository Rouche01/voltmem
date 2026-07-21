# VoltMem HTTP sidecar

HTTP surface over [`create_memory`](../voltmem/client.py) for TypeScript / Cloudflare
Workers and other non-Python clients. The engine stays Python; callers use REST
(or the upcoming `@voltmem/client` SDK).

Default domain profile: **stylens** (stable style prefs vs volatile occasion) —
same priors as [`examples/custom_classifier.py`](../examples/custom_classifier.py).

## Install & run (local)

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
| `VOLTMEM_DB_PATH` | `voltmem_sidecar.db` | SQLite path (use a volume in Docker) |
| `VOLTMEM_EMBEDDINGS` | `1` (truthy) | `0`/`false` disables embedder (hashing fallback) |
| `VOLTMEM_API_KEY` | _(empty)_ | When set, require matching `X-API-Key` on `/v1/*` |
| `VOLTMEM_PROFILE` | `stylens` | Domain registry + classifier profile |
| `HOST` | `0.0.0.0` | Bind address (`python -m sidecar`) |
| `PORT` | `8080` | Listen port |

`GET /health` is always unauthenticated.

## API

| Method | Path |
|---|---|
| GET | `/health` |
| POST | `/v1/users/{user_id}/memories` |
| GET | `/v1/users/{user_id}/memories/search?q=&limit=&min_score=` |
| GET | `/v1/users/{user_id}/memories` |
| GET | `/v1/users/{user_id}/memories/{memory_id}` |
| DELETE | `/v1/users/{user_id}/memories/{memory_id}` |
| DELETE | `/v1/users/{user_id}/memories` (clear) |
| GET | `/v1/users/{user_id}/summary` |
| GET | `/v1/users/{user_id}/domain_stats` |

### Add

```bash
curl -s -X POST "http://127.0.0.1:8080/v1/users/alice/memories" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $VOLTMEM_API_KEY" \
  -d '{"data":"I prefer darker colors and minimal fits"}'
```

Body: `{ "data": <string | message | messages>, "source"?: "...", "extract"?: bool }`.

### Search

```bash
curl -s "http://127.0.0.1:8080/v1/users/alice/memories/search?q=style%20preferences&limit=5" \
  -H "X-API-Key: $VOLTMEM_API_KEY"
```

### Domain stats (prior calibration)

```bash
curl -s "http://127.0.0.1:8080/v1/users/alice/domain_stats" \
  -H "X-API-Key: $VOLTMEM_API_KEY"
```

## Docker

From the repo root:

```bash
docker build -t voltmem-sidecar .
docker run --rm -p 8080:8080 \
  -e VOLTMEM_API_KEY=dev-secret \
  -v voltmem-data:/data \
  voltmem-sidecar
```

Image installs `.[sidecar,embeddings]`, stores the DB at `/data/voltmem.db`, and
exposes port **8080**. First start may take longer while the embedding model loads.

## Multi-tenant

One process / one SQLite file; `{user_id}` selects the namespace. Tenants never
see each other's memories.
