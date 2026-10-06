# VoltMem — agent guide

Current-truth memory for LLM agents. Volatility decides how hard a fact is to overwrite and how fast it goes stale at search time. Python is the runtime. Lean is the spec for the control law. The HTTP sidecar is how non-Python apps talk to the engine.

Version in `pyproject.toml` is the library version. `@voltmem/client` and the sidecar image tags are versioned separately.

## Layout

| Path | Role |
|---|---|
| `voltmem/` | Library. Zero required dependencies. |
| `voltmem/scoring.py` | Control law (escalation, staleness, retrieval). Runtime source of truth. |
| `voltmem/client.py` | `Memory` and `create_memory()`. |
| `voltmem/domains.py` | Domain priors, `MemoryItem`, `DomainRegistry`. |
| `sidecar/` | FastAPI HTTP API, memory browser (`static/index.html`), maintenance daemon. |
| `clients/typescript/` | `@voltmem/client`. `fetch` only. No engine. |
| `lean/` | Machine-checked spec. No mathlib. Toolchain in `lean/lean-toolchain`. |
| `tests/` | Standalone scripts. Run with `python tests/<file>.py`. |
| `experiments/` | Benchmarks and research scripts. Not the default test suite. |
| `examples/` | Demos, including `examples/chat_app`. |
| `docs/` | Deploy, research, open problems. Start at `docs/SIDECAR.md` and `docs/RESEARCH.md`. |
| `paper/` | Writeups. Do not treat them as the API. |

Public entry point: `from voltmem import create_memory`. Low-level writes go through `MemoryLayer` (`observe` / `write` / `retrieve`).

## Install and verify

```bash
pip install -e ".[embeddings,sidecar]"
```

Core tests (this is what CI runs for Python):

```bash
python tests/test_voltmem.py
python tests/test_lean_oracle.py
python experiments/voltmem_eval.py
python tests/test_client.py
VOLTMEM_EMBEDDINGS=0 VOLTMEM_MAINTENANCE=0 python tests/test_sidecar.py
VOLTMEM_EMBEDDINGS=0 VOLTMEM_MAINTENANCE=0 python tests/test_maintenance_scheduler.py
python tests/test_vector_index.py
python tests/test_classifiers.py
```

TypeScript (CI uses Node 22):

```bash
cd clients/typescript && npm install && npm test
```

Lean, after a scoring change:

```bash
cd lean && lake build
python tests/test_lean_oracle.py
```

There is no pytest runner. Each test file is a script. CI is `.github/workflows/ci.yml` (Python 3.11 and 3.12). Match that file when you add a test that should be required.

`experiments/` scripts that call OpenAI, Mem0, or a local 14B model are opt-in. Do not run them as a substitute for the core suite.

## Invariants

- **Core stays dependency-free.** `sentence-transformers`, FastAPI, LangChain, and httpx live in extras in `pyproject.toml`. Do not add them to `[project].dependencies`.
- **Scoring and Lean move together.** A change to `voltmem/scoring.py` needs the matching kernel in `lean/VoltMem/ControlLaw.lean`, a green `lake build`, and `tests/test_lean_oracle.py`. `#guard` cases in `lean/VoltMem/Oracle.lean` are locks. Do not retune constants to make one battery pass while the spec disagrees.
- **Default law is composite.** `"homeostatic"` and `"current"` stay available. Do not silently change the default.
- **False merge is worse than a missed link.** Grey frames insert as twins. Live `add()` / `remember()` stay on the heuristic path unless `verify_on_write=True` or `VOLTMEM_VERIFY_ON_WRITE=1`. Twin cleanup is sleeptime `reconcile_twins`.
- **Tenant is the isolation boundary.** Python `user_id` and HTTP `{tenant_id}` are the same id, stored as SQLite `namespace`. A domain is a fact kind plus a volatility prior inside one tenant. Say tenant in docs and the memory browser. `/v1/users/{id}` is a deprecated alias of `/v1/tenants/{id}`.
- **One sidecar process, one profile.** `VOLTMEM_PROFILE` (default `stylens`) plus optional `VOLTMEM_DOMAINS_FILE`. App-specific domains do not get hardcoded into the stylens profile.
- **Search is not raw ANN.** Vector candidates are re-ranked with volatility. Keyword-only mode uses the threshold ladder and can false-merge. The shipped claim is embeddings plus sidecar sleeptime.
- **File databases use WAL.** HTTP and the maintenance daemon share one SQLite file. Do not open it in a mode that breaks that.

## Sidecar images

`Dockerfile` build-arg `EMBEDDINGS`:

- `1` (default): sentence-transformers, tag `:latest` / version pin
- `0`: hashing scorer, tag `:slim` / `:<version>-slim`

Publish workflow: `.github/workflows/publish-sidecar.yml`, on tags `sidecar-v*`. `GET /health` and `GET /ui` stay unauthenticated. `/v1/*` requires `X-API-Key` when `VOLTMEM_API_KEY` is set.

## When you change something

| Change | Also update |
|---|---|
| HTTP route or request body | `sidecar/README.md`, `clients/typescript/src`, and its tests |
| Domain glossary or tenant wording | `docs/SIDECAR.md` and `sidecar/README.md` |
| Env var | sidecar README env table and `docs/SIDECAR.md` if operators need it |
| Scoring constant or law | Lean spec + `tests/test_lean_oracle.py` |
| New required test | `.github/workflows/ci.yml` |
| User-facing behavior | `README.md` only when the public quickstart or API table changes |

Do not commit `.env`, `*.db`, `experiments/out/`, or `lean/.lake/`.

## Style

Match the file you are editing. Module docstrings state the equation or the contract. Tests use plain `assert`s and a `sys.path` insert, not a framework. Prefer a focused test script over growing `test_voltmem.py` when the behavior is a separate surface (TTL, sidecar, classifiers).
