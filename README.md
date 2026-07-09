# VoltMem

**Current-truth memory for LLM agents.**

Most memory layers treat every fact the same — your hometown and today's mood get equal
weight. That forces a bad tradeoff: go **stale** on fast-changing facts, or get
**corrupted** when a confident-but-wrong update overwrites something durable.

VoltMem scales protection and retrieval freshness by **how fast each kind of fact
actually changes**. Volatile facts update; stable facts resist corruption; stale
volatile memories rank lower at search time.

> Mem0 remembers relevant facts. VoltMem remembers **current truth**.

**Research & benchmarks:** [docs/RESEARCH.md](docs/RESEARCH.md)

---

## Install

```bash
pip install -e ".[embeddings]"    # from a clone
# or, when published:
# pip install voltmem[embeddings]
```

Core library has **zero required dependencies**. Embeddings extras pull in
`sentence-transformers` (recommended). LangChain: `pip install -e ".[langchain]"`.

---

## Quickstart

```python
from voltmem import create_memory

mem = create_memory("app.db", user_id="alice")

mem.add("I live in Berlin")
mem.add("I prefer concise, direct answers")
mem.add("Actually I moved to Paris last month")   # updates location, not prefs

hits = mem.search("where does the user live?", limit=3)
print(hits[0]["memory"])   # Actually I moved to Paris last month
```

### Message pairs

```python
mem.add([
    {"role": "user", "content": "I'm working on a Postgres migration"},
    {"role": "assistant", "content": "Got it — I'll keep database context in mind."},
])
```

### Inject into a prompt

```python
memories = mem.search(user_message, limit=5)
context = "\n".join(f"- {m['memory']}" for m in memories)
system = f"What you know about this user:\n{context}"
```

---

## API

| Method | Description |
|---|---|
| `create_memory(db, user_id)` | Factory with auto-detected embeddings |
| `Memory.add(text \| messages)` | Store a fact; updates related memories when appropriate |
| `Memory.search(query, limit=5)` | Ranked memories (relevance + freshness) |
| `Memory.get_all()` | All active memories for this user |
| `Memory.delete(id)` | Remove one memory |
| `Memory.clear()` | Wipe user namespace |

Advanced: `mem.layer` exposes `MemoryLayer` for low-level `observe()` / `write()`.

---

## Why VoltMem

| Problem | ADD-only memory | VoltMem |
|---|---|---|
| User moves cities | Berlin and Paris both stored | **Updates** to current city |
| Old project name in haystack | Ranks by similarity | **Down-ranks** stale volatile facts |
| Confident wrong blip on stable pref | Often accepted | **Resists** corruption |

Run the side-by-side demo:

```bash
python examples/contradiction_demo.py
```

---

## Integrations

### LangChain

```bash
pip install -e ".[langchain]"
python examples/langchain_agent.py
```

```python
from voltmem.integrations.langchain import VoltMemMemory

memory = VoltMemMemory(session_id="user-42", db_path="app.db")
memory.load_memory_variables({"input": "Where do I live?"})
memory.save_context({"input": "I moved to Paris"}, {"output": "Noted."})
```

### Multi-tenant

One SQLite file, many users — `user_id` maps to an isolated namespace:

```python
alice = create_memory("app.db", user_id="alice")
bob   = create_memory("app.db", user_id="bob")
```

---

## Examples

| Script | What it shows |
|---|---|
| `examples/contradiction_demo.py` | VoltMem vs always-add on contradictions |
| `examples/quickstart_batteries.py` | `remember()` / `recall()` low-level API |
| `examples/multi_tenant.py` | One DB, many users |
| `examples/langchain_agent.py` | LangChain adapter |

---

## Domain volatility priors

| Domain | Volatility | Behavior |
|---|---|---|
| `personality_trait` | 0.05 | Very protected |
| `core_preference` | 0.08 | Very protected |
| `biographical` | 0.10 | High protection |
| `current_project` | 0.55 | Updates readily |
| `emotional_context` | 0.80 | Fast-moving |
| `current_task` | 0.90 | Minimal protection |

Custom domains: `voltmem/domains.py`.

---

## Development

```bash
pip install -e ".[all]"
python tests/test_voltmem.py
python tests/test_client.py
```

Experiments and benchmarks live in `experiments/` — see [docs/RESEARCH.md](docs/RESEARCH.md).

---

## License

MIT
