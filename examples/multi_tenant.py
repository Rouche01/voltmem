"""
Multi-tenant quickstart — one database, many users.
===================================================

Each user gets an isolated memory space via namespace. Views share one SQLite
connection and file; only reads/writes are scoped per tenant.

Run:
    .venv/bin/python examples/multi_tenant.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voltmem import MemoryLayer  # noqa: E402


def main():
    db = os.path.join(tempfile.gettempdir(), "voltmem_multi_tenant_demo.db")
    mem = MemoryLayer(db)
    alice = mem.for_user("alice")
    bob = mem.for_user("bob")

    print(f"database: {db}\n")

    alice.remember("I live in Berlin")
    alice.remember("I prefer concise answers")
    bob.remember("I live in Paris")
    bob.remember("I prefer detailed explanations")

    for name, view in [("alice", alice), ("bob", bob)]:
        print(f"--- {name} ---")
        print(f"  summary: {view.summary()}")
        print(f"  where?   {view.recall('where does the user live', top_k=1)}")
        print(f"  style?   {view.recall('prefer answers format', top_k=1)}")
        print()

    mem.close()
    print("Alice and Bob share one DB but never see each other's memories.")


if __name__ == "__main__":
    main()
