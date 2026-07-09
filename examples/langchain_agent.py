"""
LangChain + VoltMem — minimal memory hook demo (no API key required).
====================================================================

Shows VoltMemMemory wired the way ConversationChain / legacy agents expect:
load_memory_variables before the model call, save_context after.

Install integration deps:
    pip install -r requirements-integrations.txt

Run:
    .venv/bin/python examples/langchain_agent.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from voltmem.integrations.langchain import VoltMemMemory
except ImportError as exc:
    print("Missing LangChain integration dependencies.")
    print("Install with: pip install -r requirements-integrations.txt")
    raise SystemExit(1) from exc

from voltmem import EmbeddingSimilarity  # noqa: E402


def fake_llm(prompt: str) -> str:
    """Stand-in model — prints the prompt and returns a canned reply."""
    print("\n--- prompt sent to model ---")
    print(prompt)
    print("--- end prompt ---\n")
    return "I'll keep that in mind."


def run_turn(
    memory: VoltMemMemory,
    user_input: str,
    *,
    system: str = "You are a helpful assistant.",
) -> str:
    mem_vars = memory.load_memory_variables({"input": user_input})
    history = mem_vars.get(memory.memory_key, "")
    parts = [system]
    if history.strip():
        parts.append(history.strip())
    parts.append(f"Human: {user_input}")
    parts.append("Assistant:")
    prompt = "\n\n".join(parts)
    reply = fake_llm(prompt)
    memory.save_context({"input": user_input}, {"output": reply})
    return reply


def main() -> None:
    sim = EmbeddingSimilarity(verbose=True)
    memory = VoltMemMemory(
        session_id="demo-user",
        db_path=":memory:",
        similarity_fn=sim,
        top_k=3,
    )

    print("Turn 1 — user states a preference")
    run_turn(memory, "I prefer concise, direct answers.")

    print("Turn 2 — user updates location (volatility engine may audit)")
    run_turn(memory, "Actually I moved to Paris last month.")

    print("Turn 3 — question should recall Paris + concise preference")
    run_turn(memory, "Where do I live and how should you format replies?")

    memory.close()


if __name__ == "__main__":
    main()
