"""
VoltMem demo — plugged into a simple Anthropic-backed assistant.

Shows how the memory layer sits between the application and the LLM:
  1. Before each LLM call, retrieve() injects relevant memories into context.
  2. After each LLM response, the app calls observe() with new facts it extracts.

Run: python demo.py  (uses a mock LLM so no API key needed)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voltmem import MemoryLayer

# ── mock LLM (swap this with real Anthropic call) ─────────────────────────────

def call_llm(system: str, user: str) -> str:
    """Stub — replace with actual anthropic.Anthropic().messages.create(...)"""
    return f"[LLM would respond here given system='{system[:60]}...' user='{user}']"


# ── simple fact extractor (stub — replace with LLM-based extraction) ──────────

def extract_facts(text: str) -> list[dict]:
    """
    In production: call the LLM to extract structured facts from user text.
    Returns list of {content, domain, mismatch_magnitude, source}.
    Here we hand-code a few for demonstration.
    """
    facts = []
    t = text.lower()

    if "accepted" in t and "job" in t:
        facts.append({
            "content": "User accepted a job offer",
            "domain": "current_project",
            "mismatch_magnitude": 0.85,
            "source": "explicit_statement",
        })
    if "moving" in t or "berlin" in t:
        facts.append({
            "content": "User is moving / lives in Berlin",
            "domain": "location",
            "mismatch_magnitude": 0.1,
            "source": "explicit_statement",
        })
    if "direct" in t and ("prefer" in t or "like" in t):
        facts.append({
            "content": "User prefers direct communication",
            "domain": "core_preference",
            "mismatch_magnitude": 0.0,
            "source": "explicit_statement",
        })
    return facts


# ── the pluggable memory-augmented assistant ───────────────────────────────────

class MemoryAugmentedAssistant:
    def __init__(self, db_path: str = ":memory:"):
        self.mem = MemoryLayer(db_path, goal_delta_default=0.1)
        self._seed_memories()

    def _seed_memories(self):
        """Bootstrap with any known long-term facts."""
        self.mem.write(
            "User is a self-taught software engineer based in Berlin",
            domain="biographical",
            source="explicit_statement",
        )
        self.mem.write(
            "User prefers direct, concise responses",
            domain="core_preference",
            source="repeated_confirmation",
        )
        self.mem.write(
            "User is currently job-hunting in AI product engineering",
            domain="current_project",
            source="explicit_statement",
        )
        self.mem.write(
            "User has strong interests in philosophy and cognitive science",
            domain="personality_trait",
            source="strong_inference",
        )

    def chat(self, user_message: str) -> str:
        # ── 1. Retrieve relevant memories ─────────────────────────────────────
        results = self.mem.retrieve(user_message, top_k=4)
        memory_context = ""
        if results.items:
            memory_context = "\n".join(
                f"- [{item.domain}] {item.content}  (freshness: {1-s:.2f})"
                for item, s in zip(results.items, results.scores)
            )

        system_prompt = (
            "You are a helpful assistant with persistent memory about the user.\n\n"
            f"What you remember about this user:\n{memory_context}\n\n"
            "Use this context naturally. Do not mention the memory system."
        )

        # ── 2. Call LLM ───────────────────────────────────────────────────────
        response = call_llm(system_prompt, user_message)

        # ── 3. Extract and observe new facts ──────────────────────────────────
        facts = extract_facts(user_message)
        update_log = []
        for fact in facts:
            result = self.mem.observe(**fact)
            update_log.append(f"  [{result.action}] {fact['content'][:60]}")

        return response, update_log

    def show_memory_state(self):
        print("\n── Current Memory State ─────────────────────────────────────")
        print(f"  {self.mem.summary()}")
        for item in self.mem._store.all_active():
            info = self.mem.inspect(item.id)
            print(f"  [{item.domain}] {item.content[:55]}")
            print(f"    staleness={info['staleness']:.3f}  "
                  f"protection={info['protection_weight']:.2f}  "
                  f"reps={item.repetition_count}")
        print()

    def close(self):
        self.mem.close()


# ── run the demo ──────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("VoltMem demo — volatility-adjusted memory layer")
    print("=" * 65)

    assistant = MemoryAugmentedAssistant()

    conversations = [
        "I like direct answers, don't hedge too much",
        "I just accepted a job offer at an AI startup — very excited",
        "I'm still in Berlin by the way, not moving",
    ]

    for msg in conversations:
        print(f"\nUser: {msg}")
        response, updates = assistant.chat(msg)
        print(f"Assistant: {response}")
        if updates:
            print("Memory updates:")
            for u in updates:
                print(u)

    assistant.show_memory_state()
    assistant.close()


if __name__ == "__main__":
    main()
