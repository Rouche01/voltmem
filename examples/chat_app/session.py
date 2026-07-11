"""Chat session — memory recall, prompt build, store. Reusable by CLI or web."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from voltmem import Memory

from .llm import LLM

DEFAULT_SYSTEM = (
    "You are a helpful personal assistant. "
    "Use the user memories below when they are relevant. Be concise."
)


@dataclass
class TurnResult:
    user_message: str
    assistant_message: str
    recalled_memories: list[dict[str, Any]]
    write_results: list[dict[str, Any]] = field(default_factory=list)


class ChatSession:
    """One user chat loop backed by VoltMem."""

    def __init__(
        self,
        memory: Memory,
        llm: LLM,
        *,
        recall_limit: int = 5,
        system_prompt: str = DEFAULT_SYSTEM,
    ) -> None:
        self.memory = memory
        self.llm = llm
        self.recall_limit = recall_limit
        self.system_prompt = system_prompt
        self._history: list[dict[str, str]] = []

    def recall(self, query: str) -> list[dict[str, Any]]:
        return self.memory.search(query, limit=self.recall_limit)

    def list_memories(self) -> list[dict[str, Any]]:
        return self.memory.get_all()

    def search_memories(
        self, query: str, *, limit: int = 5
    ) -> list[dict[str, Any]]:
        return self.memory.search(query, limit=limit)

    def clear_memories(self) -> None:
        self.memory.clear()

    def discovery_report(self) -> dict[str, Any]:
        """Domain auto-discovery stats for ``/discovery``."""
        return self.memory.summary()

    def reset_history(self) -> None:
        self._history.clear()

    def build_messages(
        self, user_message: str
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        recalled = self.recall(user_message)
        system = self._system_with_memories(recalled)
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_message})
        return messages, recalled

    def chat(self, user_message: str) -> TurnResult:
        text = user_message.strip()
        if not text:
            raise ValueError("message must not be empty")

        messages, recalled = self.build_messages(text)
        reply = self.llm.complete(messages)

        self._history.append({"role": "user", "content": text})
        self._history.append({"role": "assistant", "content": reply})

        writes = self.memory.add([
            {"role": "user", "content": text},
            {"role": "assistant", "content": reply},
        ])
        if isinstance(writes, dict):
            writes = [writes]

        return TurnResult(
            user_message=text,
            assistant_message=reply,
            recalled_memories=recalled,
            write_results=writes,
        )

    def _system_with_memories(self, memories: list[dict[str, Any]]) -> str:
        if not memories:
            return self.system_prompt
        lines = "\n".join(f"- {m['memory']}" for m in memories)
        return f"{self.system_prompt}\n\nWhat you know about this user:\n{lines}"
