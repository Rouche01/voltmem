"""
LangChain memory adapter for VoltMem.

Install optional dependencies first:
    pip install -r requirements-integrations.txt

Usage:
    from voltmem.integrations.langchain import VoltMemMemory

    memory = VoltMemMemory(session_id="user-42", db_path="app.db")
    vars = memory.load_memory_variables({"input": "Where do I live?"})
    memory.save_context({"input": "I moved to Paris"}, {"output": "Noted."})
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Optional

from voltmem import MemoryLayer


def _import_base_memory():
    for module_name, attr in (
        ("langchain_classic.base_memory", "BaseMemory"),
        ("langchain.memory", "BaseMemory"),
    ):
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr)
        except ImportError:
            continue
    raise ImportError(
        "LangChain is required for voltmem.integrations.langchain. "
        "Install with: pip install -r requirements-integrations.txt"
    )


try:
    BaseMemory = _import_base_memory()
except ImportError as exc:
    raise ImportError(
        "LangChain is required for voltmem.integrations.langchain. "
        "Install with: pip install -r requirements-integrations.txt"
    ) from exc


def _extract_query(inputs: dict[str, Any], input_key: str) -> str:
    if input_key in inputs and inputs[input_key] is not None:
        return str(inputs[input_key])
    for key in ("input", "question", "human_input", "query"):
        if key in inputs and inputs[key] is not None:
            return str(inputs[key])
    for value in inputs.values():
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _format_memories(items: list[str], prefix: str) -> str:
    if not items:
        return ""
    lines = [f"- {item}" for item in items]
    body = "\n".join(lines)
    return f"{prefix}{body}" if prefix else body


class VoltMemMemory(BaseMemory):
    """
    LangChain memory that stores facts with VoltMem and injects fresh recall.

    Maps LangChain's per-turn hooks to VoltMem's batteries-included API:
    - ``load_memory_variables`` → ``recall(query)``
    - ``save_context`` → ``remember(user_input)`` (optional assistant text)

    ``session_id`` scopes memories to one user/conversation via VoltMem's
    namespace isolation (``for_user`` under the hood).
    """

    memory_key: str = "history"
    input_key: str = "input"
    output_key: str = "output"
    session_id: str = "default"
    top_k: int = 5
    min_score: float = 0.0
    remember_assistant: bool = False
    memory_prefix: str = "Relevant memories:\n"
    assistant_prefix: str = "Assistant said: "

    def __init__(
        self,
        *,
        db_path: str = ":memory:",
        session_id: str = "default",
        layer: Optional[MemoryLayer] = None,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output",
        top_k: int = 5,
        min_score: float = 0.0,
        remember_assistant: bool = False,
        memory_prefix: str = "Relevant memories:\n",
        assistant_prefix: str = "Assistant said: ",
        **kwargs: Any,
    ):
        super().__init__(
            memory_key=memory_key,
            input_key=input_key,
            output_key=output_key,
            session_id=session_id,
            top_k=top_k,
            min_score=min_score,
            remember_assistant=remember_assistant,
            memory_prefix=memory_prefix,
            assistant_prefix=assistant_prefix,
            **kwargs,
        )
        if layer is not None:
            self._mem = (
                layer
                if layer.namespace == session_id
                else layer.for_user(session_id)
            )
            self._owns_layer = False
        else:
            base = MemoryLayer(
                db_path,
                similarity_fn=similarity_fn,
                namespace=session_id,
            )
            self._mem = base
            self._owns_layer = True

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    @property
    def layer(self) -> MemoryLayer:
        """Underlying VoltMem layer (same namespace as ``session_id``)."""
        return self._mem

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        query = _extract_query(inputs, self.input_key)
        recalled = self._mem.recall(
            query,
            top_k=self.top_k,
            min_score=self.min_score,
        )
        return {self.memory_key: _format_memories(recalled, self.memory_prefix)}

    def save_context(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, str],
    ) -> None:
        user_text = inputs.get(self.input_key)
        if user_text is None:
            user_text = _extract_query(inputs, self.input_key)
        if user_text:
            self._mem.remember(str(user_text))
        if self.remember_assistant:
            assistant_text = outputs.get(self.output_key, "")
            if assistant_text:
                self._mem.remember(
                    f"{self.assistant_prefix}{assistant_text}",
                    source="assistant_response",
                )

    def clear(self) -> None:
        self._mem.clear()

    def close(self) -> None:
        if self._owns_layer:
            self._mem.close()
