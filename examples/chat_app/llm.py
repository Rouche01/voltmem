"""LLM backends — stdlib-only; Ollama when available, echo fallback."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol


class LLM(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the assistant reply for a chat-style message list."""


class OllamaLLM:
    def __init__(
        self,
        *,
        model: str | None = None,
        ollama_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1")
        self.base_url = ollama_url.rstrip("/")
        self.timeout = timeout

    def complete(self, messages: list[dict[str, str]]) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.4},
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode())
        message = body.get("message") or {}
        content = str(message.get("content", "")).strip()
        if not content:
            raise RuntimeError("Ollama returned an empty response")
        return content


class EchoLLM:
    """Offline stand-in — proves the memory loop without a chat model."""

    def complete(self, messages: list[dict[str, str]]) -> str:
        memories: list[str] = []
        for msg in messages:
            if msg.get("role") != "system":
                continue
            for line in str(msg.get("content", "")).splitlines():
                stripped = line.strip()
                if stripped.startswith("- "):
                    memories.append(stripped[2:])
        if memories:
            preview = "; ".join(memories[:3])
            return (
                f"Noted. (echo mode) From memory: {preview}. "
                "Start Ollama for real replies."
            )
        return (
            "Noted. (echo mode — no Ollama chat model; "
            "memories are still stored and recalled.)"
        )


def ollama_available(ollama_url: str = "http://localhost:11434") -> bool:
    try:
        urllib.request.urlopen(
            f"{ollama_url.rstrip('/')}/api/tags",
            timeout=2,
        )
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def create_llm(
    *,
    echo: bool = False,
    model: str | None = None,
    ollama_url: str = "http://localhost:11434",
) -> tuple[LLM, str]:
    """Pick Ollama when reachable, otherwise echo mode."""
    if echo or not ollama_available(ollama_url):
        return EchoLLM(), "echo"
    return OllamaLLM(model=model, ollama_url=ollama_url), "ollama"
