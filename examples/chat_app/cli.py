"""
VoltMem chat — memory-aware CLI REPL.
=====================================

Run from repo root (editable install recommended):

    pip install -e ".[embeddings]"
    python -m examples.chat_app

With Ollama (optional, for real replies):

    ollama pull llama3.1
    python -m examples.chat_app

Slash commands: /help /memories /search /clear /reset /quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voltmem import create_memory

from .display import format_memories, format_writes
from .llm import create_llm
from .session import ChatSession

DEMO_TURNS = [
    "I prefer concise, direct answers.",
    "Actually I moved to Paris last month.",
    "Where do I live and how should you format replies?",
]

COMMANDS = {
    "/help": "Show commands",
    "/memories": "List all stored memories",
    "/mem": "Alias for /memories",
    "/search": "Search memories — /search <query>",
    "/clear": "Delete all memories for this user",
    "/reset": "Clear in-session chat history (memories kept)",
    "/verbose": "Toggle showing recalled memories each turn",
    "/quit": "Exit",
    "/exit": "Exit",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VoltMem memory-aware chat CLI")
    p.add_argument("--db", default="chat_app.db", help="SQLite path")
    p.add_argument("--user-id", default="default", help="Memory namespace")
    p.add_argument("--model", default=None, help="Ollama chat model")
    p.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL",
    )
    p.add_argument("--recall-limit", type=int, default=5)
    p.add_argument("--echo", action="store_true", help="Force echo LLM (no Ollama)")
    p.add_argument(
        "--llm-extract",
        action="store_true",
        help="Use Ollama for fact extraction on each turn",
    )
    p.add_argument("--verbose-embeddings", action="store_true")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run scripted demo turns and exit (smoke test)",
    )
    p.add_argument(
        "--show-recall",
        action="store_true",
        help="Print recalled memories on every turn",
    )
    return p


def print_banner(backend: str, db: str, user_id: str) -> None:
    print("VoltMem chat")
    print(f"  db={db}  user={user_id}  llm={backend}")
    print("  Type /help for commands. Empty line ignored.\n")


def print_help() -> None:
    print("Commands:")
    for cmd, desc in COMMANDS.items():
        print(f"  {cmd:<12} {desc}")


def handle_command(
    line: str, session: ChatSession, *, show_recall: bool
) -> tuple[bool, bool]:
    """Returns (handled, show_recall). handled=False means treat as chat."""
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit"):
        return True, show_recall
    if cmd == "/help":
        print_help()
        return True, show_recall
    if cmd in ("/memories", "/mem"):
        print(format_memories(session.list_memories()))
        return True, show_recall
    if cmd == "/search":
        if not arg:
            print("  usage: /search <query>")
            return True, show_recall
        print(format_memories(session.search_memories(arg), show_score=True))
        return True, show_recall
    if cmd == "/clear":
        session.clear_memories()
        print("  memories cleared")
        return True, show_recall
    if cmd == "/reset":
        session.reset_history()
        print("  chat history reset")
        return True, show_recall
    if cmd == "/verbose":
        show_recall = not show_recall
        state = "on" if show_recall else "off"
        print(f"  recall display: {state}")
        return True, show_recall

    return False, show_recall


def run_turn(session: ChatSession, message: str, *, show_recall: bool) -> None:
    result = session.chat(message)
    if show_recall:
        print("Recalled:")
        print(format_memories(result.recalled_memories, show_score=True))
    print(f"You: {result.user_message}")
    print(f"Assistant: {result.assistant_message}")
    extra = format_writes(result.write_results)
    if extra:
        print(extra)
    print()


def run_demo(session: ChatSession, *, show_recall: bool) -> None:
    for turn in DEMO_TURNS:
        run_turn(session, turn, show_recall=show_recall)


def run_repl(session: ChatSession, *, show_recall: bool) -> None:
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        handled, show_recall = handle_command(line, session, show_recall=show_recall)
        if line.lower() in ("/quit", "/exit"):
            break
        if handled:
            continue

        try:
            run_turn(session, line, show_recall=show_recall)
        except ValueError as exc:
            print(f"  {exc}")
        except Exception as exc:
            print(f"  error: {exc}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    llm, backend = create_llm(
        echo=args.echo,
        model=args.model,
        ollama_url=args.ollama_url,
    )

    mem = create_memory(
        args.db,
        user_id=args.user_id,
        verbose=args.verbose_embeddings,
        llm_extract=args.llm_extract,
    )

    try:
        session = ChatSession(
            mem,
            llm,
            recall_limit=args.recall_limit,
        )
        print_banner(backend, args.db, args.user_id)

        if args.demo:
            run_demo(session, show_recall=args.show_recall)
        else:
            run_repl(session, show_recall=args.show_recall)
    finally:
        mem.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
