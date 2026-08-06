"""Run: ``python -m sidecar`` or ``uvicorn sidecar.app:app --host 0.0.0.0 --port 8080``."""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load repo-root ``.env`` so ``python -m sidecar`` needs no CLI exports."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


def main() -> None:
    _load_dotenv()
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("sidecar.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
