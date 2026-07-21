"""Run: ``python -m sidecar`` or ``uvicorn sidecar.app:app --host 0.0.0.0 --port 8080``."""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("sidecar.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
