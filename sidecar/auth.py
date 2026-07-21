"""API-key auth for the VoltMem sidecar.

When ``VOLTMEM_API_KEY`` is set, requests must send matching ``X-API-Key``.
When unset (local / test), auth is a no-op.
"""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import Header, HTTPException, status


def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    expected = os.environ.get("VOLTMEM_API_KEY", "").strip()
    if not expected:
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing X-API-Key",
        )
