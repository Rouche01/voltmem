# VoltMem sidecar image
FROM python:3.12-slim-bookworm

WORKDIR /app

COPY pyproject.toml README.md ./
COPY voltmem ./voltmem
COPY sidecar ./sidecar

# 1 installs sentence-transformers (PyTorch). 0 is the hashing scorer only.
ARG EMBEDDINGS=1

RUN set -eu; \
    if [ "$EMBEDDINGS" != "0" ] && [ "$EMBEDDINGS" != "1" ]; then \
      echo "EMBEDDINGS must be 0 or 1" >&2; \
      exit 1; \
    fi; \
    if [ "$EMBEDDINGS" = "1" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends build-essential \
      && rm -rf /var/lib/apt/lists/* \
      && pip install --no-cache-dir -e ".[sidecar,embeddings]" \
      && apt-get purge -y --auto-remove build-essential \
      && rm -rf /var/lib/apt/lists/*; \
    else \
      pip install --no-cache-dir -e ".[sidecar]"; \
    fi

ENV VOLTMEM_DB_PATH=/data/voltmem.db \
    VOLTMEM_EMBEDDINGS=${EMBEDDINGS} \
    VOLTMEM_PROFILE=stylens \
    HOST=0.0.0.0 \
    PORT=8080 \
    PYTHONUNBUFFERED=1

VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')"

CMD ["uvicorn", "sidecar.app:app", "--host", "0.0.0.0", "--port", "8080"]
