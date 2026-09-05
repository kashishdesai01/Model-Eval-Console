FROM ghcr.io/astral-sh/uv:0.8.22 AS uv
FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 mec
WORKDIR /app/backend
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PATH="/app/backend/.venv/bin:$PATH" \
    HF_HOME=/home/mec/.cache/huggingface PYTHONUNBUFFERED=1
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY --chown=mec:mec backend/app/ ./app/
COPY --chown=mec:mec backend/migrations/ ./migrations/
COPY --chown=mec:mec backend/alembic.ini ./
COPY --chown=mec:mec scripts/ /app/scripts/
RUN uv sync --frozen --no-dev && mkdir -p /home/mec/.cache/huggingface \
    && chown -R mec:mec /home/mec
USER mec
EXPOSE 8000 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
