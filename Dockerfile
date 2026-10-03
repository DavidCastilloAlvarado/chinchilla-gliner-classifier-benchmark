# syntax=docker/dockerfile:1.7
#
# The application source is intentionally not copied into the image. Compose
# mounts the checkout at /workspace so local code and temp/ model weights are
# used directly at runtime.

FROM ghcr.io/astral-sh/uv:0.12.1 AS uv

FROM python:3.11-slim AS cpu

COPY --from=uv /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /opt/dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

WORKDIR /workspace
EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/workspace/app"]

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS cuda

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=python3.12 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /opt/dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

WORKDIR /workspace
EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/workspace/app"]
