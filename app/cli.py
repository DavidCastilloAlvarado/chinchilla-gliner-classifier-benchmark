"""Command-line entry points for the local FastAPI server."""

from __future__ import annotations

import os

import uvicorn


HOST = "0.0.0.0"
PORT = 8000


def _run_server(*, gpu: bool = False) -> None:
    if gpu:
        os.environ["GPU_MODE"] = "cuda"
        os.environ["COMPILE_MODEL"] = "true"

    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
    )


def server() -> None:
    """Start the default CPU server."""
    _run_server()


def server_gpu() -> None:
    """Start the CUDA server with model compilation enabled."""
    _run_server(gpu=True)
