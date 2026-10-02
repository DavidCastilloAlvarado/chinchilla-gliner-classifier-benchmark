"""Environment-backed settings for the inference server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_REPOS = {
    "base": "fastino/gliner2-base-v1",
    "multi": "fastino/gliner2.5-multi-v1",
    "decide": "fastino/GLiNER2.5-multi-Decide",
}
MODEL_DIR_NAMES = {
    "base": "gliner2-base-v1",
    "multi": "gliner2.5-multi-v1",
    "decide": "GLiNER2.5-multi-Decide",
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime configuration; all values can be overridden with environment variables."""

    model_name: str = "decide"
    gpu_mode: str = "cpu"
    compile_model: bool = False
    batch_window_ms: float = 10.0
    max_batch_size: int = 8
    max_queue_size: int = 256
    request_timeout_seconds: float = 30.0
    apps_dir: Path = PROJECT_ROOT / "app" / "apps"
    model_dir: Path | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        model_name = os.getenv("MODEL_NAME", "decide").strip().lower()
        if model_name not in MODEL_REPOS:
            raise ValueError(f"MODEL_NAME must be one of {sorted(MODEL_REPOS)}, got {model_name!r}")

        gpu_mode = os.getenv("GPU_MODE", "cpu").strip().lower()
        if gpu_mode not in {"cpu", "cuda"}:
            raise ValueError("GPU_MODE must be 'cpu' or 'cuda'")

        model_dir_value = os.getenv("MODEL_DIR")
        model_dir = Path(model_dir_value).expanduser() if model_dir_value else None
        apps_dir = Path(os.getenv("APPS_DIR", str(PROJECT_ROOT / "app" / "apps"))).expanduser()

        return cls(
            model_name=model_name,
            gpu_mode=gpu_mode,
            compile_model=_env_bool("COMPILE_MODEL", False),
            batch_window_ms=max(0.0, float(os.getenv("BATCH_WINDOW_MS", "10"))),
            max_batch_size=max(1, int(os.getenv("N_CONCURRENCY", "8"))),
            max_queue_size=max(1, int(os.getenv("MAX_QUEUE_SIZE", "256"))),
            request_timeout_seconds=max(0.1, float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))),
            apps_dir=apps_dir,
            model_dir=model_dir,
        )

    @property
    def model_repo(self) -> str:
        return MODEL_REPOS[self.model_name]

    @property
    def resolved_model_dir(self) -> Path:
        if self.model_dir is not None:
            return self.model_dir
        return PROJECT_ROOT / "temp" / MODEL_DIR_NAMES[self.model_name]
