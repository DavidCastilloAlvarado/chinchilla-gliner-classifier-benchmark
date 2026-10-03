"""Environment-backed settings for the inference server."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Validated settings loaded from process environment and project ``.env``."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    model_name: str = Field("decide", validation_alias="MODEL_NAME")
    gpu_mode: str = Field("cpu", validation_alias="GPU_MODE")
    compile_model: bool = Field(False, validation_alias="COMPILE_MODEL")
    batch_window_ms: float = Field(3.0, validation_alias="BATCH_WINDOW_MS")
    max_batch_size: int = Field(
        8,
        validation_alias=AliasChoices("N_CONCURRENCY", "MAX_BATCH_SIZE"),
    )
    max_queue_size: int = Field(256, validation_alias="MAX_QUEUE_SIZE")
    request_timeout_seconds: float = Field(30.0, validation_alias="REQUEST_TIMEOUT_SECONDS")
    apps_dir: Path = Field(PROJECT_ROOT / "app" / "apps", validation_alias="APPS_DIR")
    model_dir: Path | None = Field(default=None, validation_alias="MODEL_DIR")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in MODEL_REPOS:
            raise ValueError(f"MODEL_NAME must be one of {sorted(MODEL_REPOS)}, got {value!r}")
        return value

    @field_validator("gpu_mode")
    @classmethod
    def validate_gpu_mode(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"cpu", "cuda"}:
            raise ValueError("GPU_MODE must be 'cpu' or 'cuda'")
        return value

    @field_validator("batch_window_ms")
    @classmethod
    def validate_batch_window(cls, value: float) -> float:
        return max(0.0, value)

    @field_validator("max_batch_size", "max_queue_size")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        return max(1, value)

    @field_validator("request_timeout_seconds")
    @classmethod
    def validate_request_timeout(cls, value: float) -> float:
        return max(0.1, value)

    @classmethod
    def from_env(cls) -> "Settings":
        """Compatibility constructor; ``BaseSettings`` performs the loading."""
        return cls()

    @property
    def model_repo(self) -> str:
        return MODEL_REPOS[self.model_name]

    @property
    def resolved_model_dir(self) -> Path:
        if self.model_dir is not None:
            return self.model_dir.expanduser()
        return PROJECT_ROOT / "temp" / MODEL_DIR_NAMES[self.model_name]
