"""Model loading and batched GLiNER2 operations."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from gliner2 import AutoExtractor, GLiNER2

from .config import Settings


class ModelService:
    """Owns one loaded model and executes only batched, inference-mode calls."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any | None = None
        self.load_seconds: float | None = None

    @property
    def device(self) -> str:
        return "cuda" if self.settings.gpu_mode == "cuda" else "cpu"

    def load(self) -> None:
        model_dir = Path(self.settings.resolved_model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model not found at {model_dir}. "
                f"Download it with: uv run python src/classifier/download_model.py {self.settings.model_name}"
            )
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU_MODE=cuda was requested, but torch.cuda.is_available() is false")

        loader = GLiNER2 if self.settings.model_name == "base" else AutoExtractor
        started = time.perf_counter()
        self.model = loader.from_pretrained(str(model_dir), map_location=self.device)
        self.model.eval()
        if self.settings.compile_model:
            self.model.compile()
        self.load_seconds = time.perf_counter() - started

    def warmup(self, operation: dict[str, Any]) -> float:
        """Trigger compilation and CUDA lazy initialization before readiness."""
        started = time.perf_counter()
        self.infer_batch(["warmup request"], operation)
        if self.device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - started

    def infer_batch(self, texts: list[str], operation: dict[str, Any]) -> list[Any]:
        if self.model is None:
            raise RuntimeError("Model has not been loaded")

        kind = operation["kind"]
        with torch.inference_mode():
            if kind == "classification":
                labels = operation["labels"]
                tasks = {
                    "intent": {
                        "labels": labels,
                        "multi_label": operation.get("multi_label", False),
                        "cls_threshold": operation.get("threshold", 0.5),
                    }
                }
                return self.model.batch_classify_text(
                    texts,
                    tasks,
                    batch_size=len(texts),
                    include_confidence=True,
                )

            if kind == "extraction":
                return self.model.batch_extract_entities(
                    texts,
                    operation["entities"],
                    batch_size=len(texts),
                    include_confidence=operation.get("include_confidence", True),
                    include_spans=operation.get("include_spans", True),
                )

        raise ValueError(f"Unsupported operation kind: {kind!r}")
