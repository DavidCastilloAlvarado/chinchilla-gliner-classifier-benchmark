"""Model loading and batched GLiNER2 operations."""

from __future__ import annotations

import json
import time
import types
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
                    threshold=operation.get("threshold", 0.5),
                    include_confidence=operation.get("include_confidence", True),
                    include_spans=operation.get("include_spans", True),
                )

            if kind == "systemone":
                return self._infer_system_one_batch(texts, operation)

        raise ValueError(f"Unsupported operation kind: {kind!r}")

    @staticmethod
    def _system_one_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _system_one_labels(cls, question: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        """Build GLiNER labels and output metadata for one System One question."""
        question_type = question["type"]
        if question_type == "choice":
            criteria = question["criteria"]
            labels = {
                str(label): cls._system_one_text(description) if description is not None else str(label)
                for label, description in criteria.items()
            }
            return labels, {"type": "choice", "labels": list(labels)}

        if question_type == "noul":
            criteria = question.get("criteria") or {}
            yes = criteria.get("yes", criteria.get("true", "Yes"))
            no = criteria.get("no", criteria.get("false", "No"))
            labels = {
                "true": cls._system_one_text(yes),
                "false": cls._system_one_text(no),
            }
            return labels, {"type": "noul", "labels": list(labels)}

        levels = question["criteria"]
        labels = {
            f"level_{index}": cls._system_one_text(level)
            for index, level in enumerate(levels)
        }
        return labels, {
            "type": "score",
            "labels": list(labels),
            "legend": {str(index): cls._system_one_text(level) for index, level in enumerate(levels)},
        }

    def _infer_system_one_batch(
        self,
        texts: list[str],
        operation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Run all independent System One questions in one GLiNER forward pass."""
        if self.model is None:
            raise RuntimeError("Model has not been loaded")

        schema = self.model.create_schema()
        question_specs: list[tuple[str, str, dict[str, Any]]] = []
        for question_id, question in operation["questions"].items():
            labels, metadata = self._system_one_labels(question)
            instruction = self._system_one_text(question["instructions"])
            task = f"systemone question {question_id}: {instruction}"
            schema.classification(task, labels, multi_label=False, cls_threshold=0.0)
            question_specs.append((question_id, task, metadata))

        probabilities: list[dict[str, dict[str, float]]] = [dict() for _ in texts]
        seen: dict[str, int] = {}
        # GLiNER2's public formatter returns only the winning label/confidence.
        # Capture its classification logits at the same decode point so the
        # System One contract can expose the complete probability distribution.
        original = self.model._extract_classification_result

        def capture_classification(
            model: Any,
            results: dict[str, Any],
            schema_name: str,
            schema_dict: dict[str, Any],
            embs: torch.Tensor,
            schema_tokens: list[str],
            temperature: float = 1.0,
        ) -> None:
            prompt = schema_tokens[2]
            config = model._resolve_classification_config(
                prompt,
                schema_dict.get("classifications", []),
            )
            if config is not None:
                task = config["task"]
                sample_index = seen.get(task, 0)
                seen[task] = sample_index + 1
                logits = model.classifier(embs[1:]).squeeze(-1) / temperature
                activation = config.get("class_act", "auto")
                if activation == "sigmoid":
                    probs = torch.sigmoid(logits)
                elif activation == "softmax":
                    probs = torch.softmax(logits, dim=-1)
                else:
                    probs = torch.softmax(logits, dim=-1)
                probabilities[sample_index][task] = {
                    label: float(prob.item())
                    for label, prob in zip(config["labels"], probs)
                }
            original(results, schema_name, schema_dict, embs, schema_tokens, temperature)

        self.model._extract_classification_result = types.MethodType(
            capture_classification,
            self.model,
        )
        try:
            raw_results = self.model.batch_extract(
                texts,
                schema,
                batch_size=len(texts),
                include_confidence=True,
            )
        finally:
            self.model._extract_classification_result = original

        answers: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_results):
            per_question: dict[str, Any] = {}
            for question_id, task, metadata in question_specs:
                dist = probabilities[index].get(task, {})
                if not dist:
                    raw_value = raw.get(task)
                    selected, confidence = self._selected_classification(raw_value)
                    labels = metadata["labels"]
                    dist = self._fallback_distribution(labels, selected, confidence)

                if metadata["type"] == "choice":
                    selected = max(dist, key=dist.get)
                    per_question[question_id] = {
                        "type": "choice",
                        "choice": selected,
                        "probabilities": dist,
                        "confidence": max(dist.values()),
                    }
                elif metadata["type"] == "noul":
                    per_question[question_id] = {
                        "type": "noul",
                        "noul": dist.get("true", 0.0),
                    }
                else:
                    score_dist = {
                        str(label.removeprefix("level_")): probability
                        for label, probability in dist.items()
                    }
                    score = sum(float(level) * probability for level, probability in score_dist.items())
                    per_question[question_id] = {
                        "type": "score",
                        "score": score,
                        "legend": metadata["legend"],
                        "probabilities": score_dist,
                        "confidence": max(dist.values()),
                    }
            answers.append(per_question)
        return [
            {
                "model": operation["model"],
                "answers": answer,
                "usage": {
                    "input_tokens": self._estimate_system_one_tokens(
                        texts[index],
                        operation["questions"],
                    ),
                    "output_tokens": 0,
                },
            }
            for index, answer in enumerate(answers)
        ]

    def _estimate_system_one_tokens(
        self,
        state_text: str,
        questions: dict[str, Any],
    ) -> int:
        """Estimate local input tokens for the System One-compatible usage field."""
        prompt = state_text + json.dumps(questions, ensure_ascii=False, separators=(",", ":"))
        tokenizer = getattr(getattr(self.model, "processor", None), "tokenizer", None)
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(prompt, add_special_tokens=True))
            except Exception:
                pass
        return max(1, len(prompt.split()))

    @staticmethod
    def _selected_classification(value: Any) -> tuple[str | None, float]:
        if isinstance(value, dict):
            return value.get("label"), float(value.get("confidence", 0.0))
        if isinstance(value, (tuple, list)) and value:
            return value[0], float(value[1]) if len(value) > 1 else 0.0
        return None, 0.0

    @staticmethod
    def _fallback_distribution(
        labels: list[str],
        selected: str | None,
        confidence: float,
    ) -> dict[str, float]:
        if not labels:
            return {}
        selected = selected if selected in labels else labels[0]
        confidence = min(1.0, max(0.0, confidence))
        remainder = (1.0 - confidence) / max(1, len(labels) - 1)
        return {
            label: confidence if label == selected else remainder
            for label in labels
        }
