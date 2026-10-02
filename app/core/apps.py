"""Load immutable, named application schemas from app/apps/*.json."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

APP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class AppRegistry:
    def __init__(self, apps_dir: Path) -> None:
        self.apps_dir = apps_dir
        self.apps: dict[str, dict[str, Any]] = {}

    def load(self) -> None:
        self.apps_dir.mkdir(parents=True, exist_ok=True)
        self.apps = {}
        for path in sorted(self.apps_dir.glob("*.json")):
            definition = json.loads(path.read_text(encoding="utf-8"))
            app_id = definition.get("app_id") or path.stem
            if not APP_ID_RE.fullmatch(app_id):
                raise ValueError(f"Invalid app_id {app_id!r} in {path}")
            self._validate(app_id, definition, path)
            self.apps[app_id] = definition

    def get(self, app_id: str) -> dict[str, Any]:
        try:
            return self.apps[app_id]
        except KeyError as exc:
            raise KeyError(f"Unknown app_id {app_id!r}; available apps: {sorted(self.apps)}") from exc

    def operation(self, app_id: str, usage_type: str) -> dict[str, Any]:
        definition = self.get(app_id)
        try:
            config = definition[usage_type]
        except KeyError as exc:
            raise KeyError(f"App {app_id!r} does not define usage type {usage_type!r}") from exc

        if usage_type == "classification":
            return {
                "kind": "classification",
                "labels": config["labels"],
                "multi_label": bool(config.get("multi_label", False)),
                "threshold": float(config.get("threshold", config.get("cls_threshold", 0.5))),
            }
        if usage_type == "extraction":
            return {
                "kind": "extraction",
                "entities": config["entities"],
                "include_confidence": bool(config.get("include_confidence", True)),
                "include_spans": bool(config.get("include_spans", True)),
            }
        raise ValueError(f"Unsupported usage type {usage_type!r}")

    def first_operation(self) -> dict[str, Any] | None:
        for app_id in sorted(self.apps):
            if "classification" in self.apps[app_id]:
                return self.operation(app_id, "classification")
        return None

    @staticmethod
    def _validate(app_id: str, definition: dict[str, Any], path: Path) -> None:
        if "classification" not in definition and "extraction" not in definition:
            raise ValueError(f"{path}: app {app_id!r} must define classification or extraction")

        classification = definition.get("classification")
        if classification is not None:
            labels = classification.get("labels")
            if not isinstance(labels, (list, dict)) or not labels:
                raise ValueError(f"{path}: classification.labels must be a non-empty list or object")
            if isinstance(labels, list) and not all(isinstance(label, str) and label for label in labels):
                raise ValueError(f"{path}: classification.labels must contain non-empty strings")
            if isinstance(labels, dict) and not all(
                isinstance(label, str) and label and isinstance(description, str)
                for label, description in labels.items()
            ):
                raise ValueError(f"{path}: classification label descriptions must be string-to-string")

        extraction = definition.get("extraction")
        if extraction is not None:
            entities = extraction.get("entities")
            if not isinstance(entities, (list, dict)) or not entities:
                raise ValueError(f"{path}: extraction.entities must be a non-empty list or object")
