"""Business logic for legacy and stored-application endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException

from app.api.common import submit_inference
from app.core.dependencies import get_runtime
from app.core.lifespan import Runtime

from .schema import AppUsageRequest, ClassificationRequest, InferenceResponse, UsageType


class ApplicationService:
    """Coordinates legacy classification and stored app-schema operations."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    @staticmethod
    def labels_with_descriptions(
        classes: list[str],
        descriptions: dict[str, str] | None,
    ) -> list[str] | dict[str, str]:
        if not descriptions:
            return classes
        unknown = set(descriptions).difference(classes)
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Descriptions contain labels not present in classes: {sorted(unknown)}",
            )
        return {label: descriptions.get(label, label) for label in classes}

    async def classify(self, payload: ClassificationRequest) -> InferenceResponse:
        operation = {
            "kind": "classification",
            "labels": self.labels_with_descriptions(payload.classes, payload.descriptions),
            "multi_label": payload.multi_label,
            "threshold": payload.threshold,
        }
        result = await submit_inference(self.runtime, payload.query, operation)
        return InferenceResponse.model_validate(result)

    async def use_app(self, app_id: str, usage_type: UsageType, payload: AppUsageRequest) -> InferenceResponse:
        try:
            operation = self.runtime.registry.operation(app_id, usage_type)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        result = await submit_inference(self.runtime, payload.query, operation)
        return InferenceResponse.model_validate(result)


def get_application_service(runtime: Runtime = Depends(get_runtime)) -> ApplicationService:
    """Dependency factory for application controllers."""
    return ApplicationService(runtime)
