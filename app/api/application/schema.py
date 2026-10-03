"""Legacy and stored application request/response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):
    query: str = Field(min_length=1, description="Text to classify")
    classes: list[str] = Field(min_length=1, description="Candidate labels")
    descriptions: dict[str, str] | None = Field(default=None, description="Optional label-to-description mapping")
    multi_label: bool = False
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class AppUsageRequest(BaseModel):
    query: str = Field(min_length=1, description="Text to process with the stored app schema")


class InferenceMeta(BaseModel):
    batch_size: int
    queue_wait_ms: float
    inference_ms: float
    device: str


class InferenceResponse(BaseModel):
    result: Any
    meta: InferenceMeta


UsageType = Literal["classification", "extraction"]
