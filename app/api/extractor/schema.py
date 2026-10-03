"""Entity extraction request and response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExtractionRequest(BaseModel):
    model: str = Field(min_length=1, description="Requested local GLiNER2 model identifier")
    state: str | dict[str, Any] | list[Any] = Field(description="Text or structured state to inspect")
    entities: dict[str, str | None] = Field(
        min_length=1,
        description="Entity categories mapped to optional descriptions; use null when no description is needed",
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum entity confidence threshold")
    include_confidence: bool = True
    include_spans: bool = True


class ExtractionResponse(BaseModel):
    model: str
    entities: dict[str, list[Any]]
    usage: dict[str, Any] | None = None
