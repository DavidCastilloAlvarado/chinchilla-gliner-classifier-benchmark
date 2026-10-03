"""HTTP request and response models."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):
    query: str = Field(min_length=1, description="Text to classify")
    classes: list[str] = Field(min_length=1, description="Candidate labels")
    descriptions: dict[str, str] | None = Field(
        default=None,
        description="Optional label-to-description mapping",
    )
    multi_label: bool = False
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class AppUsageRequest(BaseModel):
    query: str = Field(min_length=1, description="Text to process with the stored app schema")


class SystemOneChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Any = Field(description="The decision to make")
    criteria: dict[str, Any] = Field(min_length=1, description="Option names and optional descriptions")


class SystemOneNoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Any = Field(description="The yes/no proposition to evaluate")
    criteria: dict[str, Any] | None = Field(
        default=None,
        description="Optional descriptions for yes/true and no/false",
    )


class SystemOneScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Any = Field(description="The ordered rubric to apply")
    criteria: list[Any] = Field(min_length=2, max_length=10, description="Ordered score levels")


SystemOneQuestion = Annotated[
    Union[SystemOneChoiceQuestion, SystemOneNoulQuestion, SystemOneScoreQuestion],
    Field(discriminator="type"),
]


class SystemOneRequest(BaseModel):
    model: str = Field(min_length=1, description="Requested System One model identifier")
    state: str | dict[str, Any] | list[Any] = Field(description="State to evaluate")
    questions: dict[str, SystemOneQuestion] = Field(min_length=1, description="Named independent questions")


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, dict[str, Any]]
    usage: dict[str, Any] | None = None


class InferenceMeta(BaseModel):
    batch_size: int
    queue_wait_ms: float
    inference_ms: float
    device: str


class InferenceResponse(BaseModel):
    result: Any
    meta: InferenceMeta


UsageType = Literal["classification", "extraction"]
