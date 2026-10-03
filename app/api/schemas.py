"""Backward-compatible schema exports.

New code should import models from the endpoint package that owns them:
``application.schema``, ``extractor.schema``, or ``systemone.schema``.
"""

from .application.schema import (
    AppUsageRequest,
    ClassificationRequest,
    InferenceMeta,
    InferenceResponse,
    UsageType,
)
from .extractor.schema import ExtractionRequest, ExtractionResponse
from .systemone.schema import (
    SystemOneChoiceQuestion,
    SystemOneNoulQuestion,
    SystemOneQuestion,
    SystemOneRequest,
    SystemOneResponse,
    SystemOneScoreQuestion,
)

__all__ = [
    "AppUsageRequest",
    "ClassificationRequest",
    "ExtractionRequest",
    "ExtractionResponse",
    "InferenceMeta",
    "InferenceResponse",
    "SystemOneChoiceQuestion",
    "SystemOneNoulQuestion",
    "SystemOneQuestion",
    "SystemOneRequest",
    "SystemOneResponse",
    "SystemOneScoreQuestion",
    "UsageType",
]
