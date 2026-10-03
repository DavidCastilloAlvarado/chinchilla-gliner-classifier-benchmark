"""Business logic for entity extraction requests."""

from __future__ import annotations

from typing import Any

from fastapi import Depends

from app.api.common import extraction_state_to_text, require_loaded_model, submit_inference
from app.core.dependencies import get_runtime
from app.core.lifespan import Runtime

from .schema import ExtractionRequest, ExtractionResponse


class ExtractionService:
    """Coordinates GLiNER2 entity extraction and output shaping."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    async def extract(self, payload: ExtractionRequest) -> ExtractionResponse:
        served_model = require_loaded_model(self.runtime, payload.model)
        query = extraction_state_to_text(payload.state)
        operation: dict[str, Any] = {
            "kind": "extraction",
            "entities": {
                label: description if description is not None else label
                for label, description in payload.entities.items()
            },
            "threshold": payload.threshold,
            "include_confidence": payload.include_confidence,
            "include_spans": payload.include_spans,
        }
        result = await submit_inference(self.runtime, query, operation)
        raw_result = result["result"]
        entities = raw_result.get("entities", raw_result)
        return ExtractionResponse(
            model=served_model,
            entities=entities,
            usage={
                "input_tokens": self.runtime.model._estimate_system_one_tokens(
                    query,
                    payload.entities,
                ),
                "output_tokens": 0,
            },
        )


def get_extraction_service(runtime: Runtime = Depends(get_runtime)) -> ExtractionService:
    """Dependency factory for the extraction controller."""
    return ExtractionService(runtime)
