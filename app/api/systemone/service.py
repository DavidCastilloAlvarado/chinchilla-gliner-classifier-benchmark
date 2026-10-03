"""Business logic for System One requests."""

from __future__ import annotations

from typing import Any

from fastapi import Depends

from app.api.common import state_to_text, submit_inference, require_loaded_model
from app.core.dependencies import get_runtime
from app.core.lifespan import Runtime

from .schema import SystemOneRequest, SystemOneResponse


class SystemOneService:
    """Coordinates model validation, batching, and System One response shaping."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    async def evaluate(self, payload: SystemOneRequest) -> SystemOneResponse:
        served_model = require_loaded_model(self.runtime, payload.model)
        operation: dict[str, Any] = {
            "kind": "systemone",
            "model": served_model,
            "questions": {
                question_id: question.model_dump(mode="json", exclude_none=True)
                for question_id, question in payload.questions.items()
            },
        }
        result = await submit_inference(
            self.runtime,
            state_to_text(payload.state),
            operation,
        )
        return SystemOneResponse.model_validate(result["result"])


def get_systemone_service(runtime: Runtime = Depends(get_runtime)) -> SystemOneService:
    """Dependency factory; FastAPI resolves the runtime argument."""
    return SystemOneService(runtime)
