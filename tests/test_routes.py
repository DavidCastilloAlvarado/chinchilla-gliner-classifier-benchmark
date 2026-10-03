"""Unit tests for endpoint adapters and OpenAPI configuration."""

import unittest
from types import SimpleNamespace

from fastapi import HTTPException

from app.api.routes import _system_one_state_text, system_one
from app.api.schemas import SystemOneRequest
from app.main import app


class FakeBatcher:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    async def submit(self, query: str, operation: dict) -> dict:
        self.calls.append((query, operation))
        return {"result": self.result, "meta": {}}


class SystemOneRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.loaded_model = "fastino/GLiNER2.5-multi-Decide"
        self.batcher = FakeBatcher(
            {
                "model": self.loaded_model,
                "answers": {"urgent": {"type": "noul", "noul": 0.8}},
                "usage": {"input_tokens": 4, "output_tokens": 0},
            }
        )
        self.runtime = SimpleNamespace(
            batcher=self.batcher,
            settings=SimpleNamespace(model_repo=self.loaded_model),
        )

    async def test_system_one_builds_industry_operation(self) -> None:
        payload = SystemOneRequest.model_validate(
            {
                "model": self.loaded_model,
                "state": {"message": "The card is lost."},
                "questions": {
                    "urgent": {
                        "type": "noul",
                        "instructions": "Does this need immediate attention?",
                    }
                },
            }
        )

        response = await system_one(payload, self.runtime)

        assert response.model == self.loaded_model
        assert response.answers["urgent"]["type"] == "noul"
        assert self.batcher.calls[0][0] == '{"message":"The card is lost."}'
        assert self.batcher.calls[0][1]["kind"] == "systemone"

    async def test_system_one_rejects_a_model_not_loaded_by_server(self) -> None:
        payload = SystemOneRequest.model_validate(
            {
                "model": "jev-latest",
                "state": "A card is lost.",
                "questions": {
                    "urgent": {
                        "type": "noul",
                        "instructions": "Is this urgent?",
                    }
                },
            }
        )

        with self.assertRaises(HTTPException) as raised:
            await system_one(payload, self.runtime)

        assert raised.exception.status_code == 422
        assert self.batcher.calls == []


def test_state_text_preserves_strings_and_structured_json() -> None:
    assert _system_one_state_text("plain text") == "plain text"
    assert _system_one_state_text({"a": 1}) == '{"a":1}'
    assert _system_one_state_text(["a", 2]) == '["a",2]'


def test_swagger_contains_valid_system_one_examples_and_no_redoc() -> None:
    openapi = app.openapi()
    operation = openapi["paths"]["/v1/systemone"]["post"]
    request_example = operation["requestBody"]["content"]["application/json"]["examples"]["banking_triage"]["value"]
    response_example = operation["responses"]["200"]["content"]["application/json"]["examples"]["banking_triage"]["value"]

    assert request_example["model"] == "fastino/GLiNER2.5-multi-Decide"
    assert request_example["questions"]["urgent"]["type"] == "noul"
    assert response_example["answers"]["severity"]["type"] == "score"
    assert app.redoc_url is None
    assert "/api/redoc" not in openapi["paths"]
