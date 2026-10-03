"""Unit tests for endpoint adapters and OpenAPI configuration."""

import unittest
from types import SimpleNamespace

from fastapi import HTTPException

from app.api.routes import _extraction_state_text, _system_one_state_text, extraction, system_one
from app.api.schemas import ExtractionRequest, SystemOneRequest
from app.api.extractor.service import ExtractionService
from app.api.systemone.service import SystemOneService
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
            model=SimpleNamespace(
                _estimate_system_one_tokens=lambda query, entities: 12,
            ),
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

        response = await system_one(payload, SystemOneService(self.runtime))

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
            await system_one(payload, SystemOneService(self.runtime))

        assert raised.exception.status_code == 422
        assert self.batcher.calls == []

    async def test_extraction_returns_entities_and_spans(self) -> None:
        self.batcher.result = {
            "entities": {
                "email": [
                    {"text": "a@example.com", "confidence": 0.99, "start": 0, "end": 13}
                ]
            }
        }
        payload = ExtractionRequest.model_validate(
            {
                "model": self.loaded_model,
                "state": "Contact a@example.com",
                "entities": {"email": "Email addresses"},
                "include_confidence": True,
                "include_spans": True,
            }
        )

        response = await extraction(payload, ExtractionService(self.runtime))

        assert response.model == self.loaded_model
        assert response.entities["email"][0]["text"] == "a@example.com"
        assert response.usage == {"input_tokens": 12, "output_tokens": 0}
        assert self.batcher.calls[0][0] == "Contact a@example.com"
        assert self.batcher.calls[0][1]["kind"] == "extraction"
        assert self.batcher.calls[0][1]["threshold"] == 0.5
        assert self.batcher.calls[0][1]["include_spans"] is True


def test_state_text_preserves_strings_and_structured_json() -> None:
    assert _system_one_state_text("plain text") == "plain text"
    assert _system_one_state_text({"a": 1}) == '{"a":1}'
    assert _system_one_state_text(["a", 2]) == '["a",2]'
    assert _extraction_state_text({"message": "source text", "language": "es"}) == "source text"


def test_swagger_contains_valid_system_one_examples_and_no_redoc() -> None:
    openapi = app.openapi()
    operation = openapi["paths"]["/v1/systemone"]["post"]
    request_examples = operation["requestBody"]["content"]["application/json"]["examples"]
    response_examples = operation["responses"]["200"]["content"]["application/json"]["examples"]

    assert set(request_examples) == {"banking_triage", "police_incident_classification", "pii_detection"}
    assert set(response_examples) == set(request_examples)
    for example in request_examples.values():
        request = SystemOneRequest.model_validate(example["value"])
        assert request.model == "fastino/GLiNER2.5-multi-Decide"

    assert request_examples["police_incident_classification"]["value"]["questions"]["incident_severity"]["type"] == "score"
    assert request_examples["pii_detection"]["value"]["questions"]["contains_email"]["type"] == "noul"
    assert response_examples["banking_triage"]["value"]["answers"]["severity"]["type"] == "score"

    extraction_operation = openapi["paths"]["/v1/extraction"]["post"]
    extraction_request = extraction_operation["requestBody"]["content"]["application/json"]["examples"]["pii_extraction"]["value"]
    ExtractionRequest.model_validate(extraction_request)
    assert extraction_request["model"] == "fastino/GLiNER2.5-multi-Decide"
    assert extraction_request["threshold"] == 0.1
    assert extraction_request["include_confidence"] is True
    assert extraction_request["include_spans"] is True
    assert extraction_operation["responses"]["200"]["content"]["application/json"]["examples"]["pii_extraction"]["value"]["entities"]["email"]
    assert app.redoc_url is None
    assert "/api/redoc" not in openapi["paths"]
