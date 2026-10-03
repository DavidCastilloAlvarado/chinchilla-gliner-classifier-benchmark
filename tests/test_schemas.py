"""Validation tests for the public FastAPI request models."""

import pytest
from pydantic import ValidationError

from app.api.schemas import SystemOneRequest


@pytest.fixture
def system_one_payload() -> dict:
    return {
        "model": "fastino/GLiNER2.5-multi-Decide",
        "state": {"message": "I lost my card."},
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "What is the banking intent?",
                "criteria": {
                    "card_block": "Block a lost or stolen card",
                    "check_balance": "Check an account balance",
                },
            },
            "urgent": {
                "type": "noul",
                "instructions": "Does this need immediate attention?",
            },
            "severity": {
                "type": "score",
                "instructions": "How severe is this?",
                "criteria": ["Low", "High"],
            },
        },
    }


def test_system_one_accepts_all_question_primitives(system_one_payload: dict) -> None:
    request = SystemOneRequest.model_validate(system_one_payload)

    assert request.model == "fastino/GLiNER2.5-multi-Decide"
    assert request.questions["intent"].type == "choice"
    assert request.questions["urgent"].type == "noul"
    assert request.questions["severity"].type == "score"


def test_system_one_rejects_unknown_question_type(system_one_payload: dict) -> None:
    system_one_payload["questions"]["unknown"] = {
        "type": "ranking",
        "instructions": "Rank this",
        "criteria": ["a", "b"],
    }

    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(system_one_payload)


def test_system_one_rejects_score_with_one_level(system_one_payload: dict) -> None:
    system_one_payload["questions"]["severity"]["criteria"] = ["Only one level"]

    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(system_one_payload)
