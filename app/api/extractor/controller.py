"""FastAPI controller for GLiNER2 entity extraction."""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends

from .schema import ExtractionRequest, ExtractionResponse
from .service import ExtractionService, get_extraction_service

router = APIRouter()

EXTRACTION_REQUEST_EXAMPLES = {
    "pii_extraction": {
        "summary": "Extract PII entities with confidence and spans",
        "description": "Extract names, email addresses, phone numbers, and financial identifiers.",
        "value": {
            "model": "fastino/GLiNER2.5-multi-Decide",
            "state": {"message": "Mi nombre es Ana López, mi correo es ana.lopez@example.com y mi teléfono es +51 999 123 456.", "language": "es"},
            "entities": {
                "person_name": "A person's full name or named individual.",
                "email": "An email address.",
                "phone": "A telephone or mobile number.",
                "financial_identifier": "A bank account, card, or payment identifier.",
            },
            "threshold": 0.1,
            "include_confidence": True,
            "include_spans": True,
        },
    }
}

EXTRACTION_RESPONSE_EXAMPLES = {
    "pii_extraction": {
        "summary": "Extracted PII entities",
        "value": {
            "model": "fastino/GLiNER2.5-multi-Decide",
            "entities": {
                "person_name": [{"text": "Ana López", "confidence": 0.17, "start": 13, "end": 22}],
                "email": [{"text": "ana.lopez@example.com", "confidence": 0.66, "start": 37, "end": 58}],
                "phone": [{"text": "+51 999 123 456", "confidence": 0.24, "start": 76, "end": 91}],
                "financial_identifier": [],
            },
            "usage": {"input_tokens": 63, "output_tokens": 0},
        },
    }
}


@router.post(
    "/v1/extraction",
    response_model=ExtractionResponse,
    response_model_exclude_none=True,
    summary="Extract typed entities from text",
    description="Extract named entity categories from one state using GLiNER2.",
    responses={200: {"description": "Extracted entities grouped by category.", "content": {"application/json": {"examples": EXTRACTION_RESPONSE_EXAMPLES}}}},
)
async def extraction(
    payload: ExtractionRequest = Body(..., openapi_examples=EXTRACTION_REQUEST_EXAMPLES),
    service: ExtractionService = Depends(get_extraction_service),
) -> ExtractionResponse:
    """Extract requested entity categories with optional confidence and spans."""
    return await service.extract(payload)
