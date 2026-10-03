"""FastAPI controller for the System One endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends

from .schema import SystemOneRequest, SystemOneResponse
from .service import SystemOneService, get_systemone_service

router = APIRouter()

SYSTEM_ONE_REQUEST_EXAMPLES = {
    "banking_triage": {
        "summary": "Banking intent triage",
        "description": "Evaluate several independent decisions over one banking message.",
        "value": {
            "model": "fastino/GLiNER2.5-multi-Decide",
            "state": {"message": "Perdí mi tarjeta y necesito bloquearla de inmediato.", "language": "es"},
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Which banking intent best describes the message?",
                    "criteria": {
                        "card_block": "The customer wants to block or freeze a lost or stolen card.",
                        "check_balance": "The customer wants to know an account balance.",
                        "transfer_money": "The customer wants to send money to another account.",
                    },
                },
                "urgent": {
                    "type": "noul",
                    "instructions": "Does this request require immediate attention?",
                    "criteria": {
                        "true": "A lost or stolen payment card needs prompt blocking.",
                        "false": "The request can wait without immediate risk.",
                    },
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe is the customer issue?",
                    "criteria": ["Low impact", "Moderate impact", "High impact"],
                },
            },
        },
    },
    "police_incident_classification": {
        "summary": "Police incident classification",
        "description": "Classify the incident and assess weapons, damage, medical response, and severity.",
        "value": {
            "model": "fastino/GLiNER2.5-multi-Decide",
            "state": {
                "message": "Ayer por la noche, aproximadamente a las 23:30, tres sujetos encapuchados entraron a mi tienda. Uno de ellos tenía un arma de fuego y me obligó a abrir la caja registradora, mientras los otros dos guardaban licores en mochilas. Se llevaron unos 5,000 soles y huyeron en una camioneta pick-up blanca sin placas. No hubo heridos, pero destrozaron los estantes de vidrio de la entrada.",
                "language": "es",
            },
            "questions": {
                "primary_incident_category": {
                    "type": "choice",
                    "instructions": "What is the primary category of this police incident?",
                    "criteria": {
                        "armed_robbery": "Theft involving the use of weapons, force, or fear.",
                        "burglary": "Illegal entry into a building to commit theft, usually without confronting victims.",
                        "vandalism": "Willful destruction or damage to property without intent to steal.",
                        "assault": "Physical attack or threat of bodily harm.",
                    },
                },
                "subclass_property_damage": {
                    "type": "noul",
                    "instructions": "Did the incident result in physical damage to the property or premises?",
                    "criteria": {
                        "true": "Windows, doors, shelves, or other structures were broken or vandalized.",
                        "false": "No physical property damage is reported.",
                    },
                },
                "subclass_weapons_involved": {
                    "type": "noul",
                    "instructions": "Were firearms or other deadly weapons explicitly mentioned by the reporter?",
                    "criteria": {
                        "true": "Suspects carried or used guns, knives, or similar weapons.",
                        "false": "No weapons were seen or mentioned.",
                    },
                },
                "subclass_vehicle_type": {
                    "type": "choice",
                    "instructions": "What type of getaway vehicle was used by the suspects, if any?",
                    "criteria": {
                        "motorcycle": "A two-wheeled motorized vehicle.",
                        "sedan_or_car": "A standard passenger car.",
                        "truck_or_pickup": "A pickup truck, van, or heavy vehicle.",
                        "none_or_on_foot": "Suspects fled on foot or no vehicle was mentioned.",
                    },
                },
                "medical_dispatch_needed": {
                    "type": "noul",
                    "instructions": "Are there reported injuries that require an immediate ambulance or medical dispatch?",
                    "criteria": {
                        "true": "The reporter mentions physical injuries, bleeding, or individuals needing medical help.",
                        "false": "The reporter explicitly states there are no injuries or does not mention any harm.",
                    },
                },
                "incident_severity": {
                    "type": "score",
                    "instructions": "Based on the presence of weapons, financial loss, and property damage, how severe is this incident?",
                    "criteria": [
                        "Low severity: Minor theft or vandalism, no weapons, no injuries.",
                        "Medium severity: Significant theft or damage, but no weapons or direct threats.",
                        "High severity: Use of weapons, large financial loss, or direct threats to life, requiring priority police response.",
                        "Critical severity: Active shooter, severe injuries, or ongoing hostage situations.",
                    ],
                },
            },
        },
    },
    "pii_detection": {
        "summary": "PII detection and redaction decisions",
        "description": "Detect PII categories before sending text to downstream systems.",
        "value": {
            "model": "fastino/GLiNER2.5-multi-Decide",
            "state": {"message": "Mi nombre es Ana López y mi correo es ana.lopez@example.com.", "language": "es"},
            "questions": {
                "contains_person_name": {"type": "noul", "instructions": "Does the input contain a person's name?"},
                "contains_email": {"type": "noul", "instructions": "Does the input contain an email address?"},
                "pii_category": {
                    "type": "choice",
                    "instructions": "What is the most sensitive PII category present in the input?",
                    "criteria": {
                        "person_name": "A person's first name, last name, or full name.",
                        "email": "An email address.",
                        "phone": "A telephone or mobile number.",
                        "financial": "A bank account, card, or payment identifier.",
                        "none": "No personally identifiable information is present.",
                    },
                },
                "redaction_required": {"type": "noul", "instructions": "Should this input be redacted before being shared?"},
            },
        },
    },
}

SYSTEM_ONE_RESPONSE_EXAMPLES = {
    "banking_triage": {
        "summary": "Structured System One decisions",
        "value": {
            "model": "fastino/GLiNER2.5-multi-Decide",
            "answers": {
                "intent": {"type": "choice", "choice": "card_block", "probabilities": {"card_block": 0.94, "check_balance": 0.03, "transfer_money": 0.03}, "confidence": 0.94},
                "urgent": {"type": "noul", "noul": 0.97},
                "severity": {"type": "score", "score": 1.84, "legend": {"0": "Low impact", "1": "Moderate impact", "2": "High impact"}, "probabilities": {"0": 0.02, "1": 0.12, "2": 0.86}, "confidence": 0.86},
            },
            "usage": {"input_tokens": 123, "output_tokens": 0},
        },
    },
    "police_incident_classification": {
        "summary": "Structured police incident decisions",
        "value": {"model": "fastino/GLiNER2.5-multi-Decide", "answers": {"primary_incident_category": {"type": "choice", "choice": "armed_robbery", "probabilities": {"armed_robbery": 0.97, "burglary": 0.01, "vandalism": 0.01, "assault": 0.01}, "confidence": 0.97}, "subclass_property_damage": {"type": "noul", "noul": 0.98}, "subclass_weapons_involved": {"type": "noul", "noul": 0.99}, "subclass_vehicle_type": {"type": "choice", "choice": "truck_or_pickup", "probabilities": {"motorcycle": 0.01, "sedan_or_car": 0.01, "truck_or_pickup": 0.97, "none_or_on_foot": 0.01}, "confidence": 0.97}, "medical_dispatch_needed": {"type": "noul", "noul": 0.01}, "incident_severity": {"type": "score", "score": 2.44, "legend": {"0": "Low severity", "1": "Medium severity", "2": "High severity", "3": "Critical severity"}, "probabilities": {"0": 0.01, "1": 0.04, "2": 0.50, "3": 0.45}, "confidence": 0.50}}, "usage": {"input_tokens": 220, "output_tokens": 0}},
    },
    "pii_detection": {
        "summary": "PII detection and redaction decisions",
        "value": {"model": "fastino/GLiNER2.5-multi-Decide", "answers": {"contains_person_name": {"type": "noul", "noul": 0.99}, "contains_email": {"type": "noul", "noul": 0.99}, "pii_category": {"type": "choice", "choice": "email", "probabilities": {"person_name": 0.20, "email": 0.45, "phone": 0.30, "financial": 0.03, "none": 0.02}, "confidence": 0.45}, "redaction_required": {"type": "noul", "noul": 0.99}}, "usage": {"input_tokens": 70, "output_tokens": 0}},
    },
}


@router.post(
    "/v1/systemone",
    response_model=SystemOneResponse,
    response_model_exclude_none=True,
    summary="Evaluate typed System One decisions",
    description="Evaluate independent choice, noul, and score questions over one state.",
    responses={200: {"description": "Typed decisions for every submitted question.", "content": {"application/json": {"examples": SYSTEM_ONE_RESPONSE_EXAMPLES}}}},
)
async def system_one(
    payload: SystemOneRequest = Body(..., openapi_examples=SYSTEM_ONE_REQUEST_EXAMPLES),
    service: SystemOneService = Depends(get_systemone_service),
) -> SystemOneResponse:
    """Evaluate independent choice, noul, and score questions over one state."""
    return await service.evaluate(payload)
