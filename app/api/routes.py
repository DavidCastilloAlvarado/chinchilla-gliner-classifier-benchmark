"""Backward-compatible import shim for the refactored API controllers.

Use ``app.api.router`` for application registration and import endpoint code from
its owning package under ``app.api.systemone``, ``app.api.extractor``, or
``app.api.application``.
"""

from .application.controller import app_usage, classify, liveness, metrics, readiness
from .common import extraction_state_to_text, state_to_text
from .extractor.controller import extraction
from .router import router
from .systemone.controller import system_one

_system_one_state_text = state_to_text
_extraction_state_text = extraction_state_to_text

__all__ = [
    "_extraction_state_text",
    "_system_one_state_text",
    "app_usage",
    "classify",
    "extraction",
    "liveness",
    "metrics",
    "readiness",
    "router",
    "system_one",
]
