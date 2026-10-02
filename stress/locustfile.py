"""Locust load test for the GLiNER2 FastAPI micro-batcher.

Examples:
    uv run locust -f stress/locustfile.py -H http://127.0.0.1:8000
    uv run locust -f stress/locustfile.py --headless -u 16 -r 16 -t 30s \
        -H http://127.0.0.1:8000

Use SERVER_MODE=adhoc to test requests with client-provided labels instead of
stored app schemas. The server's /metrics endpoint is the authoritative source
for observed model batch sizes.
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, constant, task

APP_ID = os.getenv("LOCUST_APP_ID", "banking_es")
USAGE_TYPE = os.getenv("LOCUST_USAGE_TYPE", "classification")
SERVER_MODE = os.getenv("SERVER_MODE", "stored").lower()
WAIT_MIN = float(os.getenv("LOCUST_WAIT_MIN", "0"))
WAIT_MAX = float(os.getenv("LOCUST_WAIT_MAX", "0"))

QUERIES = [
    "Quiero consultar el saldo de mi cuenta corriente.",
    "Necesito transferir dinero a otra cuenta bancaria.",
    "Quiero pagar la factura de electricidad.",
    "Perdí mi tarjeta de débito y necesito bloquearla.",
    "Hay un cargo que no reconozco en mi cuenta.",
    "Quiero solicitar un préstamo personal.",
    "Quiero abrir una tarjeta de crédito nueva.",
    "Me gustaría cerrar mi cuenta bancaria.",
    "Quiero abrir una cuenta de ahorros.",
    "Necesito restablecer la contraseña de la aplicación.",
]

CLASSES = [
    "check_balance",
    "transfer_money",
    "pay_bill",
    "report_lost_card",
    "report_fraud",
    "apply_for_loan",
    "apply_for_credit_card",
    "close_account",
    "open_account",
    "reset_password",
]

DESCRIPTIONS = {
    "check_balance": "Consultar el saldo o el dinero disponible en una cuenta",
    "transfer_money": "Mover o enviar dinero de una cuenta a otra cuenta o persona",
    "pay_bill": "Pagar una factura o recibo de un servicio",
    "report_lost_card": "Reportar una tarjeta perdida o extraviada y bloquearla",
    "report_fraud": "Reportar una transacción no autorizada o fraude",
    "apply_for_loan": "Solicitar un préstamo o crédito",
    "apply_for_credit_card": "Solicitar o abrir una tarjeta de crédito nueva",
    "close_account": "Cerrar o dar de baja una cuenta bancaria",
    "open_account": "Abrir una cuenta bancaria nueva",
    "reset_password": "Restablecer o cambiar la contraseña de banca en línea",
}


class BankingUser(HttpUser):
    """Generate concurrent requests to one FastAPI pod."""

    wait_time = constant(0) if WAIT_MIN == WAIT_MAX == 0 else between(WAIT_MIN, WAIT_MAX)

    @task
    def classify_or_extract(self) -> None:
        query = random.choice(QUERIES)
        if SERVER_MODE == "adhoc":
            path = "/api/v1/classify"
            payload = {
                "query": query,
                "classes": CLASSES,
                "descriptions": DESCRIPTIONS,
            }
            name = "POST /api/v1/classify"
        else:
            path = f"/app/{APP_ID}/usage/{USAGE_TYPE}/"
            payload = {"query": query}
            name = f"POST /app/{{app_id}}/usage/{USAGE_TYPE}"

        with self.client.post(path, json=payload, name=name, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
                return
            try:
                body = response.json()
                meta = body["meta"]
                if not isinstance(meta.get("batch_size"), int):
                    response.failure("response meta.batch_size is missing or invalid")
            except (ValueError, KeyError, TypeError) as exc:
                response.failure(f"invalid inference response: {exc}")
