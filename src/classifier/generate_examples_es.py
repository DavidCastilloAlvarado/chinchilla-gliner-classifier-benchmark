"""Generate 1,000 SPANISH banking intent classification examples (10 intents x 100).

Usage:
    uv run python src/classifier/generate_examples_es.py

Output: data/banking_intents_es.jsonl  (one JSON object per line: id, text, intent)
"""

import json
import random
from pathlib import Path

SEED = 42
N_PER_INTENT = 100
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "banking_intents_es.jsonl"

MONTO = ["50", "100", "250", "500", "1.000", "2.500", "750", "3.000", "150", "900"]
MONEDA = ["dólares", "euros", "libras", "pesos", "yenes", "dólares"]
CUENTA = [
    "cuenta de ahorros", "cuenta corriente", "cuenta de cheques", "cuenta de dinero en efectivo",
    "cuenta de ahorros", "cuenta corriente", "cuenta empresarial", "cuenta conjunta",
]
TARJETA = ["tarjeta de débito", "tarjeta de crédito", "tarjeta bancaria", "tarjeta ATM"]
FACTURA = ["recibo de luz", "recibo de agua", "recibo de internet", "recibo de teléfono", "renta", "póliza de seguro", "recibo de gas", "suscripción de streaming"]
PRESTAMO = ["préstamo personal", "préstamo automotriz", "hipoteca", "préstamo estudiantil", "préstamo empresarial", "préstamo sobre la equity de la vivienda"]
TIEMPO = ["hoy", "ayer", "esta mañana", "anoche", "hace unos minutos", "hace una semana", "el mes pasado"]
LUGAR = ["el centro comercial", "una gasolinera", "un supermercado", "una tienda en línea", "un restaurante", "el aeropuerto", "una farmacia", "un hotel"]
TIPO_TARJETA = ["de millas", "con cash back", "de bajo costo anual", "empresarial", "estudiantil", "premium"]
CANAL = ["la banca en línea", "la app móvil", "el portal web", "la app del banco", "la página web"]
PREFIXES = ["", "", "", "Hola, ", "Buenos días, ", "Buenas tardes, "]

TEMPLATES: dict[str, list[str]] = {
    "check_balance": [
        "¿Puedo consultar el saldo de mi {cuenta}?",
        "¿Cuánto dinero tengo en mi {cuenta}?",
        "Me gustaría ver el saldo actual de mi {cuenta}.",
        "¿Cuál es el saldo de mi {cuenta} ahora mismo?",
        "Por favor, muéstrame cuánto me queda en la {cuenta}.",
        "Quisiera saber el saldo disponible de mi {cuenta}.",
        "¿Me puede decir el saldo de mi {cuenta}?",
        "¿Cuántos {moneda} hay en mi {cuenta}?",
        "Necesito consultar el saldo de mi {cuenta} antes de hacer una compra.",
        "¿Cuál es el monto total en mi {cuenta}?",
    ],
    "transfer_money": [
        "Quiero transferir {monto} {moneda} a mi cuenta de ahorros.",
        "Por favor, envíe {monto} {moneda} a la cuenta que termina en 4821.",
        "Necesito mover {monto} {moneda} de mi cuenta corriente a ahorros.",
        "Transfiera {monto} {moneda} a la cuenta de mi amigo, por favor.",
        "Me gustaría hacer una transferencia bancaria de {monto} {moneda}.",
        "Envíe {monto} {moneda} a la cuenta que tengo registrada.",
        "Quiero hacer una transferencia de {monto} {moneda} a mi otra cuenta.",
        "Por favor, transfiera {monto} {moneda} a la cuenta de cheques de mi hermano.",
        "Necesito enviar dinero, {monto} {moneda}, a otro banco.",
        "Mueva {monto} {moneda} de mi cuenta corriente a mi cuenta de ahorros.",
    ],
    "pay_bill": [
        "Quiero pagar mi {factura} en línea.",
        "Por favor, programe un pago para mi {factura}.",
        "¿Puedo configurar un pago automático para mi {factura}?",
        "Me gustaría pagar la {factura} de {monto} {moneda}.",
        "¿Cómo pago mi {factura} desde mi cuenta?",
        "Por favor, realice un pago de {monto} {moneda} por mi {factura}.",
        "Necesito pagar mi {factura} antes de la fecha de vencimiento.",
        "¿Puede ayudarme a pagar la {factura} desde mi cuenta corriente?",
        "Quiero programar un pago recurrente para mi {factura}.",
        "Por favor, procese un pago por mi {factura} hoy.",
    ],
    "report_lost_card": [
        "Perdí mi {tarjeta}, ¿puede bloquearla?",
        "Mi {tarjeta} no está, necesito cancelarla.",
        "No encuentro mi {tarjeta}, por favor congélela.",
        "Creo que perdí mi {tarjeta} {tiempo}, por favor desactívela.",
        "Mi {tarjeta} se extravió, quiero reportarla como perdida.",
        "Por favor, bloquee mi {tarjeta}, creo que la perdí.",
        "Perdí mi {tarjeta} y quiero cancelarla.",
        "Mi {tarjeta} no aparece por ninguna parte, por favor suspenda.",
        "Necesito reportar mi {tarjeta} como perdida.",
        "¿Puede desactivar mi {tarjeta}? La perdí.",
    ],
    "report_fraud": [
        "Hay una transacción no autorizada en mi {tarjeta}, es fraude.",
        "Veo un cargo que no hice en mi {tarjeta}, por favor repórtelo.",
        "Alguien usó mi {tarjeta} sin permiso, quiero reportar fraude.",
        "Necesito impugnar un cargo fraudulento de {monto} {moneda} en mi {tarjeta}.",
        "Hay una transacción sospechosa en mi cuenta, creo que es fraude.",
        "No autoricé la última compra en mi {tarjeta}, por favor investigue.",
        "Mi {tarjeta} fue usada fraudulentamente {tiempo}, quiero reportarlo.",
        "Por favor, marque un cargo fraudulento en mi {tarjeta} de {lugar}.",
        "Quiero reportar una transacción estafada en mi {tarjeta}.",
        "Una persona desconocida hizo una compra con mi {tarjeta}, esto es fraude.",
    ],
    "apply_for_loan": [
        "Quiero solicitar un {prestamo}.",
        "¿Puedo obtener un {prestamo} con mi crédito actual?",
        "Me gustaría solicitar un {prestamo} de {monto} {moneda}.",
        "¿Cómo solicito un {prestamo}?",
        "Necesito un {prestamo}, ¿cuáles son los requisitos?",
        "Por favor, inicie una solicitud de {prestamo} para mí.",
        "Quiero financiar {monto} {moneda} con un {prestamo}.",
        "¿Cuál es la tasa de interés de su {prestamo}?",
        "Estoy interesado en obtener un {prestamo}, ¿puede ayudarme a solicitarlo?",
        "Necesito pedir prestados {monto} {moneda}, quiero un {prestamo}.",
    ],
    "apply_for_credit_card": [
        "Quiero solicitar una nueva tarjeta de crédito {tipo_tarjeta}.",
        "¿Puedo obtener una tarjeta de crédito {tipo_tarjeta}?",
        "Me gustaría solicitar una tarjeta de crédito.",
        "¿Cómo solicito una tarjeta de crédito?",
        "Quiero una tarjeta de crédito con bajo costo anual.",
        "Por favor, envíeme el formulario de solicitud de tarjeta de crédito.",
        "Estoy buscando una tarjeta de crédito {tipo_tarjeta}.",
        "¿Puedo actualizar mi tarjeta actual a una tarjeta de crédito premium?",
        "Quiero abrir una nueva cuenta de tarjeta de crédito.",
        "¿Qué tarjetas de crédito ofrecen? Quiero solicitar una.",
        "Me gustaría solicitar una tarjeta de crédito {tipo_tarjeta}.",
        "¿Puede contarme sobre su tarjeta de crédito {tipo_tarjeta}?",
    ],
    "close_account": [
        "Quiero cerrar mi {cuenta}.",
        "Por favor, cancele mi {cuenta}, ya no la necesito.",
        "Me gustaría dar de baja mi {cuenta}.",
        "¿Cómo cierro mi {cuenta}?",
        "Quiero cerrar mi {cuenta} y retirar el saldo restante.",
        "Por favor, cierre mi {cuenta} a fin de mes.",
        "Ya no quiero esta {cuenta}, por favor ciérrela.",
        "Me estoy yendo del banco, necesito cerrar mi {cuenta}.",
        "¿Puede ayudarme a cerrar mi {cuenta} en línea?",
        "Quiero desactivar y cerrar mi {cuenta}.",
    ],
    "open_account": [
        "Quiero abrir una nueva {cuenta}.",
        "¿Puedo abrir una {cuenta} en su banco?",
        "Me gustaría abrir una {cuenta}, ¿qué necesito?",
        "¿Cómo abro una {cuenta} en línea?",
        "Quiero iniciar una nueva {cuenta} con ustedes.",
        "Por favor, ayúdeme a abrir una {cuenta} a mi nombre.",
        "Estoy buscando abrir una {cuenta} con saldo mínimo bajo.",
        "¿Cuáles son las comisiones para abrir una {cuenta}?",
        "Quiero abrir una {cuenta} para mi negocio.",
        "¿Puedo abrir una {cuenta} desde la app móvil?",
    ],
    "reset_password": [
        "Olvidé mi contraseña de {canal}, necesito restablecerla.",
        "¿Puede ayudarme a restablecer mi contraseña de {canal}?",
        "No puedo iniciar sesión en {canal}, necesito recuperar mi contraseña.",
        "Por favor, envíeme un enlace para restablecer la contraseña de {canal}.",
        "Quiero cambiar mi contraseña de {canal}.",
        "Mi contraseña de {canal} no funciona, necesito restablecerla.",
        "¿Cómo restablezco mi contraseña de {canal}?",
        "Me quedé fuera de {canal}, necesito recuperar mi contraseña.",
        "Quiero actualizar mi contraseña de la app móvil.",
        "Por favor, ayúdeme a configurar una nueva contraseña para mi cuenta.",
    ],
}

SLOTS = {
    "monto": MONTO,
    "moneda": MONEDA,
    "cuenta": CUENTA,
    "tarjeta": TARJETA,
    "factura": FACTURA,
    "prestamo": PRESTAMO,
    "tiempo": TIEMPO,
    "lugar": LUGAR,
    "tipo_tarjeta": TIPO_TARJETA,
    "canal": CANAL,
}


def generate() -> list[dict]:
    rng = random.Random(SEED)
    examples = []
    for intent, templates in TEMPLATES.items():
        seen: set[str] = set()
        count = 0
        attempts = 0
        while count < N_PER_INTENT:
            attempts += 1
            if attempts > 10_000:
                raise RuntimeError(f"Could not generate {N_PER_INTENT} unique examples for {intent}")
            template = rng.choice(templates)
            text = template
            for slot, values in SLOTS.items():
                text = text.replace("{" + slot + "}", rng.choice(values))
            text = rng.choice(PREFIXES) + text
            if text in seen:
                continue
            seen.add(text)
            examples.append({"id": len(examples) + 1, "text": text, "intent": intent})
            count += 1
    rng.shuffle(examples)
    for i, ex in enumerate(examples, start=1):
        ex["id"] = i
    return examples


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    examples = generate()
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples to {OUT_PATH}")
    for intent in TEMPLATES:
        n = sum(1 for e in examples if e["intent"] == intent)
        print(f"  {intent}: {n}")


if __name__ == "__main__":
    main()
