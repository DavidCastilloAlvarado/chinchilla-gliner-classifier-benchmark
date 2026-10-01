"""Generate 1,000 banking intent classification examples (10 intents x 100).

Usage:
    uv run python src/classifier/generate_examples.py

Output: data/banking_intents.jsonl  (one JSON object per line: id, text, intent)
"""

import json
import random
from pathlib import Path

SEED = 42
N_PER_INTENT = 100
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "banking_intents.jsonl"

AMOUNTS = ["50", "100", "250", "500", "1,000", "2,500", "750", "3,000", "150", "900"]
CURRENCIES = ["dollars", "euros", "pounds", "pesos", "yen", "dollars"]
ACCOUNTS = [
    "savings account", "checking account", "current account", "money market account",
    "savings account", "checking account", "business account", "joint account",
]
CARDS = ["debit card", "credit card", "bank card", "ATM card"]
BILLS = ["electricity bill", "water bill", "internet bill", "phone bill", "rent", "insurance premium", "gas bill", "streaming subscription"]
LOANS = ["personal loan", "auto loan", "mortgage", "student loan", "business loan", "home equity loan"]
TIMEFRAMES = ["today", "yesterday", "this morning", "last night", "a few minutes ago", "earlier today", "this week", "last month"]
PLACES = ["the mall", "a gas station", "a grocery store", "an online shop", "a restaurant", "the airport", "a pharmacy", "a hotel"]
CARD_TYPES = ["travel rewards", "cash back", "low annual fee", "business", "student", "premium"]
PREFIXES = ["", "", "", "Hi, ", "Hello, ", "Hey, "]

# Each intent: list of templates. {slot} placeholders are filled randomly.
TEMPLATES: dict[str, list[str]] = {
    "check_balance": [
        "Can I check the balance of my {account}?",
        "How much money do I have in my {account}?",
        "I'd like to see the current balance on my {account}.",
        "What's my {account} balance right now?",
        "Please show me how much is left in my {account}.",
        "I want to know my available balance in the {account}.",
        "Could you tell me the balance of my {account}?",
        "How many {currency} are in my {account}?",
        "I need to check my {account} balance before I make a purchase.",
        "What is the total amount in my {account}?",
    ],
    "transfer_money": [
        "I want to transfer {amount} {currency} to my savings.",
        "Please send {amount} {currency} to account ending in 4821.",
        "I need to move {amount} {currency} from my checking to savings.",
        "Transfer {amount} {currency} to my friend's account, please.",
        "I'd like to wire {amount} {currency} to another bank.",
        "Send {amount} {currency} to the account I have on file.",
        "I want to make a transfer of {amount} {currency} to my other account.",
        "Please transfer {amount} {currency} to my brother's checking account.",
        "I need to send money, {amount} {currency}, to a different bank.",
        "Move {amount} {currency} from my current account to my savings.",
    ],
    "pay_bill": [
        "I want to pay my {bill} online.",
        "Please schedule a payment for my {bill}.",
        "Can I set up an automatic payment for my {bill}?",
        "I'd like to pay the {bill} of {amount} {currency}.",
        "How do I pay my {bill} from my account?",
        "Please make a payment of {amount} {currency} for my {bill}.",
        "I need to pay my {bill} before the due date.",
        "Can you help me pay the {bill} from my checking account?",
        "I want to schedule a recurring payment for my {bill}.",
        "Please process a payment for my {bill} today.",
    ],
    "report_lost_card": [
        "I lost my {card}, can you block it?",
        "My {card} is missing, I need to cancel it.",
        "I can't find my {card}, please freeze it.",
        "I think I lost my {card} {timeframe}, please deactivate it.",
        "My {card} went missing, I want to report it lost.",
        "Please block my {card}, I believe I lost it.",
        "I misplaced my {card} and I want to cancel it.",
        "My {card} is nowhere to be found, please suspend it.",
        "I need to report my {card} as lost.",
        "Can you deactivate my {card}? I lost it.",
    ],
    "report_fraud": [
        "There is an unauthorized transaction on my {card}, it's fraud.",
        "I see a charge I didn't make on my {card}, please report it.",
        "Someone used my {card} without permission, I want to report fraud.",
        "I need to dispute a fraudulent charge of {amount} {currency} on my {card}.",
        "There's a suspicious transaction on my account, I think it's fraud.",
        "I didn't authorize the last purchase on my {card}, please investigate.",
        "My {card} was used fraudulently {timeframe}, I want to report it.",
        "Please flag a fraudulent charge on my {card} from {place}.",
        "I want to report a scam transaction on my {card}.",
        "An unknown person made a purchase with my {card}, this is fraud.",
    ],
    "apply_for_loan": [
        "I want to apply for a {loan}.",
        "Can I get a {loan} with my current credit?",
        "I'd like to request a {loan} of {amount} {currency}.",
        "How do I apply for a {loan}?",
        "I need a {loan}, what are the requirements?",
        "Please start an application for a {loan} for me.",
        "I want to finance {amount} {currency} with a {loan}.",
        "What's the interest rate on your {loan}?",
        "I'm interested in getting a {loan}, can you help me apply?",
        "I need to borrow {amount} {currency}, I want a {loan}.",
    ],
    "apply_for_credit_card": [
        "I want to apply for a new {card_type} credit card.",
        "Can I get a {card_type} credit card?",
        "I'd like to request a credit card application.",
        "How do I apply for a credit card?",
        "I want a credit card with a low annual fee.",
        "Please send me the credit card application form.",
        "I'm looking for a {card_type} credit card.",
        "Can I upgrade my current card to a premium credit card?",
        "I want to open a new credit card account.",
        "What credit cards do you offer? I want to apply.",
        "I'd like to request a {card_type} credit card.",
        "Can you tell me about your {card_type} credit card?",
    ],
    "close_account": [
        "I want to close my {account}.",
        "Please cancel my {account}, I no longer need it.",
        "I'd like to terminate my {account}.",
        "How do I close my {account}?",
        "I want to shut down my {account} and withdraw the remaining balance.",
        "Please close my {account} effective end of month.",
        "I no longer want this {account}, please close it.",
        "I'm leaving the bank, I need to close my {account}.",
        "Can you help me close my {account} online?",
        "I want to deactivate and close my {account}.",
    ],
    "open_account": [
        "I want to open a new {account}.",
        "Can I open a {account} at your bank?",
        "I'd like to open a {account}, what do I need?",
        "How do I open a {account} online?",
        "I want to start a new {account} with you.",
        "Please help me open a {account} in my name.",
        "I'm looking to open a {account} with a low minimum balance.",
        "What are the fees to open a {account}?",
        "I want to open a {account} for my business.",
        "Can I open a {account} from the mobile app?",
    ],
    "reset_password": [
        "I forgot my {channel} password, I need to reset it.",
        "Can you help me reset my {channel} password?",
        "I can't log in to {channel}, I need to recover my password.",
        "Please send me a password reset link for {channel}.",
        "I want to change my {channel} password.",
        "My {channel} password isn't working, I need to reset it.",
        "How do I reset my {channel} password?",
        "I locked myself out of {channel}, I need to recover my password.",
        "I want to update my password for the mobile app.",
        "Please help me set a new password for my account.",
    ],
}

SLOTS = {
    "amount": AMOUNTS,
    "currency": CURRENCIES,
    "account": ACCOUNTS,
    "card": CARDS,
    "bill": BILLS,
    "loan": LOANS,
    "timeframe": TIMEFRAMES,
    "place": PLACES,
    "card_type": CARD_TYPES,
    "channel": ["online banking", "the mobile app", "the web portal", "the banking app", "the website"],
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
