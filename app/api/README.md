# System One endpoint

This directory contains the HTTP contract for the local GLiNER2 serving API.
The free-form, industry-compatible endpoint is:

```http
POST /v1/systemone
```

It follows the TypeSafe/System One request and response shape described in the
[TypeSafe API reference](https://docs.typesafe.ai/api): one state is evaluated
against one or more named, independent questions. The `model` field must match
the actual local repository loaded by the server. With the default
`MODEL_NAME=decide`, use `fastino/GLiNER2.5-multi-Decide`; sending
`jev-latest` is not valid for this local server and returns HTTP 422.

## Swagger

Swagger UI is the only interactive API documentation exposed by this service:

```text
http://localhost:8000/api/doc
```

The `/v1/systemone` operation includes a complete banking triage request example
and a matching response example. Use the **Try it out** button to submit the
request against the running local server. The OpenAPI document is available at:

```text
http://localhost:8000/api/openapi.json
```

ReDoc is intentionally disabled.

## Request

```json
{
  "model": "fastino/GLiNER2.5-multi-Decide",
  "state": {
    "message": "Perdí mi tarjeta y necesito bloquearla de inmediato.",
    "language": "es"
  },
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Which banking intent best describes the message?",
      "criteria": {
        "card_block": "The customer wants to block or freeze a lost or stolen card.",
        "check_balance": "The customer wants to know an account balance.",
        "transfer_money": "The customer wants to send money to another account."
      }
    },
    "urgent": {
      "type": "noul",
      "instructions": "Does this request require immediate attention?",
      "criteria": {
        "true": "A lost or stolen payment card needs prompt blocking.",
        "false": "The request can wait without immediate risk."
      }
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is the customer issue?",
      "criteria": ["Low impact", "Moderate impact", "High impact"]
    }
  }
}
```

`choice` criteria are a map of option names to descriptions. `noul` is a yes/no
proposition and may include optional `true`/`false` descriptions. `score` criteria
are an ordered list with two to ten levels. The question IDs are preserved in the
`answers` map.

## Response

```json
{
  "model": "fastino/GLiNER2.5-multi-Decide",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "card_block",
      "probabilities": {
        "card_block": 0.94,
        "check_balance": 0.03,
        "transfer_money": 0.03
      },
      "confidence": 0.94
    },
    "urgent": {
      "type": "noul",
      "noul": 0.97
    },
    "severity": {
      "type": "score",
      "score": 1.84,
      "legend": {
        "0": "Low impact",
        "1": "Moderate impact",
        "2": "High impact"
      },
      "probabilities": {
        "0": 0.02,
        "1": 0.12,
        "2": 0.86
      },
      "confidence": 0.86
    }
  },
  "usage": {
    "input_tokens": 123,
    "output_tokens": 0
  }
}
```

`choice` probabilities cover every submitted option and identify the selected
option in `choice`. `noul` is the probability of the `true` outcome. `score` is
the probability-weighted index of the ordered levels; its `legend` maps numeric
indices back to the submitted criteria. Local `input_tokens` is estimated with
the loaded tokenizer and `output_tokens` is zero because GLiNER2 emits structured
decisions rather than generated text.

## Compatibility and batching

The endpoint uses the existing bounded asynchronous micro-batcher. Requests with
the same normalized question schema can share one GLiNER2 inference batch. The
server setting `N_CONCURRENCY=8` is the maximum number of HTTP requests in one
model microbatch; it is independent from the number of Locust virtual users.

- `200`: typed decision response
- `422`: invalid request or malformed question schema
- `429`: bounded inference queue is full
- `503`: model is unavailable or inference failed
- `504`: request exceeded `REQUEST_TIMEOUT_SECONDS`

The previous `/api/v1/classify` endpoint remains available as a legacy
GLiNER2-native interface. New free-form integrations should use `/v1/systemone`.
