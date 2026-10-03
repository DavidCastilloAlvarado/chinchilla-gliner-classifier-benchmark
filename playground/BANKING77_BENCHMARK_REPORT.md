# Banking77 bilingual SystemOne benchmark report

Run timestamp (UTC): `2026-10-03T22:18:30+00:00`

## Result ranking

Language variants are ranked by macro-F1, then accuracy. The test split contains 3,080 utterances and 77 intents. This run evaluated all 3,080 test examples per language. Spanish rows are translations of the corresponding English rows with labels and order preserved.

| Rank | Test language | Accuracy | Macro-F1 | Mean latency | P95 latency | Examples/s | Failed requests |
|---|---|---|---|---|---|---|---|
| 1 | EN | 60.19% | 58.63% | 671.5 ms | 888.1 ms | 11.90 | 0 |
| 2 | ES | 55.62% | 54.18% | 676.9 ms | 924.2 ms | 11.80 | 0 |

**Higher-ranked variant:** EN (macro-F1 58.63%; accuracy 60.19%).

## Public Jev comparison

Public independent evaluation results are available for Jev. OpenRouter's [Jev 1.13 vs. Claude Opus 5 Banking77 evaluation](https://openrouter.ai/blog/insights/jev-vs-claude-opus-5-classification/) provides the closest direct comparison: it used the full 3,080-example PolyAI test split and 8 concurrent requests.

| System | Test examples | Accuracy | Macro-F1 | Median latency | P95 latency | Invalid responses | Reported cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Local SystemOne run: `fastino/GLiNER2.5-multi-Decide` | 3,080 | 60.19% (1,854/3,080) | 58.63% | 668.8 ms | 888.1 ms | 0 | Not measured |
| Jev 1.13 (OpenRouter, 22 Sep 2026) | 3,080 | 81.0% (95% bootstrap CI: 79.6–82.3%) | 80.5% | 175 ms | 270 ms | 0 | $0.34 total ($0.11/1,000) |

In these reported runs, Jev's accuracy is **20.8 percentage points higher** and its macro-F1 is **21.9 points higher**. The local run's observed median and p95 latencies are approximately **3.8×** and **3.3×** Jev's reported values. This is not a controlled head-to-head comparison: Jev was served through OpenRouter, while the local run used the configured SystemOne endpoint and GLiNER2 model. The prompt criteria also differ: the local run generates short descriptions mechanically from label names, whereas OpenRouter authored one-line criteria. The local accuracy's approximate 95% Wilson interval is 58.45–61.91%, which does not overlap OpenRouter's bootstrap interval; no paired statistical test was performed.

A separate evaluation by [Deußer et al.](https://arxiv.org/abs/2609.37647) reports **79.7% accuracy** (95% bootstrap CI: 78.2–81.2%) on 3,076 examples of `mteb/banking77`, with ECE 0.087. That split differs slightly from the 3,080-row PolyAI test split used here. The paper's raw responses are archived on [Zenodo](https://doi.org/10.5281/zenodo.23039006).

No public Jev result was identified for the Spanish-translated Banking77 test set used here; the cited paper's multilingual results use other datasets. The cited Jev scores are independent evaluation results rather than vendor-published guarantees. ECE was not calculated for the local run, so calibration cannot be compared directly with the paper's ECE.

## Per-language details

| Language | Examples | Correct | Accuracy | Macro precision | Macro recall | Macro-F1 | Mean / median latency | P90 / P95 / P99 latency | Throughput | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EN | 3,080 | 1,854 | 60.19% | 66.14% | 60.19% | 58.63% | 671.5 / 668.8 ms | 758.4 / 888.1 / 1232.3 ms | 11.90 ex/s | 258.9 s |
| ES | 3,080 | 1,713 | 55.62% | 61.09% | 55.62% | 54.18% | 676.9 / 671.6 ms | 816.9 / 924.2 / 1337.4 ms | 11.80 ex/s | 260.9 s |

## Benchmark setup

- Dataset: [PolyAI/Banking77 on Hugging Face](https://huggingface.co/datasets/PolyAI/banking77), test split; 3,080 utterances and 77 intents.
- English CSV: [`data/banking77_test.csv`](../data/banking77_test.csv) (SHA-256 `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`).
- Spanish CSV: [`data/banking77_test_es.csv`](../data/banking77_test_es.csv) (SHA-256 `7ff67379ef690fd8e39d473f48bc2d609051e17c7402e54a1df473a22cf0a155`).
- Model requested: `fastino/GLiNER2.5-multi-Decide`. The endpoint URL is loaded from the project `.env`; it is intentionally not copied into this report.
- Requests: one SystemOne `choice` request per row; 8 concurrent workers, timeout 30.0s, up to four attempts for transient failures.
- Both languages use identical English instructions and canonical English intent option keys/descriptions. The utterance language is the only changed input factor, so the comparison measures cross-lingual robustness under a fixed label prompt.
- Latency is client-observed HTTP round-trip time, including endpoint queueing and response handling; throughput is completed examples divided by wall-clock benchmark time.
- Spanish data was machine-translated from the English test text using Google Translate's public web translation endpoint. Labels and row order were preserved. Translation artifacts or ambiguity can affect Spanish scores; the Spanish scores are not a native-human-translation result.
- Accuracy counts failed or invalid choices as incorrect. Macro-F1 gives equal weight to each of the 77 intents. Per-intent rank is descending F1, with accuracy used as a tiebreaker.

## Intent rankings by language

The full per-intent leaderboard is shown below for each language. `Accuracy` is class recall in this single-label task; `F1` incorporates both false positives and false negatives.

### EN intent ranking

| Rank | Intent | F1 | Accuracy | Support |
|---|---|---|---|---|
| 1 | passcode_forgotten | 97.56% | 100.00% | 40 |
| 2 | edit_personal_details | 93.02% | 100.00% | 40 |
| 3 | visa_or_mastercard | 92.31% | 90.00% | 40 |
| 4 | getting_virtual_card | 85.39% | 95.00% | 40 |
| 5 | top_up_by_cash_or_cheque | 85.33% | 80.00% | 40 |
| 6 | age_limit | 85.11% | 100.00% | 40 |
| 7 | automatic_top_up | 84.06% | 72.50% | 40 |
| 8 | lost_or_stolen_phone | 84.06% | 72.50% | 40 |
| 9 | contactless_not_working | 83.72% | 90.00% | 40 |
| 10 | apple_pay_or_google_pay | 83.15% | 92.50% | 40 |
| 11 | verify_source_of_funds | 81.63% | 100.00% | 40 |
| 12 | terminate_account | 79.21% | 100.00% | 40 |
| 13 | pending_card_payment | 77.78% | 70.00% | 40 |
| 14 | pending_cash_withdrawal | 77.33% | 72.50% | 40 |
| 15 | verify_top_up | 77.08% | 92.50% | 40 |
| 16 | card_about_to_expire | 76.92% | 75.00% | 40 |
| 17 | top_up_limits | 76.47% | 97.50% | 40 |
| 18 | direct_debit_payment_not_recognised | 76.47% | 65.00% | 40 |
| 19 | change_pin | 76.19% | 100.00% | 40 |
| 20 | country_support | 75.00% | 90.00% | 40 |
| 21 | pending_top_up | 73.85% | 60.00% | 40 |
| 22 | transaction_charged_twice | 73.39% | 100.00% | 40 |
| 23 | top_up_failed | 73.33% | 82.50% | 40 |
| 24 | declined_transfer | 73.02% | 57.50% | 40 |
| 25 | disposable_card_limits | 71.79% | 70.00% | 40 |
| 26 | activate_my_card | 71.56% | 97.50% | 40 |
| 27 | Refund_not_showing_up | 70.97% | 55.00% | 40 |
| 28 | transfer_fee_charged | 70.89% | 70.00% | 40 |
| 29 | lost_or_stolen_card | 70.10% | 85.00% | 40 |
| 30 | exchange_charge | 68.24% | 72.50% | 40 |
| 31 | card_not_working | 68.13% | 77.50% | 40 |
| 32 | cash_withdrawal_charge | 67.65% | 57.50% | 40 |
| 33 | request_refund | 67.37% | 80.00% | 40 |
| 34 | extra_charge_on_statement | 66.67% | 72.50% | 40 |
| 35 | declined_card_payment | 64.37% | 70.00% | 40 |
| 36 | get_disposable_virtual_card | 63.83% | 75.00% | 40 |
| 37 | card_linking | 63.49% | 50.00% | 40 |
| 38 | card_payment_wrong_exchange_rate | 63.49% | 50.00% | 40 |
| 39 | top_up_reverted | 62.07% | 45.00% | 40 |
| 40 | cancel_transfer | 61.90% | 65.00% | 40 |
| 41 | card_payment_fee_charged | 61.05% | 72.50% | 40 |
| 42 | wrong_amount_of_cash_received | 60.00% | 45.00% | 40 |
| 43 | pending_transfer | 58.82% | 50.00% | 40 |
| 44 | pin_blocked | 58.70% | 67.50% | 40 |
| 45 | failed_transfer | 57.14% | 45.00% | 40 |
| 46 | virtual_card_not_working | 56.14% | 40.00% | 40 |
| 47 | cash_withdrawal_not_recognised | 55.42% | 57.50% | 40 |
| 48 | wrong_exchange_rate_for_cash_withdrawal | 54.24% | 40.00% | 40 |
| 49 | verify_my_identity | 54.17% | 97.50% | 40 |
| 50 | card_delivery_estimate | 54.05% | 50.00% | 40 |
| 51 | card_payment_not_recognised | 53.85% | 52.50% | 40 |
| 52 | exchange_rate | 53.06% | 97.50% | 40 |
| 53 | balance_not_updated_after_cheque_or_cash_deposit | 50.00% | 37.50% | 40 |
| 54 | transfer_timing | 49.59% | 75.00% | 40 |
| 55 | supported_cards_and_currencies | 48.39% | 37.50% | 40 |
| 56 | transfer_into_account | 48.21% | 67.50% | 40 |
| 57 | fiat_currency_support | 47.89% | 42.50% | 40 |
| 58 | card_arrival | 45.98% | 50.00% | 40 |
| 59 | reverted_card_payment? | 45.16% | 35.00% | 40 |
| 60 | unable_to_verify_identity | 43.08% | 35.00% | 40 |
| 61 | declined_cash_withdrawal | 41.38% | 30.00% | 40 |
| 62 | receiving_money | 39.44% | 35.00% | 40 |
| 63 | balance_not_updated_after_bank_transfer | 38.46% | 25.00% | 40 |
| 64 | top_up_by_card_charge | 38.46% | 25.00% | 40 |
| 65 | exchange_via_app | 37.04% | 37.50% | 40 |
| 66 | topping_up_by_card | 35.00% | 35.00% | 40 |
| 67 | card_swallowed | 32.65% | 20.00% | 40 |
| 68 | card_acceptance | 31.68% | 40.00% | 40 |
| 69 | getting_spare_card | 29.79% | 17.50% | 40 |
| 70 | atm_support | 28.99% | 50.00% | 40 |
| 71 | transfer_not_received_by_recipient | 28.04% | 37.50% | 40 |
| 72 | beneficiary_not_allowed | 22.64% | 15.00% | 40 |
| 73 | why_verify_identity | 13.95% | 7.50% | 40 |
| 74 | order_physical_card | 10.91% | 7.50% | 40 |
| 75 | compromised_card | 8.51% | 5.00% | 40 |
| 76 | top_up_by_bank_transfer_charge | 8.51% | 5.00% | 40 |
| 77 | get_physical_card | 0.00% | 0.00% | 40 |

### EN most common confusions

| Gold intent | Predicted intent | Count |
|---|---|---|
| why_verify_identity | verify_my_identity | 32 |
| unable_to_verify_identity | verify_my_identity | 24 |
| order_physical_card | get_physical_card | 24 |
| get_physical_card | pin_blocked | 21 |
| declined_cash_withdrawal | atm_support | 21 |
| card_swallowed | atm_support | 21 |
| failed_transfer | transfer_not_received_by_recipient | 17 |
| get_physical_card | change_pin | 16 |
| Refund_not_showing_up | request_refund | 16 |
| fiat_currency_support | exchange_rate | 15 |

### ES intent ranking

| Rank | Intent | F1 | Accuracy | Support |
|---|---|---|---|---|
| 1 | visa_or_mastercard | 91.14% | 90.00% | 40 |
| 2 | age_limit | 88.64% | 97.50% | 40 |
| 3 | edit_personal_details | 86.75% | 90.00% | 40 |
| 4 | passcode_forgotten | 86.02% | 100.00% | 40 |
| 5 | getting_virtual_card | 85.06% | 92.50% | 40 |
| 6 | verify_source_of_funds | 83.33% | 100.00% | 40 |
| 7 | top_up_limits | 82.98% | 97.50% | 40 |
| 8 | terminate_account | 82.47% | 100.00% | 40 |
| 9 | automatic_top_up | 79.45% | 72.50% | 40 |
| 10 | pending_cash_withdrawal | 79.41% | 67.50% | 40 |
| 11 | contactless_not_working | 79.12% | 90.00% | 40 |
| 12 | pending_card_payment | 76.06% | 67.50% | 40 |
| 13 | card_about_to_expire | 75.95% | 75.00% | 40 |
| 14 | lost_or_stolen_phone | 75.00% | 60.00% | 40 |
| 15 | virtual_card_not_working | 74.63% | 62.50% | 40 |
| 16 | disposable_card_limits | 74.42% | 80.00% | 40 |
| 17 | activate_my_card | 73.08% | 95.00% | 40 |
| 18 | declined_transfer | 73.02% | 57.50% | 40 |
| 19 | lost_or_stolen_card | 72.53% | 82.50% | 40 |
| 20 | verify_top_up | 71.88% | 57.50% | 40 |
| 21 | card_not_working | 71.60% | 72.50% | 40 |
| 22 | top_up_failed | 70.97% | 82.50% | 40 |
| 23 | change_pin | 67.80% | 100.00% | 40 |
| 24 | card_linking | 67.65% | 57.50% | 40 |
| 25 | country_support | 67.53% | 65.00% | 40 |
| 26 | card_payment_wrong_exchange_rate | 66.67% | 57.50% | 40 |
| 27 | transaction_charged_twice | 64.52% | 100.00% | 40 |
| 28 | pending_top_up | 64.41% | 47.50% | 40 |
| 29 | get_disposable_virtual_card | 64.20% | 65.00% | 40 |
| 30 | pin_blocked | 61.54% | 60.00% | 40 |
| 31 | request_refund | 61.39% | 77.50% | 40 |
| 32 | top_up_reverted | 61.02% | 45.00% | 40 |
| 33 | transfer_fee_charged | 60.98% | 62.50% | 40 |
| 34 | declined_card_payment | 60.61% | 50.00% | 40 |
| 35 | cancel_transfer | 60.00% | 67.50% | 40 |
| 36 | top_up_by_cash_or_cheque | 59.70% | 50.00% | 40 |
| 37 | card_payment_fee_charged | 59.26% | 60.00% | 40 |
| 38 | wrong_amount_of_cash_received | 58.06% | 45.00% | 40 |
| 39 | Refund_not_showing_up | 57.14% | 40.00% | 40 |
| 40 | cash_withdrawal_charge | 54.24% | 40.00% | 40 |
| 41 | direct_debit_payment_not_recognised | 54.24% | 40.00% | 40 |
| 42 | cash_withdrawal_not_recognised | 53.66% | 55.00% | 40 |
| 43 | exchange_rate | 53.62% | 92.50% | 40 |
| 44 | failed_transfer | 53.33% | 40.00% | 40 |
| 45 | card_delivery_estimate | 52.63% | 50.00% | 40 |
| 46 | extra_charge_on_statement | 52.27% | 57.50% | 40 |
| 47 | supported_cards_and_currencies | 50.79% | 40.00% | 40 |
| 48 | top_up_by_card_charge | 50.75% | 42.50% | 40 |
| 49 | apple_pay_or_google_pay | 50.67% | 95.00% | 40 |
| 50 | transfer_timing | 47.46% | 70.00% | 40 |
| 51 | wrong_exchange_rate_for_cash_withdrawal | 47.27% | 32.50% | 40 |
| 52 | exchange_charge | 46.75% | 45.00% | 40 |
| 53 | verify_my_identity | 46.43% | 97.50% | 40 |
| 54 | pending_transfer | 45.16% | 35.00% | 40 |
| 55 | reverted_card_payment? | 44.83% | 32.50% | 40 |
| 56 | transfer_into_account | 44.27% | 72.50% | 40 |
| 57 | card_arrival | 40.48% | 42.50% | 40 |
| 58 | unable_to_verify_identity | 39.29% | 27.50% | 40 |
| 59 | atm_support | 37.04% | 50.00% | 40 |
| 60 | declined_cash_withdrawal | 34.62% | 22.50% | 40 |
| 61 | fiat_currency_support | 34.29% | 45.00% | 40 |
| 62 | balance_not_updated_after_cheque_or_cash_deposit | 33.80% | 30.00% | 40 |
| 63 | card_swallowed | 33.33% | 22.50% | 40 |
| 64 | topping_up_by_card | 32.79% | 25.00% | 40 |
| 65 | card_payment_not_recognised | 31.25% | 25.00% | 40 |
| 66 | receiving_money | 30.61% | 37.50% | 40 |
| 67 | beneficiary_not_allowed | 30.51% | 22.50% | 40 |
| 68 | card_acceptance | 29.91% | 40.00% | 40 |
| 69 | exchange_via_app | 29.21% | 32.50% | 40 |
| 70 | transfer_not_received_by_recipient | 27.64% | 42.50% | 40 |
| 71 | getting_spare_card | 25.00% | 17.50% | 40 |
| 72 | balance_not_updated_after_bank_transfer | 17.02% | 10.00% | 40 |
| 73 | compromised_card | 8.89% | 5.00% | 40 |
| 74 | order_physical_card | 7.69% | 5.00% | 40 |
| 75 | top_up_by_bank_transfer_charge | 4.55% | 2.50% | 40 |
| 76 | get_physical_card | 0.00% | 0.00% | 40 |
| 77 | why_verify_identity | 0.00% | 0.00% | 40 |

### ES most common confusions

| Gold intent | Predicted intent | Count |
|---|---|---|
| why_verify_identity | verify_my_identity | 39 |
| get_physical_card | change_pin | 26 |
| unable_to_verify_identity | verify_my_identity | 25 |
| order_physical_card | get_physical_card | 23 |
| Refund_not_showing_up | request_refund | 22 |
| failed_transfer | transfer_not_received_by_recipient | 18 |
| top_up_by_bank_transfer_charge | transfer_fee_charged | 16 |
| card_swallowed | atm_support | 16 |
| pending_transfer | transfer_timing | 15 |
| declined_cash_withdrawal | atm_support | 14 |

## Reproduce

```bash
uv run python playground/benchmark_banking77.py
```

The script downloads/validates the official test CSV, creates the Spanish CSV if absent, then writes the JSONL predictions, aggregate metrics, and this report. Use `--limit 20` for a small endpoint smoke run; use `--force-translation` to regenerate the Spanish file.

## Dataset citation and sources

BANKING77: Casanueva et al., [Efficient Intent Detection with Dual Sentence Encoders (2020)](https://arxiv.org/abs/2003.04807). Dataset card: [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77). The dataset is listed as CC BY 4.0 on its Hugging Face card. Source test CSV: [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/banking_data/test.csv).

Machine translation: [Google Translate](https://translate.google.com/); generated `2026-10-03T22:09:50+00:00`.
