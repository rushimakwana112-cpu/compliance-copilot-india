# Evaluation report

Run 2026-10-16 04:29 · model `groq:qwen/qwen3.8-27b` · embeddings `llama-text-embed-v2` · threshold 0.38 · top_k 12 → keep 5 · chunks 1800 chars / 200 overlap · router `rules`

GROQ_MODEL `qwen/qwen3.8-27b` · avg prompt tokens 2,963 (RAG answers, n=12, budget 3000 for passages) · 37 asked, 6 skipped (not in the metrics)

| Metric | Value | n |
|---|---|---|
| route accuracy | 100% | 37 |
| retrieval hit@4 | 100% | 12 |
| answer contains | 100% | 33 |
| not_found precision | 100% | 5 |
| not_found recall | 100% | 5 |
| median latency (s) | 0.00 | 37 |
| avg prompt tokens | 2,963 | 12 |

## Per question

| id | type | route | answered as | hit@4 | contains | latency |
|---|---|---|---|---|---|---|
| q01 | rag | | skipped (doc not ingested) | | | |
| q02 | rag | rag ✓ | rag | ✓ | ✓ | 6.4s |
| q03 | rag | rag ✓ | rag | ✓ | ✓ | 6.6s |
| q04 | rag | rag ✓ | rag | ✓ | ✓ | 3.8s |
| q05 | rag | rag ✓ | rag | ✓ | ✓ | 16.0s |
| q06 | rag | rag ✓ | rag | ✓ | ✓ | 13.5s |
| q07 | rag | rag ✓ | rag | ✓ | ✓ | 13.2s |
| q08 | tool | tool ✓ | tool |  | ✓ | 0.0s |
| q09 | rag | | skipped (doc not ingested) | | | |
| q10 | rag | | skipped (doc not ingested) | | | |
| q11 | rag | | skipped (doc not ingested) | | | |
| q12 | rag | | skipped (doc not ingested) | | | |
| q13 | rag | rag ✓ | rag | ✓ | ✓ | 10.8s |
| q14 | rag | rag ✓ | rag | ✓ | ✓ | 6.0s |
| q15 | rag | rag ✓ | rag | ✓ | ✓ | 11.4s |
| q16 | rag | | skipped (doc not ingested) | | | |
| q17 | tool | tool ✓ | tool |  | ✓ | 0.0s |
| q18 | tool | tool ✓ | tool |  | ✓ | 0.5s |
| q19 | tool | tool ✓ | tool |  | ✓ | 0.0s |
| q20 | tool | tool ✓ | tool |  | ✓ | 0.0s |
| q21 | tool | tool ✓ | tool |  | ✓ | 0.0s |
| q22 | guide | guide ✓ | guide |  | ✓ | 0.0s |
| q23 | guide | guide ✓ | guide |  | ✓ | 0.5s |
| q24 | guide | guide ✓ | guide |  | ✓ | 0.0s |
| q25 | guide | guide ✓ | guide |  | ✓ | 0.0s |
| q26 | not_found | out_of_scope ✓ | not_found |  | ✓ | 0.0s |
| q27 | not_found | out_of_scope ✓ | not_found |  |  | 0.0s |
| q28 | not_found | guide ✓ | not_found |  |  | 0.0s |
| q29 | not_found | rag ✓ | not_found |  |  | 2.0s |
| q30 | not_found | guide ✓ | not_found |  |  | 0.0s |
| q31 | rag | rag ✓ | rag | ✓ | ✓ | 4.2s |
| q32 | rag | rag ✓ | rag | ✓ | ✓ | 5.4s |
| q33 | rag | rag ✓ | rag | ✓ | ✓ | 4.6s |
| q34 | tool | tool ✓ | tool |  | ✓ | 0.0s |
| q35 | tool | tool ✓ | tool |  | ✓ | 0.0s |
| q36 | guide | guide ✓ | guide |  | ✓ | 0.0s |
| q37 | tool | tool ✓ | guide |  | ✓ | 0.0s |
| q38 | tool | tool ✓ | tool |  | ✓ | 0.0s |
| q39 | guide | guide ✓ | guide |  | ✓ | 0.0s |
| q40 | tool | tool ✓ | tool |  | ✓ | 0.5s |
| q41 | guide | guide ✓ | guide |  | ✓ | 0.0s |
| q42 | tool | tool ✓ | tool |  | ✓ | 0.0s |
| q43 | guide | guide ✓ | guide |  | ✓ | 0.0s |

## Skipped

- **q01** Is ESIC applicable on ₹24,000 gross?: doc not ingested (needs ESIC)
- **q09** Professional tax for ₹15,000 salary in Ahmedabad: doc not ingested (needs Gujarat Professional Tax|Gujarat PT)
- **q10** What about Maharashtra?: doc not ingested (needs Maharashtra Professional Tax|Maharashtra PT)
- **q11** Compare Gujarat and Maharashtra professional tax: doc not ingested (needs Gujarat Professional Tax|Gujarat PT&Maharashtra Professional Tax|Maharashtra PT)
- **q12** Does TDS apply on salary below the basic exemption limit?: doc not ingested (needs TDS|Income-tax|Income Tax|CBDT)
- **q16** What did the old Payment of Gratuity Act say about eligibility?: doc not ingested (needs Payment of Gratuity Act)
