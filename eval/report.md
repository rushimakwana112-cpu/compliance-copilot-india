# Evaluation report

Run 2026-10-15 12:48 · model `groq:qwen/qwen3.8-27b` · embeddings `llama-text-embed-v2` · threshold 0.38 · top_k 12 → keep 5 · chunks 1800 chars / 200 overlap · router `rules`

GROQ_MODEL `qwen/qwen3.8-27b` · avg prompt tokens 2,879 (RAG answers, n=11, budget 3000 for passages) · 26 asked, 7 skipped (not in the metrics)

| Metric | Value | n |
|---|---|---|
| route accuracy | 100% | 26 |
| retrieval hit@4 | 92% | 12 |
| answer contains | 91% | 22 |
| not_found precision | 83% | 6 |
| not_found recall | 100% | 5 |
| median latency (s) | 1.93 | 26 |
| avg prompt tokens | 2,879 | 11 |

## Per question

| id | type | route | answered as | hit@4 | contains | latency |
|---|---|---|---|---|---|---|
| q01 | rag | | skipped (doc not ingested) | | | |
| q02 | rag | rag ✓ | rag | ✓ | ✓ | 7.7s |
| q03 | rag | rag ✓ | rag | ✓ | ✓ | 5.3s |
| q04 | rag | rag ✓ | rag | ✓ | ✓ | 3.6s |
| q05 | rag | rag ✓ | rag | ✓ | ✓ | 14.6s |
| q06 | rag | rag ✓ | rag | ✓ | ✓ | 12.0s |
| q07 | rag | rag ✓ | not_found | ✗ | ✗ | 1.9s |
| q08 | rag | | skipped (doc not ingested) | | | |
| q09 | rag | | skipped (doc not ingested) | | | |
| q10 | rag | | skipped (doc not ingested) | | | |
| q11 | rag | | skipped (doc not ingested) | | | |
| q12 | rag | | skipped (doc not ingested) | | | |
| q13 | rag | rag ✓ | rag | ✓ | ✓ | 8.3s |
| q14 | rag | rag ✓ | rag | ✓ | ✓ | 5.4s |
| q15 | rag | rag ✓ | rag | ✓ | ✓ | 10.2s |
| q16 | rag | | skipped (doc not ingested) | | | |
| q17 | tool | tool ✓ | tool |  | ✓ | 0.0s |
| q18 | tool | tool ✓ | tool |  | ✓ | 0.4s |
| q19 | tool | tool ✓ | tool |  | ✓ | 0.0s |
| q20 | tool | tool ✓ | tool |  | ✓ | 0.0s |
| q21 | tool | tool ✓ | tool |  | ✓ | 0.0s |
| q22 | sql | sql ✓ | sql |  | ✓ | 0.6s |
| q23 | sql | sql ✓ | sql |  | ✓ | 0.9s |
| q24 | sql | sql ✓ | sql |  | ✓ | 0.4s |
| q25 | sql | sql ✓ | sql |  | ✓ | 0.5s |
| q26 | not_found | out_of_scope ✓ | not_found |  | ✓ | 0.0s |
| q27 | not_found | out_of_scope ✓ | not_found |  |  | 0.0s |
| q28 | not_found | rag ✓ | not_found |  |  | 1.9s |
| q29 | not_found | rag ✓ | not_found |  |  | 2.0s |
| q30 | not_found | rag ✓ | not_found |  |  | 1.9s |
| q31 | rag | rag ✓ | rag | ✓ | ✗ | 4.0s |
| q32 | rag | rag ✓ | rag | ✓ | ✓ | 5.3s |
| q33 | rag | rag ✓ | rag | ✓ | ✓ | 4.2s |

## Failures to look at

- **q07** Is bonus paid to contract workers too? → `not_found`: I couldn't find this in my documents for Central + Gujarat, so I won't guess.  Try rephrasing, pick a different state, or add the official document from the sid
- **q31** Definition of wages under the Code on Wages → `rag`: Yes, "wages" means all remuneration, whether by way of salaries, allowances or otherwise, expressed in terms of money or capable of being so expressed which wou

## Skipped

- **q01** Is ESIC applicable on ₹24,000 gross?: doc not ingested (needs ESIC)
- **q08** When must the employer deposit PF contributions each month?: doc not ingested (needs EPF Scheme|Employees' Provident Funds Scheme)
- **q09** Professional tax for ₹15,000 salary in Ahmedabad: doc not ingested (needs Gujarat Professional Tax|Gujarat PT)
- **q10** What about Maharashtra?: doc not ingested (needs Maharashtra Professional Tax|Maharashtra PT)
- **q11** Compare Gujarat and Maharashtra professional tax: doc not ingested (needs Gujarat Professional Tax|Gujarat PT&Maharashtra Professional Tax|Maharashtra PT)
- **q12** Does TDS apply on salary below the basic exemption limit?: doc not ingested (needs TDS|Income-tax|Income Tax|CBDT)
- **q16** What did the old Payment of Gratuity Act say about eligibility?: doc not ingested (needs Payment of Gratuity Act)
