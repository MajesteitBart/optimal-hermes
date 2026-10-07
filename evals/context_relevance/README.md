# context_relevance: evals for Jev relevance scoring

Measures the relevance-scoring features (`agent/relevance*.py`, user docs in
`website/docs/user-guide/features/context-relevance.md`) on labeled synthetic data, with live calls
to TypeSafe's Jev model. Only the repository's public bundled and optional skills and synthetic
text are sent.

## Running

Needs `TYPESAFE_API_KEY`. Every response is cached in `results/typesafe_cache.json` by request body,
so re-running a report, or a threshold sweep that reuses scores, costs nothing.

```bash
cd evals/context_relevance
python skill_selection_eval.py                                    # all variants, min_score 5,6,7
python skill_selection_eval.py --variants tool_aware+rerank --min 6,6.5,7
python skill_selection_eval.py --limit 20                         # smoke run
python memory_gates_eval.py                                       # write gate, review gate, recall filter
python memory_gates_eval.py --only write
```

Results land in `results/` as JSON with a summary and per-request details.

## Data (`data/`, see `data/README.md`)

| File | Items | Used for |
| --- | --- | --- |
| `skill_requests.json` | 140 | skill selection: single, multi, none, near miss, follow-up |
| `memory_candidates.json` | 70 | write gate: keep vs reject, by kind |
| `review_conversations.json` | 30 | review gate: durable window or not |
| `recall_cases.json` | 25 | recall filter, 5 with an injection bullet |

Requests whose gold skills are platform-gated on the host (macOS-only skills on Windows) are set
aside, not scored as misses. Two skills the dataset may not name are neutral when attached.

## Results that set the shipped defaults (2026-10-07, jev-1.13.0)

Skill selection, 124 requests, close read on, tool-aware rubric:

| `min_score` | Recall | Precision | Needless attach | Exact |
| --- | --- | --- | --- | --- |
| 6 | 0.855 | 0.741 | 10.0% | 0.661 |
| 6.5 (default) | 0.831 | 0.797 | 7.5% | 0.702 |
| 7 | 0.819 | 0.846 | 5.0% | 0.758 |

The original rubric without the close read, at `min_score` 6: recall 0.843, precision 0.659,
needless attach 17.5%, exact 0.597. Uncached latency with the close read: p50 983 ms, p95 1,272 ms,
about 62k input tokens per request.

Memory gates, with the shipped thresholds: write gate 69/70 correct (routing a superseding entry to
`replace` counts as keep), review gate caught 15/15 durable windows and skipped 12/30 reviews,
recall filter 100% recall at 97% precision with no injection kept. These thresholds were tuned on
the same data, so they are an upper bound.
