# Augmented Table Integration Experiment

## Motivation

Task013 identified errors involving values in flattened PDF tables. Task014 showed that explicit table rows can be retrieved in a table-only index. This experiment asks whether those rows remain useful when they compete with the complete production corpus. The result is an experiment, not a production change.

## Experimental design and control

At evaluated commit `441a5ef511b29669490a700daf36b8ff0ca0145e`, the validated production `IndexFlatIP` held 338 chunks. Its existing vectors were reconstructed in vector-ID order; none was re-embedded. The unchanged BGE-M3 wrapper embedded 250 Task014 rows from three PDFs. The separate 588-vector `IndexFlatIP` used dense top-20, the unchanged BGE reranker, and final top-5. No boost, deduplication, or query-specific rule was used. The production baseline and augmented corpus shared the same model instances during the 80-question comparison. Embedding and reranking ran on CPU.

All three fetched PDFs matched their normalized source and accepted Task014 versions:

| Source | PDF SHA-256 | Rows |
|---|---|---:|
| `admission-capacity-pdf` | `748b164c0d1a5e7c82d7fcc4e8e417334be73f062b02c0bed5245e66ae2850c6` | 75 |
| `entrance-exams-list-pdf` | `656ccb00805760f0685c703587c0d5ae5f6dad9ca11ed098f72e6a2180f3d4d7` | 93 |
| `tuition-order-128-pdf` | `455d11c699f84b20739df78aa830bd37ff091b3a13ad776eb11f466a5893544e` | 82 |

The new extraction also matched all three saved Task014 table artifacts field for field. The frozen Task006 and Task010 dataset hashes and the Task011 report hash remained unchanged.

## Phase A — full 80-question retrieval

The production side reproduced the accepted reranked baseline, including exact hit counts. Page metrics have 56 labeled questions; other metrics have 80.

| Metric | Production | Augmented |
|---|---:|---:|
| Primary R@1 | 61/80 = 0.7625 | 65/80 = 0.8125 |
| Primary R@3 | 67/80 = 0.8375 | 69/80 = 0.8625 |
| Primary R@5 | 70/80 = 0.8750 | 72/80 = 0.9000 |
| Primary MRR@5 | 0.8046 | 0.8442 |
| Accepted R@1 | 62/80 = 0.7750 | 67/80 = 0.8375 |
| Accepted R@3 | 69/80 = 0.8625 | 70/80 = 0.8750 |
| Accepted R@5 | 71/80 = 0.8875 | 73/80 = 0.9125 |
| Accepted MRR@5 | 0.8208 | 0.8650 |
| Page R@1 | 32/56 = 0.5714 | 38/56 = 0.6786 |
| Page R@3 | 41/56 = 0.7321 | 45/56 = 0.8036 |
| Page R@5 | 47/56 = 0.8393 | 47/56 = 0.8393 |
| Page MRR@5 | 0.6658 | 0.7440 |

## Table subset and non-table guardrail

For the frozen 16 table questions, Primary R@1/3/5 changed from **11/13/13** to **14/15/16**. Page R@1/3/5 changed from **6/9/13** to **10/12/12**. Expected pages for `gen-012` (`ret-020`) and `gen-022` (`ret-044`) fell out of augmented top-5, while `gen-020` (`ret-038`) gained its expected page. Thus table Page R@5 lost one net hit despite stronger early ranks.

For the other 64 questions, Primary R@1 changed from **50/64** to **51/64**, but Primary R@5 fell from **57/64** to **56/64**. `ret-052` and `ret-071` lost their primary source from top-5; `ret-039` gained it. Accepted R@5 remained **57/64**, and Page R@5 rose from **34/40** to **35/40**.

Representation competition was substantial. Across all 80 queries, table rows occupied an average **6.925/20** dense candidates and **1.0125/5** reranked results. For the 16 table questions these averages were **12.9375/20** and **2.8125/5**; for the other 64, **5.4219/20** and **0.5625/5**. The local JSON report contains each question's ranks and candidate-composition counts.

## Mandatory rows

All four expected rows remained in augmented reranked top-5. At least one production chunk was also present for three questions.

| Question | Table row dense rank | Table row reranked rank | Production chunk in top-5 | First representation |
|---|---:|---:|---|---|
| `gen-008` | 1 | 1 | Yes | Table row |
| `gen-011` | 1 | 1 | Yes | Table row |
| `gen-021` | 3 | 2 | Yes | Table row |
| `gen-027` | 2 | 2 | No | Table row |

## Frozen retrieval gate

| Condition | Result |
|---|---|
| Production baseline reproduced | Pass |
| Four mandatory rows in augmented top-5 | Pass |
| Overall Primary R@5 does not fall | Pass |
| Overall Accepted R@5 does not fall | Pass |
| Overall Page R@5 does not fall | Pass |
| Primary R@1 loss is at most 0.025 | Pass |
| Primary MRR@5 loss is at most 0.025 | Pass |
| Table-16 Page R@5 does not fall | **Fail: 13/16 → 12/16** |
| Non-table-64 Primary R@5 has no net loss | **Fail: 57/64 → 56/64** |

Phase A verdict: `retrieval_gate_failed`. The thresholds were not changed after measurement.

## Phase B — generation

Not run. The frozen retrieval gate forbids calling Ollama when Phase A fails. No new answers, citations, latency measurements, or critical-token diagnostics were produced. The Task011 generation report remains the baseline; it was not modified.

## Limitations and verdict

Final verdict: **`augmented_retrieval_failed_gate`**. The four structural probes and aggregate retrieval metrics improved, but two predeclared regression guards failed. The extra 250 vectors increase index size and reranking competition. Retrieval metrics do not establish factual correctness of generated answers; human semantic validation would still be needed before any production adoption.

The engineering next step is to investigate the specific displaced pages and sources, including dense candidate pools and table-row competition, in a separate experiment. Task015 does not adopt the augmented index in production.
