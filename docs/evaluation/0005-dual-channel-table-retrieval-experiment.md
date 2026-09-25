# Dual-Channel Table Retrieval Experiment

## Motivation

Task015 showed aggregate retrieval gains from adding explicit PDF table rows to the 338 production chunks, but its shared dense top-20 displaced evidence: table Page R@5 fell from 13/16 to 12/16 and non-table Primary R@5 fell from 57/64 to 56/64. Task016 protected the production dense top-20 before adding five candidates from a separate table index. This remained an isolated experiment.

## Design and experimental control

At evaluated commit `807c263aff36fff9a73702de304de18b9d4044d1`, the production channel used the validated, unchanged 338-vector `IndexFlatIP` to return top-20. A separate 250-vector `IndexFlatIP` embedded the exact Task014 table rows and returned top-5. Each dual query used one normalized BGE-M3 query vector for both channels. The 25 candidates were sorted by dense score and experiment-global vector ID, then the existing `BAAI/bge-reranker-v2-m3` scored all 25 and returned top-5. The production baseline used the same loaded embedder and reranker. An independent direct production search checked candidate preservation; this parity check encoded the query a second time. Embedding and reranking ran on CPU.

No production vectors were copied or re-embedded. There was no routing, score boost, final quota, deduplication, prompt change, or production index write. The frozen Task006 and Task010 dataset hashes and the Task011 report hash remained unchanged.

## Source verification

The three fetched PDF hashes matched the normalized documents and the accepted Task014 artifacts. Extraction reproduced all saved tables field for field.

| Source | SHA-256 | Rows |
|---|---|---:|
| `admission-capacity-pdf` | `748b164c0d1a5e7c82d7fcc4e8e417334be73f062b02c0bed5245e66ae2850c6` | 75 |
| `entrance-exams-list-pdf` | `656ccb00805760f0685c703587c0d5ae5f6dad9ca11ed098f72e6a2180f3d4d7` | 93 |
| `tuition-order-128-pdf` | `455d11c699f84b20739df78aa830bd37ff091b3a13ad776eb11f466a5893544e` | 82 |

## Full 80-question retrieval

The production results reproduced the accepted baseline. For every question, the dual production channel's top-20 chunk IDs, order, and scores matched a direct production dense search within the preset numerical tolerance: **80/80 preserved**. Page metrics use 56 labeled questions; other metrics use 80.

| Metric | Production | Dual channel |
|---|---:|---:|
| Primary R@1 | 61/80 = 0.7625 | 63/80 = 0.7875 |
| Primary R@3 | 67/80 = 0.8375 | 69/80 = 0.8625 |
| Primary R@5 | 70/80 = 0.8750 | 73/80 = 0.9125 |
| Primary MRR@5 | 0.8046 | 0.8321 |
| Accepted R@1 | 62/80 = 0.7750 | 65/80 = 0.8125 |
| Accepted R@3 | 69/80 = 0.8625 | 71/80 = 0.8875 |
| Accepted R@5 | 71/80 = 0.8875 | 74/80 = 0.9250 |
| Accepted MRR@5 | 0.8208 | 0.8546 |
| Page R@1 | 32/56 = 0.5714 | 36/56 = 0.6429 |
| Page R@3 | 41/56 = 0.7321 | 45/56 = 0.8036 |
| Page R@5 | 47/56 = 0.8393 | 47/56 = 0.8393 |
| Page MRR@5 | 0.6658 | 0.7202 |

## Frozen 16 table questions

Primary R@1/3/5 changed from **11/13/13** to **14/15/16**; Primary MRR@5 changed from **0.7396** to **0.9083**. Accepted R@5 changed from **14/16** to **16/16**. Page R@1/3/5 changed from **6/9/13** to **10/12/12**; Page MRR@5 changed from **0.5177** to **0.6771**.

The expected page left top-5 for `ret-020` (`gen-012`) and `ret-044` (`gen-022`), both previously at rank 4. It entered top-5 for `ret-038` (`gen-020`). Thus early Page ranks improved, while Page R@5 lost one net hit.

## Remaining 64 questions

Primary R@1 changed from **50/64** to **49/64**; Primary R@5 stayed **57/64**. Accepted R@5 rose from **57/64** to **58/64**. Page R@5 rose from **34/40** to **35/40**; Primary MRR@5 changed from **0.8208** to **0.8130**.

The exact new Primary@5 miss was **`ret-052`**. `ret-039` was recovered, but the frozen gate does not allow a recovery to compensate for a new miss. There were **no new Accepted@5 misses**. For `ret-052`, all five final results were table rows from `tuition-order-128-pdf`; its primary source disappeared from top-5 while an accepted alternative remained. This is reranker-stage competition despite complete preservation of production dense candidates.

## Candidate competition

The table channel returned exactly five rows per question. Across 80 questions, its 400 candidates came from `admission-capacity-pdf` (155), `entrance-exams-list-pdf` (181), and `tuition-order-128-pdf` (64). Mean table-row occupancy in final top-5 was:

| Group | Dual channel | Task015 shared-index experiment |
|---|---:|---:|
| All 80 | 0.7375 | 1.0125 |
| Table 16 | 2.3125 | 2.8125 |
| Non-table 64 | 0.3438 | 0.5625 |

The separate channel reduced table-row occupancy, but did not remove the observed regressions. These averages were diagnostic, not gate thresholds.

## Mandatory rows

All four expected table rows reached final top-5. “Production present” means at least one production chunk in final top-5; it does not assert that the chunk contains the same fact.

| Question | Table top-5 rank | Merged dense position | Final reranked rank | Production present | First representation |
|---|---:|---:|---:|---|---|
| `gen-008` | 1 | 1 | 1 | Yes | Table row |
| `gen-011` | 1 | 1 | 1 | Yes | Table row |
| `gen-021` | 3 | 3 | 2 | Yes | Table row |
| `gen-027` | 2 | 2 | 2 | No | Table row |

## Frozen retrieval gate

| Condition | Result and evidence |
|---|---|
| 1. Production baseline reproduced | Pass: accepted hit counts and MRR |
| 2. Production top-20 preserved | Pass: 80/80 identical IDs and order, scores within tolerance |
| 3. Four mandatory rows in final top-5 | Pass: ranks 1, 1, 2, 2 |
| 4. Overall Primary R@5 not lower | Pass: 70/80 → 73/80 |
| 5. Overall Accepted R@5 not lower | Pass: 71/80 → 74/80 |
| 6. Overall Page R@5 not lower | Pass: 47/56 → 47/56 |
| 7. Primary R@1 loss at most 0.025 | Pass: 61/80 → 63/80 |
| 8. Primary MRR@5 loss at most 0.025 | Pass: 0.8046 → 0.8321 |
| 9. Table-16 Page R@5 not lower | **Fail: 13/16 → 12/16** |
| 10. No new non-table Primary@5 miss | **Fail: `ret-052`** |
| 11. No new non-table Accepted@5 miss | Pass: none |

Phase A verdict: **`dual_channel_retrieval_gate_failed`**. The gate was not relaxed after observing the data.

## Phase B — generation

Skipped by the frozen gate. **Ollama was not called**, and no new generation answers, citations, latency measurements, or critical-token diagnostics were produced. The Task011 generation baseline was not rerun or modified.

## Verdict and engineering implication

Final verdict: **`dual_channel_retrieval_failed_gate`**. The separate channel protected all production dense candidates and improved aggregate retrieval, but the unchanged reranker still displaced an expected table page and one unrelated primary source from final top-5. The evidence does not support a production-adoption task for this 20+5 configuration. Any future proposal should investigate the specific reranker competition separately; Task016 does not tune or adopt a replacement.

This is a single retrieval experiment. No generation quality or factual accuracy was measured, and human semantic confirmation would remain necessary before production adoption.
