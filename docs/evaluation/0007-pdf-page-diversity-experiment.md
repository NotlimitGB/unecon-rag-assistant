# Post-Reranker PDF Page Diversity Experiment

## Method and controls

Evaluated commit: `0c3f942799324c21cffdf01c06ea0ff11e431f67` on `main`. Task006 and Task010 dataset SHA-256 values were `52de939e1ba13d1558c3e96fa6cec2cabdb69fca9158996aa1a4282d4167511e` and `11a284b4279f627ae787d0ec4777e0884733b4ec937d6658be3857c2dd15d0a4`. The frozen Task011 generation report SHA-256 was `9e9a2d8052bc74b3bc397f98a83b9b703e3d5fb52d90c75a2fe0f0cc998ef6ff`.

The experiment reused Task016's validated 338-vector production index, rebuilt the separate 250-row table index with the unchanged BGE-M3 embedder, selected production top-20 and table top-5, and scored the 25 merged candidates once with the unchanged BGE reranker. The only new rule walks that complete reranker order and accepts no more than **two PDF results per `(source_id, page_start)`** until five results are selected. A production chunk and a table row on the same PDF page share the limit. HTML has no page limit. Scores and relative order are unchanged. The cap has no tuning parameter or fallback. Embedding and reranking used CPU, selected by the existing `auto` settings. Production retrieval, generation, prompts, source artifacts, models, and datasets were not changed.

All three official PDF byte hashes matched the accepted Task014 sources: `admission-capacity-pdf` `748b164c0d1a5e7c82d7fcc4e8e417334be73f062b02c0bed5245e66ae2850c6` (75 rows), `entrance-exams-list-pdf` `656ccb00805760f0685c703587c0d5ae5f6dad9ca11ed098f72e6a2180f3d4d7` (93), and `tuition-order-128-pdf` `455d11c699f84b20739df78aa830bd37ff091b3a13ad776eb11f466a5893544e` (82). Table extraction reproduced the saved artifacts. The production index hash matched Task016's recorded hash; its corpus had 338 chunks. The production top-20 matched direct search on **80/80** questions. All **80 complete lists of 25 IDs, positions, and scores** matched Task017's local trace within the frozen score tolerance of `1e-4`. The accepted production and raw Task016 metrics also reproduced before the diversity results were interpreted.

## Phase A: retrieval results

Primary and Accepted denominators are 80; Page uses the 56 questions with verified page labels. Counts are shown at ranks 1/3/5, followed by MRR@5.

| Metric | Production | Raw Task016 | PDF page cap |
|---|---:|---:|---:|
| Primary R@1/3/5 | 61/67/70 | 63/69/73 | **63/70/74** |
| Primary MRR@5 | 0.8046 | 0.8321 | **0.8363** |
| Accepted R@1/3/5 | 62/69/71 | 65/71/74 | **65/71/74** |
| Accepted MRR@5 | 0.8208 | 0.8546 | **0.8546** |
| Page R@1/3/5 | 32/41/47 | 36/45/47 | **36/45/49** |
| Page MRR@5 | 0.6658 | 0.7202 | **0.7274** |

| Subset and R@5 | Production | Raw Task016 | PDF page cap |
|---|---:|---:|---:|
| Table 16 Primary / Accepted / Page | 13 / 14 / 13 | 16 / 16 / 12 | **16 / 16 / 14** |
| Other 64 Primary / Accepted | 57 / 57 | 57 / 58 | **58 / 58** |
| Other 64 Page (40 labeled) | 34 | 35 | **35** |

Relative to production top-5, the capped policy introduced **no new** Primary, Accepted, or Page loss. It recovered Primary hits for `ret-037`, `ret-038`, `ret-039`, `ret-051`; Accepted hits for `ret-037`, `ret-038`, `ret-039`; and Page hits for `ret-038`, `ret-039`. Within the other 64, there were no new Primary or Accepted misses and no lost successful page label.

The policy changed final top-5 on **14/80** questions: `ret-012`, `ret-013`, `ret-014`, `ret-018`, `ret-020`, `ret-034`, `ret-036`, `ret-044`, `ret-051`, `ret-052`, `ret-054`, `ret-055`, `ret-059`, `ret-060`. It skipped 44 same-page candidates while filling those lists (maximum eight in a question); 30 replacements came from original ranks 6–10 (26) or 11–15 (4). Eight of the 16 table questions and six of the other 64 changed. The count of questions with three or more final results from one PDF source and page fell from **14 to 0**; three or more table rows from one page fell from **8 to 0**. The broader count with three or more results from any one source fell only from **64 to 60** because different pages and HTML remain unrestricted. The ignored retrieval JSON contains every original and selected top-5, skip, original rank, replacement pairing, and per-question movement.

All four mandatory rows survived the cap. Their raw reranker rank and capped rank were **1→1** (`gen-008`, capacity p. 2), **1→1** (`gen-011`, capacity p. 3), **2→2** (`gen-021`, exams p. 2), and **2→2** (`gen-027`, tuition p. 2). Each had a second selected result from its own PDF page; no mandatory row was skipped.

### Three displaced-evidence cases

| Question | Expected evidence | Raw Task016 rank | Capped rank | Observed selection effect |
|---|---|---:|---:|---|
| `ret-020` | Capacity PDF p. 4 | 9 | **5** | Four p. 1 siblings at raw ranks 3, 4, 5, 8 were skipped. |
| `ret-044` | Exams PDF p. 8 | 7 | **5** | Two p. 1 table rows at raw ranks 4, 5 were skipped. |
| `ret-052` | Primary `tuition` HTML | 6 | **3** | Three tuition PDF p. 3 rows at raw ranks 3, 4, 5 were skipped. HTML was uncapped. |

The rule recovers the expected page or source for these cases by altering selection after reranking. It does not show that the model's scores were wrong, and a page hit does not prove the selected chunk contains every fact needed to answer.

### Seventeen frozen Phase A conditions

| # | Condition | Result |
|---:|---|---|
| 1 | Accepted production baseline reproduced | Pass: Primary/Accepted/Page R@5 **70/71/47** and MRR matched |
| 2 | Raw Task016 reproduced | Pass: Primary/Accepted/Page R@5 **73/74/47** and MRR matched |
| 3 | All Task017 25-candidate orders and scores matched | Pass: **80/80** |
| 4 | Five final results per question | Pass: **80/80** |
| 5 | Four mandatory rows in final top-5 | Pass: ranks **1, 1, 2, 2** |
| 6 | Overall Primary R@5 ≥ 70/80 | Pass: **74/80** |
| 7 | Overall Accepted R@5 ≥ 71/80 | Pass: **74/80** |
| 8 | Overall Page R@5 ≥ 47/56 | Pass: **49/56** |
| 9 | Table Primary R@5 ≥ 16/16 | Pass: **16/16** |
| 10 | Table Accepted R@5 ≥ 16/16 | Pass: **16/16** |
| 11 | Table Page R@5 ≥ 13/16 | Pass: **14/16** |
| 12 | No new Primary@5 miss among other 64 | Pass: **0** |
| 13 | No new Accepted@5 miss among other 64 | Pass: **0** |
| 14 | No lost successful production Page@5 label among 56 | Pass: **0** |
| 15 | `ret-020` page 4 recovered | Pass: rank **5** |
| 16 | `ret-044` page 8 recovered | Pass: rank **5** |
| 17 | `ret-052` primary `tuition` HTML recovered | Pass: rank **3** |

Phase A passed **17/17** conditions without changing the frozen gate.

## Phase B: controlled generation

Only after Phase A passed, the unchanged Task009 `AnswerService` ran once for each of the frozen 16 supported questions, using the exact cached diversity top-5 for that question. There was no repeated retrieval or reranking in this phase. The local Ollama tag was exactly `qwen3.5:9b`; settings remained `grounded-answer-v1`, temperature 0.1, 512 output tokens, `think=false`, reranked top-5. Frozen Task011 responses were read from its unchanged report. The ignored generation JSON preserves each original and new answer, complete citations and selected context texts, statuses, source/page comparison, and latency.

| Structural measure (16 questions) | Frozen Task011 | Diversity context |
|---|---:|---:|
| Valid structured responses | 16 | **16** |
| Answered / insufficient evidence | 12 / 4 | **15 / 1** |
| Answered with citation | 12 | **15** |
| Accepted-source hit / Primary-source hit | 10 / 10 | **14 / 14** |
| Expected-page hit | 9 | **13** |

Three prior refusals became answers (`gen-007`, `gen-025`, `gen-030`); `gen-012` remained a refusal. There were **zero new refusals and zero errors**. Mean/median full-call latency for the new 16 answers was **2.861/2.330 seconds**. Four mandatory probes remained answered, cited their expected table rows, and contained their preregistered numeric tokens: `gen-008` **686**, `gen-011` **8**, `gen-021` **60**, `gen-027` **179500** (allowing space as a thousands separator). These are string checks, not factual grading. For the two repaired retrieval cases tied to generation, `gen-012` remained `insufficient_evidence` despite the page-4 candidate entering top-5, while `gen-022` remained answered and cited exams PDF p. 8. The full old and new answer text is retained verbatim in the ignored report.

## Verdict and limits

**`diversity_integration_promising`**: all 17 retrieval gates passed; Phase B completed without structural errors or new refusals; four required answers and numeric tokens were present. This is an engineering verdict for the tested frozen questions. Citation/source hits, token presence, and structural validity do **not** establish that the generated answers are factually correct or complete. In particular, `gen-012` still refused despite a recovered page, so retrieval improvement alone did not resolve every generation case. Human semantic review remains necessary before any production decision.

The complete retrieval and generation reports are local ignored files under `data/processed/evaluation/`. The policy remains isolated to `app.experiments.table_aware`; production retrieval and generation are unchanged.
