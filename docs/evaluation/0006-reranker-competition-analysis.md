# Reranker Competition Analysis

## Motivation

Task015's shared dense pool displaced production candidates. Task016 separated the channels and preserved the canonical production top‑20 for all 80 questions, yet lost two expected PDF pages and one primary HTML source at the final top‑5 cutoff. This report examines the unchanged reranker's complete 25-candidate ordering. It does not measure generated answers or factual correctness.

## Method and integrity

Evaluated commit: `f06b6c2b3fba6af50287e9b7b95a21e931edbd1e`. The Task006 dataset SHA‑256 remained `52de939e1ba13d1558c3e96fa6cec2cabdb69fca9158996aa1a4282d4167511e`. All three PDF bytes matched the Task014/016 hashes; `lines_strict` reproduced 250 saved rows. The validated production index contained 338 chunks. BGE‑M3 and `BAAI/bge-reranker-v2-m3` ran on CPU with the accepted settings, including reranker max length 512.

For every frozen question, the existing Task016 path produced production top‑20 plus table top‑5, merged them in dense-score order, and called the existing `rerank_candidates` once with `top_k=25`. Its first five IDs, order and scores matched the saved Task016 top‑5; direct production top‑20 matched for **80/80** questions. The production baseline and Task016 dual metrics reproduced, including overall Primary R@1/3/5 **63/69/73 of 80**, Accepted R@5 **74/80**, Page R@1/3/5 **36/45/47 of 56**, table‑16 Primary/Page R@5 **16/12 of 16**, and non-table‑64 Primary/Accepted/Page R@5 **57/58 of 64, 35 of 40**. The complete 2000-candidate trace is in the ignored local JSON artifact.

The score distribution below uses raw reranker logits and inclusive linear interpolation for quartiles. Text lengths are exact candidate characters and whitespace-separated words after whitespace normalization. Pair lengths use the loaded `XLMRobertaTokenizer` without truncation, then compare with the configured 512-token limit; none of the 2000 pairs exceeds it. Lexical coverage uses casefolded unique Unicode word tokens without stemming or stopwords. These are descriptive measures, not ranking features or semantic grades.

## Global statistics

| Questions | Representation | Candidates | Score mean / median / p25 / p75 | Top‑5 entries | Mean / median reranker rank | Mean chars / words / pair tokens | Truncated |
|---|---|---:|---|---:|---:|---:|---:|
| All 80 | Production chunk | 1600 | −3.85 / −4.17 / −5.93 / −1.92 | 341 (21.31%) | 12.01 / 12 | 998 / 130 / 244 | 0 |
| All 80 | Table row | 400 | −5.36 / −5.78 / −8.28 / −3.43 | 59 (14.75%) | 16.96 / 20 | 567 / 80 / 157 | 0 |
| Table 16 | Production chunk | 320 | −3.71 / −3.89 / −5.40 / −2.14 | 43 (13.44%) | 14.25 / 15 | 987 / 129 / 251 | 0 |
| Table 16 | Table row | 80 | −1.53 / −1.51 / −3.75 / −0.16 | 37 (46.25%) | 8.01 / 6 | 564 / 80 / 157 | 0 |
| Other 64 | Production chunk | 1280 | −3.89 / −4.28 / −6.07 / −1.85 | 298 (23.28%) | 11.45 / 11 | 1000 / 130 / 243 | 0 |
| Other 64 | Table row | 320 | −6.32 / −6.52 / −8.98 / −5.02 | 22 (6.88%) | 19.19 / 21 | 567 / 80 / 157 | 0 |

The 400 table candidates occupied 59 positions at ranks 1–5, 33 at 6–10 and 308 at 11–25. On table questions, those counts were 37/19/24; on other questions, 22/14/284. Thus table rows gained scores *within the table-question subset*, while their overall mean score was lower than the prose candidates' mean. These conditional distributions do not establish a general model preference for tables.

At least three final results came from one source for 64 questions; this includes ordinary production chunks. At least three table rows from one source appeared for nine questions, and from one source **and page** for eight. A table row entered final top‑5 for 24 questions. Among those 24, primary ranks improved/worsened/stayed equal on **5/1/18** questions; accepted ranks on **5/0/19**; expected-page ranks on **5/2/17**.

## Lost vs recovered evidence

Relative to production-only top‑5, the dual path lost expected-page hits for **`ret-020`** and **`ret-044`**, and the primary-source hit for **`ret-052`**. No Accepted@5 hit was lost. It recovered primary hits for `ret-037`, `ret-038`, `ret-039`, `ret-051`; accepted hits for `ret-037`, `ret-038`, `ret-039`; and page hits for `ret-038`, `ret-039`. A recovery does not cancel an individual new miss under Task016's frozen gate. Full displaced-candidate IDs, scores and entrants are in the local JSON.

## ret-020 — total budget places

The question asks for the total across the bachelor and specialist list. The verified page‑4 source contains the total line **276 budget, 2608 paid general competition, 138 separate competition**. Two page‑4 production chunks survived dense selection at channel ranks **5** and **13**, merged positions **10** and **18**, then reranker ranks **9** (score **−0.0119**) and **19** (score **−2.4979**). The total line is in the latter chunk at rank 19. This distinction matters: the best page‑4 candidate by score is not the chunk containing the total.

All five table-channel candidates were individual program rows from **page 1** of the same PDF. They occupied all five final places, with scores **3.3043, 2.8968, 2.3361, 2.0876, 1.9575**. The fifth-place cutoff exceeded the best page‑4 score by **1.9694** and the total-bearing chunk by **4.4554**. The rank‑5/6 margin was **1.5789**. All expected candidates had fewer than 512 pair tokens; truncation was absent. The table rows repeatedly state per-program budget counts, while the question requests the aggregate.

**Primary pattern:** `expected_evidence_low_reranker_score`. **Contributors:** `source_saturation`, `same_page_row_saturation`. Candidate absence and truncation are ruled out. The trace supports a page‑1 sibling-row takeover at the cutoff; it does not establish a universal score advantage for table text.

## ret-044 — format of internal exams

The expected page‑8 production chunks survived dense selection at channel ranks **7** and **12**, merged positions **8** and **17**, then reranker ranks **7** (score **−1.6411**) and **11** (score **−2.1942**). The first gives computer testing and distance format; the second carries the exception for an in-person interview. Both were within the 512-token limit.

Final rank 1 was a page‑1 production chunk from the same PDF (score **2.0572**), but its text introduces the list of fields, including that format information will be given; it does not itself state the actual format. Ranks 3–5 were three **page‑1** table rows about exam subjects and minimum scores, at **−0.8280, −0.8800, −0.9331**. Rank 2 was a deadlines PDF chunk. The expected page‑8 chunk was **0.7079** below the fifth-place cutoff; the rank‑5/6 margin was **0.6107**. The three sibling rows share page and much of their table context, yet concern subjects and score thresholds rather than the requested format.

**Primary pattern:** `expected_evidence_low_reranker_score`. **Contributors:** `source_saturation`, `same_page_row_saturation`. The requested evidence was present and untruncated. The observed competition favors adjacent page‑1 context over page‑8 format details in this query; no independent semantic relevance score was assigned.

## ret-052 — which tuition order

The primary `tuition` HTML chunk explicitly names **приказ № 128 от 13.04.2026**. It survived production dense selection at channel rank **4**, entered the merged list at position **9**, and finished **sixth** at score **2.2335**. Five page‑3 rows from the accepted `tuition-order-128-pdf` occupied ranks 1–5 with scores **2.9502, 2.9450, 2.8231, 2.6897, 2.4019**. The HTML chunk missed the cutoff by **0.1684**, equal to the rank‑5/6 margin. Each winning row's serialized source title also names order № 128, although its cells describe a particular program and price. Their IDs are recorded in full in the local trace; all five share the same PDF page.

**Primary pattern:** `source_saturation`. **Contributors:** `expected_evidence_low_reranker_score`, `same_page_row_saturation`. The accepted PDF source remained in top‑5, so this is specifically a **Primary@5** regression, not an Accepted@5 failure. Both the HTML chunk and winning rows were untruncated. A narrow cutoff gap makes the boundary worth testing later, but no instability threshold or counterfactual reranking was introduced here.

## Cross-case conclusion and engineering implication

All three expected sources/pages survived the protected dense selection. Their losses occurred after the 25 candidates competed for five final slots. In the two page regressions, multiple sibling rows from a different page of the same official PDF occupied final slots; in `ret-052`, five rows from one PDF page displaced the direct HTML page by a small score margin. None of the 2000 pairs was truncated, so token-length truncation does not explain these observed losses. The frozen labels and exact texts support these statements, but the experiment did not grade candidate correctness or generated answers.

The single next experiment best supported by this evidence is an **isolated source/page diversity constraint applied after the unchanged reranker scores**, tested against the same frozen 80-question gate. Task017 does not implement or endorse such a rule for production. Its benefit and any new losses remain unmeasured.

The diagnostic JSON remains under `data/processed/evaluation/reranker_competition_analysis.json` and is ignored by Git. Production retrieval, table extraction, datasets, prompts, dependencies and parameters were unchanged. Ollama was not called.
