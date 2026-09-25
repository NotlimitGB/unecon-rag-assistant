# Table-Aware PDF Representation Experiment

## Scope

Task013 found that several answers confused values in linearized PDF tables or lost the heading that prices are **per semester**. This isolated Task014 experiment tests explicit table rows from three existing official PDFs. The production ingestion, normalized PDF artifacts, `paragraph-aware-v1` chunks, index, retrieval service, answer service, prompts, and frozen datasets remain unchanged. Ollama was not used; the experiment does not measure factual-answer improvement.

Evaluated production commit: `4d815d31b6df81171884f2206b79dd8f87cde2be`. The 16 questions were selected mechanically from Task010 where the linked Task006 primary source is one of the three PDFs. No question or reference fact was edited.

## Method

`official PDF → PyMuPDF table extraction → header/value row serialization → BGE-M3 → IndexFlatIP top-20 → BGE reranker top-5`

Each row records its source, page, table and row ordinal, original cells, explicit column/value pairs, table context, deterministic ID, text and SHA-256. An experimental FAISS index and its metadata are separate from the production index. The same BGE-M3 embedder and BGE reranker instances evaluated production and experimental candidates. Page hits require the expected source **and** page.

## Extraction strategy

PyMuPDF **1.28.2**, one generic `Page.find_tables(strategy="lines_strict")` strategy for all three PDFs. A preliminary check of line-based, text-based, and mixed detection showed that `text` split these tables into excessive rows/columns. `lines_strict` recovered the target admission rows with five columns, the target exam row with eight, and the tuition row with ten. Extraction code has no branch for a particular program, question, number or PDF. Evaluation probes contain the expected program identities and values separately.

The serializer preserves column order and the library's header text. Sparse initial rows continuing a merged header are attached to the corresponding columns. Missing or duplicate headers retain their original text and receive neutral column identifiers. The nearest text block above a table on the same page supplies bounded table context. In the tuition table, the extracted header «Стоимость обучения за семестр, руб.» and the continuation «для граждан РФ» are included with the 179 500-ruble cell.

## Source-version verification

All fetched bytes matched the SHA-256 stored in the existing normalized artifacts **before** any experimental output was written.

| Source ID | Expected and fetched SHA-256 | Match | Tables | Rows | Ambiguous headers |
|---|---|---:|---:|---:|---:|
| `admission-capacity-pdf` | `748b164c0d1a5e7c82d7fcc4e8e417334be73f062b02c0bed5245e66ae2850c6` | yes | 4 | 75 | 3 |
| `entrance-exams-list-pdf` | `656ccb00805760f0685c703587c0d5ae5f6dad9ca11ed098f72e6a2180f3d4d7` | yes | 9 | 93 | 0 |
| `tuition-order-128-pdf` | `455d11c699f84b20739df78aa830bd37ff091b3a13ad776eb11f466a5893544e` | yes | 3 | 82 | 10 |

The ambiguity counts cover **all detected tables**, including rows outside the four probes. Ambiguous headers remain marked rather than assigned an invented meaning.

## Mandatory probes

| ID | Previous problem | Table-aware evidence | Mapping preserved | Expected row dense → reranked rank |
|---|---|---|---:|---:|
| `gen-008` | 686 general paid places confused with 105 separate foreign places | Менеджмент row: budget 36; general paid 686; separate paid 105 under distinct extracted headers | yes | 1 → 1 |
| `gen-011` | 8 separate foreign places confused with 20 general paid places | Торговое дело row: budget 3; general paid 20; separate paid 8 under distinct headers | yes | 1 → 1 |
| `gen-021` | Budget threshold 60 confused with paid threshold 40 | Экономика / Математика row: paid 40 and budget 60 under distinct headers | yes | 3 → 2 |
| `gen-027` | Semester heading absent from price chunks | Очная Экономика row: 179 500 under an extracted «за семестр» header and «для граждан РФ» subheader | yes | 2 → 2 |

All four expected rows reached experimental top-5. These probes test representation and retrieval only. They do not show that an LLM would produce a correct answer from these rows.

## 16-question retrieval

| Page Recall | Production reranked | Experimental table-aware |
|---|---:|---:|
| @1 | 6/16 (37.50%) | 11/16 (68.75%) |
| @3 | 9/16 (56.25%) | 13/16 (81.25%) |
| @5 | 13/16 (81.25%) | 13/16 (81.25%) |

Per-question page ranks and both top-5 lists are stored in the ignored experiment JSON. A page hit does not prove that the correct semantic row was retrieved; the four row-level probes above provide a separate check.

## Regressions

- Improved expected-page rank: `gen-007`, `gen-019`, `gen-020`, `gen-023`, `gen-027`, `gen-028`.
- Worsened: `gen-012`, `gen-022` (production page rank 4; experimental miss in top-5).
- Unchanged: `gen-008`, `gen-009`, `gen-010`, `gen-011`, `gen-021`, `gen-025`, `gen-029`, `gen-030`.

The experiment indexes only table rows from three PDFs, whereas production searches the complete corpus. This changes the candidate pool; the Page Recall comparison is a feasibility measure, not an isolated causal estimate for representation alone. Thirteen extracted column headers remain ambiguous elsewhere in the three PDFs. No answer-generation comparison was performed.

## Verdict

**`promising`** under the preset rule: three SHA checks passed; one generic strategy worked; all four structural probes passed; their rows survived into top-5; experimental Page Recall@5 equaled the production baseline. The two regressions and unchanged @5 mean a production migration is **not** established by this experiment.

## Engineering implication

The evidence supports a later, separately evaluated production integration experiment with explicit table structure. That task should investigate the two page-rank regressions and ambiguous headers before changing the live RAG pipeline. Task014 changed no production behavior.
