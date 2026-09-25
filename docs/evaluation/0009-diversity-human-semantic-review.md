# Diversity Human Semantic Review

## Review basis

Human reviewer examined all 16 frozen Task018 cases, including each exact generated answer or refusal, reference facts, five contexts supplied to generation, and answer citations. The recorded values are the explicitly confirmed human judgments. No LLM judge, Task012 diagnostic scores, or automatic conversion of retrieval, citation, or numeric-token checks supplied semantic scores. The detailed review and its mechanically derived summary remain local, ignored artifacts.

## Method and coverage

The Task019 rubric scores substantive answers from 0 to 2 on factual correctness, faithfulness to the supplied official contexts, completeness, and citation support. Context sufficiency is a separate human judgment: `sufficient`, `insufficient`, or `unclear`. A refusal receives a context-sufficiency judgment but no answer-quality scores.

Full validation passed for **16/16** cases. There were **15 substantive answers scored** and one refusal, `gen-012`. The human marked 15 contexts sufficient, one insufficient (`gen-012`), and none unclear. All 15 scored answers have four integer scores; the refusal retains four `null` scores.

## Aggregate human results

| Dimension | Mean / 2 | Score 2 | Score 1 | Score 0 |
|---|---:|---:|---:|---:|
| Factual correctness | 26/15 = **1.7333** | 11 (73.33%) | 4 | 0 |
| Faithfulness | 25/15 = **1.6667** | 10 (66.67%) | 5 | 0 |
| Completeness | 28/15 = **1.8667** | 13 (86.67%) | 2 | 0 |
| Citation support | 25/15 = **1.6667** | 10 (66.67%) | 5 | 0 |

**Nine of 15 answers (60%)** received 2 on all four dimensions: `gen-007`, `gen-009`, `gen-010`, `gen-011`, `gen-019`, `gen-020`, `gen-021`, `gen-023`, `gen-027`. No scored answer received a 0 on any dimension; the remaining answers had partial issues, not a recorded total failure of the core answer.

## Cases requiring attention

| Case | Human-confirmed observation |
|---|---|
| `gen-008` | The answer's 791 adds 686 paid places in the general competition and 105 in the separate competition; the frozen reference expects 686. The reviewer marked `needs_gold_review = true` to clarify the meaning of “платных мест всего”. The reference was not changed. |
| `gen-012` | The expected status is `answered`, while the model returned `insufficient_evidence`. None of its five supplied contexts contains the required aggregate value 276. The reviewer judged the **supplied context insufficient**; this does not erase the status mismatch against the frozen dataset. No answer-quality scores were assigned. |
| `gen-022` | The answer presents an interview as a general applicant choice, while the cited context limits it to a specified group and subjects. It also states the distance format less definitely than the official context. |
| `gen-025` | The cited evidence supports the order number and date, but does not visibly support the full order title quoted in the answer. Additional cited master's tuition rows offer little support for the main answer. |
| `gen-028` | The price is present, but the cited fragment lacks the “за семестр” column heading, so the payment period is supported less clearly than the amount. |
| `gen-029` | The official row says “лица, не имеющие гражданства РФ”; the answer narrows this to “иностранных граждан”. The primary price of 210,000 rubles is supported. |
| `gen-030` | The price is present, but the cited fragment lacks the “за семестр” column heading, leaving the period less directly supported than the amount. |

The only gold-review flag is `gen-008`. The only refusal is `gen-012`; its four semantic answer scores remain absent. These two cases require different follow-up: gold wording and retrieval-context sufficiency, respectively.

## Interpretation and engineering implication

Task018 now has both frozen retrieval evidence and a completed human semantic review of its 16 controlled generation cases. The results support proceeding to a **separate production-adoption task for the tested retrieval architecture**, with the `gen-008` gold ambiguity, the `gen-012` context gap, and the partial answer/citation issues above carried forward explicitly. This review does not prove the entire system factually correct and does not switch production retrieval or generation.
