"""Offline, evidence-locked human review of the saved Task018 answers."""

import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT
from app.evaluation.dataset import load_dataset
from app.evaluation.generation_dataset import load_generation_dataset
from app.evaluation.generation_runner import SCORES
from app.ingestion.manifest import load_manifest

REVIEW_ID = "pdf-page-diversity-human-review-v1"
EXPERIMENT_ID = "pdf-page-diversity-v1"
QUESTION_IDS = (
    "gen-007",
    "gen-008",
    "gen-009",
    "gen-010",
    "gen-011",
    "gen-012",
    "gen-019",
    "gen-020",
    "gen-021",
    "gen-022",
    "gen-023",
    "gen-025",
    "gen-027",
    "gen-028",
    "gen-029",
    "gen-030",
)
PROBE_IDS = {"gen-008", "gen-011", "gen-021", "gen-027"}
DATASET_HASHES = {
    "generation": "11a284b4279f627ae787d0ec4777e0884733b4ec937d6658be3857c2dd15d0a4",
    "retrieval": "52de939e1ba13d1558c3e96fa6cec2cabdb69fca9158996aa1a4282d4167511e",
}
OUTPUT_DIR = PROJECT_ROOT / "data/processed/evaluation"
RUBRIC = {
    "factual_correctness": {
        "2": (
            "Все существенные факты согласуются с эталоном и официальными контекстами; "
            "нет ошибочных чисел, дат, условий или оговорок."
        ),
        "1": (
            "Суть верна, но есть незначительная неточность, неполная оговорка "
            "или иной нефатальный дефект."
        ),
        "0": (
            "Существенное утверждение неверно или противоречит свидетельству, "
            "либо неверен основной ответ."
        ),
    },
    "faithfulness": {
        "2": (
            "Каждое существенное фактическое утверждение прямо подтверждается "
            "предоставленными официальными контекстами."
        ),
        "1": (
            "Основные утверждения подтверждены, но второстепенное утверждение или "
            "вывод не подтверждается напрямую."
        ),
        "0": (
            "Существенное утверждение не подтверждается контекстами или противоречит им; "
            "внешние знания не заменяют свидетельство."
        ),
    },
    "completeness": {
        "2": (
            "Ответ охватывает существенные эталонные факты и важные оговорки, "
            "необходимые для вопроса."
        ),
        "1": "Основной ответ есть, но отсутствует существенная деталь или оговорка.",
        "0": "Отсутствует запрошенная основная информация или вопрос по существу не раскрыт.",
    },
    "citation_support": {
        "2": (
            "Процитированные контексты прямо подтверждают все существенные утверждения, "
            "требующие подтверждения."
        ),
        "1": (
            "Цитаты подтверждают суть, но существенная деталь подтверждена слабо, "
            "косвенно или лишь неприведённым контекстом."
        ),
        "0": (
            "Цитаты не подтверждают существенное утверждение, указывают на неверное "
            "свидетельство или отсутствуют там, где нужны."
        ),
    },
    "context_sufficiency": {
        "sufficient": (
            "Пять предоставленных контекстов достаточны для материально "
            "правильного ответа."
        ),
        "insufficient": "Пять контекстов недостаточны для материально правильного ответа.",
        "unclear": (
            "По предоставленным свидетельствам нельзя уверенно решить, достаточны ли контексты."
        ),
    },
}
TOP_KEYS = {
    "schema_version",
    "review_id",
    "source_generation_report_sha256",
    "source_retrieval_report_sha256",
    "generation_dataset_sha256",
    "retrieval_dataset_sha256",
    "reviewer_name",
    "reviewed_at",
    "reviewer_note",
    "scoring_rubric",
    "cases",
}
EDITABLE_TOP = {"reviewer_name", "reviewed_at", "reviewer_note"}
EDITABLE_CASE = {
    "context_sufficiency",
    "factual_correctness",
    "faithfulness",
    "completeness",
    "citation_support",
    "needs_gold_review",
    "review_note",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return raw


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _same_json(left: Any, right: Any) -> bool:
    """Compare immutable JSON evidence without equating booleans and integers."""
    return json.dumps(left, ensure_ascii=False, sort_keys=True) == json.dumps(
        right, ensure_ascii=False, sort_keys=True
    )


def expected_packet(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Reconstruct every immutable field from frozen local files only."""
    dataset_dir = root / "data/evaluation"
    report_dir = root / "data/processed/evaluation"
    generation_path = dataset_dir / "generation_questions.json"
    retrieval_path = dataset_dir / "retrieval_questions.json"
    for label, path in (("generation", generation_path), ("retrieval", retrieval_path)):
        _require(_sha(path) == DATASET_HASHES[label], f"frozen {label} dataset hash changed")
    manifest = load_manifest(root / "data/source_manifest.json")
    retrieval_dataset = load_dataset(retrieval_path, manifest)
    generation_dataset = load_generation_dataset(generation_path, retrieval_dataset)
    sources = {source.id: source for source in manifest.sources}
    gold = {item["question_id"]: item for item in retrieval_dataset["questions"]}
    generation_gold = {item["question_id"]: item for item in generation_dataset["questions"]}

    generation_path = report_dir / "pdf_page_diversity_generation_experiment.json"
    retrieval_path = report_dir / "pdf_page_diversity_retrieval_experiment.json"
    generation_sha, retrieval_sha = _sha(generation_path), _sha(retrieval_path)
    generation, retrieval = _load(generation_path), _load(retrieval_path)
    _require(
        generation.get("schema_version") == 1
        and generation.get("experiment_id") == EXPERIMENT_ID
        and generation.get("verdict") == "diversity_integration_promising"
        and generation.get("generation_question_ids") == list(QUESTION_IDS),
        "Task018 generation report identity changed",
    )
    rows = generation.get("questions")
    _require(
        isinstance(rows, list)
        and len(rows) == 16
        and [row.get("question_id") for row in rows if isinstance(row, dict)] == list(QUESTION_IDS),
        "Task018 generation question order changed",
    )
    _require(
        generation.get("status_movement", {}).get("error") == 0
        and generation["status_movement"].get("new_refusal") == 0,
        "Task018 generation errors or new refusals",
    )
    probes = generation.get("critical_probes")
    _require(
        isinstance(probes, list)
        and {probe.get("question_id") for probe in probes if isinstance(probe, dict)} == PROBE_IDS
        and len(probes) == 4,
        "Task018 mandatory probes changed",
    )
    _require(
        retrieval.get("schema_version") == 1
        and retrieval.get("experiment_id") == EXPERIMENT_ID
        and retrieval.get("phase_a_verdict") == "diversity_retrieval_gate_passed"
        and retrieval.get("corpus_counts")
        == {"production_chunks": 338, "table_rows": 250, "questions": 80}
        and isinstance(retrieval.get("gate"), dict)
        and len(retrieval["gate"]) == 17
        and all(value is True for value in retrieval["gate"].values()),
        "Task018 retrieval gate changed",
    )
    selections = retrieval.get("selection_diagnostics")
    _require(
        isinstance(selections, list)
        and len(selections) == 80
        and [row.get("question_id") for row in selections if isinstance(row, dict)]
        == [item["question_id"] for item in retrieval_dataset["questions"]],
        "Task018 retrieval question set changed",
    )
    selection_by_id = {row["question_id"]: row for row in selections}
    cases = []
    statuses = []
    for row in rows:
        question_id = row["question_id"]
        result = row.get("augmented")
        _require(isinstance(result, dict), f"missing generation result: {question_id}")
        item = generation_gold[question_id]
        retrieval_id = item["retrieval_question_id"]
        label = gold[retrieval_id]
        _require(
            result.get("success") is True
            and result.get("error") is None
            and result.get("question_id") == question_id
            and result.get("question") == item["question"]
            and result.get("retrieval_question_id") == retrieval_id
            and result.get("expected_status") == item["expected_status"]
            and result.get("reference_facts") == item["reference_facts"]
            and result.get("manual_review_note") == item["manual_review_note"]
            and result.get("category") == label["category"]
            and result.get("difficulty") == label["difficulty"]
            and result.get("primary_source_id") == label["primary_source_id"]
            and result.get("acceptable_source_ids") == label["acceptable_source_ids"]
            and result.get("expected_pages") == label["expected_pages"]
            and result.get("actual_status") in {"answered", "insufficient_evidence"}
            and isinstance(result.get("answer"), str),
            f"Task018 generation evidence changed: {question_id}",
        )
        statuses.append(result["actual_status"])
        saved_contexts = row.get("diversity_contexts")
        top5 = selection_by_id[retrieval_id].get("diversity_top5")
        _require(
            isinstance(saved_contexts, list)
            and isinstance(top5, list)
            and len(saved_contexts) == len(top5) == 5,
            f"missing five Task018 contexts: {question_id}",
        )
        citations = result.get("citations")
        cited_contexts = result.get("cited_contexts")
        _require(
            isinstance(citations, list) and isinstance(cited_contexts, list),
            f"invalid citations: {question_id}",
        )
        context_map = {}
        contexts = []
        for position, (saved, ranked) in enumerate(zip(saved_contexts, top5, strict=True), 1):
            _require(
                isinstance(saved, dict)
                and isinstance(ranked, dict)
                and set(saved) == {"context_id", "chunk_id", "representation", "text"}
                and saved["context_id"] == f"C{position}"
                and saved["chunk_id"] == ranked.get("chunk_id")
                and saved["representation"] == ranked.get("representation")
                and saved["representation"] in {"production_chunk", "table_row"}
                and isinstance(saved["text"], str)
                and bool(saved["text"].strip())
                and ranked.get("source_id") in sources,
                f"Task018 context/retrieval mismatch: {question_id}/C{position}",
            )
            source = sources[ranked["source_id"]]
            page = ranked.get("page")
            _require(
                (source.source_type == "html" and page is None)
                or (source.source_type == "pdf" and type(page) is int and page > 0),
                f"Task018 page provenance mismatch: {question_id}/C{position}",
            )
            context = {
                "context_id": saved["context_id"],
                "chunk_id": saved["chunk_id"],
                "representation": saved["representation"],
                "source_id": source.id,
                "source_title": source.title,
                "page": page,
                "text": saved["text"],
                "cited": False,
            }
            contexts.append(context)
            context_map[context["context_id"]] = context
        seen_citations = set()
        for citation in citations:
            _require(isinstance(citation, dict), f"invalid citation: {question_id}")
            context_id = citation.get("context_id")
            context = context_map.get(context_id)
            _require(
                context is not None
                and context_id not in seen_citations
                and set(citation)
                == {
                    "context_id",
                    "chunk_id",
                    "source_id",
                    "source_title",
                    "source_url",
                    "source_type",
                    "page",
                }
                and citation.get("chunk_id") == context["chunk_id"]
                and citation.get("source_id") == context["source_id"]
                and citation.get("source_title") == context["source_title"]
                and citation.get("source_url") == sources[context["source_id"]].url
                and citation.get("source_type") == sources[context["source_id"]].source_type
                and citation.get("page") == context["page"],
                f"citation-to-context mismatch: {question_id}/{context_id}",
            )
            seen_citations.add(context_id)
            context["cited"] = True
        _require(
            cited_contexts
            == [
                {
                    "context_id": citation["context_id"],
                    "text": context_map[citation["context_id"]]["text"],
                }
                for citation in citations
            ],
            f"cited context text mismatch: {question_id}",
        )
        _require(
            bool(citations) if result["actual_status"] == "answered" else not citations,
            f"citation/status mismatch: {question_id}",
        )
        cases.append(
            {
                "question_id": question_id,
                "retrieval_question_id": retrieval_id,
                "question": item["question"],
                "category": label["category"],
                "difficulty": label["difficulty"],
                "expected_status": item["expected_status"],
                "reference_facts": item["reference_facts"],
                "manual_review_note": item["manual_review_note"],
                "primary_source_id": label["primary_source_id"],
                "acceptable_source_ids": label["acceptable_source_ids"],
                "expected_pages": label["expected_pages"],
                "actual_status": result["actual_status"],
                "answer": result["answer"],
                "citations": citations,
                "retrieved_contexts": contexts,
                "answer_scoring_applicable": result["actual_status"] == "answered",
                "context_sufficiency": None,
                **{score: None for score in SCORES},
                "needs_gold_review": False,
                "review_note": "",
            }
        )
    _require(
        statuses.count("answered") == 15
        and statuses.count("insufficient_evidence") == 1
        and cases[5]["question_id"] == "gen-012"
        and cases[5]["actual_status"] == "insufficient_evidence",
        "Task018 answer/refusal counts changed",
    )
    return {
        "schema_version": 1,
        "review_id": REVIEW_ID,
        "source_generation_report_sha256": generation_sha,
        "source_retrieval_report_sha256": retrieval_sha,
        "generation_dataset_sha256": DATASET_HASHES["generation"],
        "retrieval_dataset_sha256": DATASET_HASHES["retrieval"],
        "reviewer_name": "",
        "reviewed_at": "",
        "reviewer_note": "",
        "scoring_rubric": RUBRIC,
        "cases": cases,
    }


def validate_review(review: Any, expected: dict[str, Any], partial: bool = False) -> dict[str, Any]:
    _require(isinstance(review, dict) and set(review) == TOP_KEYS, "invalid review schema")
    for key in TOP_KEYS - EDITABLE_TOP - {"cases"}:
        _require(_same_json(review[key], expected[key]), f"immutable review field changed: {key}")
    for key in EDITABLE_TOP:
        _require(isinstance(review[key], str), f"invalid reviewer field: {key}")
    if not partial:
        _require(
            review["reviewer_name"].strip() and review["reviewed_at"].strip(),
            "reviewer name and review date are required",
        )
    _require(
        isinstance(review["cases"], list) and len(review["cases"]) == 16,
        "review must contain 16 cases",
    )
    completed_answer_ids = []
    fully_reviewed = partially_reviewed = untouched = 0
    for case, source in zip(review["cases"], expected["cases"], strict=True):
        qid = source["question_id"]
        _require(isinstance(case, dict) and set(case) == set(source), f"invalid case schema: {qid}")
        for key in set(source) - EDITABLE_CASE:
            _require(
                _same_json(case[key], source[key]),
                f"immutable case evidence changed: {qid}/{key}",
            )
        sufficiency = case["context_sufficiency"]
        _require(
            sufficiency is None
            or (isinstance(sufficiency, str) and sufficiency in RUBRIC["context_sufficiency"]),
            f"invalid context sufficiency: {qid}",
        )
        _require(
            type(case["needs_gold_review"]) is bool and isinstance(case["review_note"], str),
            f"invalid human review field: {qid}",
        )
        for score in SCORES:
            value = case[score]
            _require(
                value is None or (type(value) is int and value in (0, 1, 2)),
                f"invalid {score}: {qid}",
            )
            if not source["answer_scoring_applicable"]:
                _require(value is None, f"refusal cannot have answer scores: {qid}")
        complete = sufficiency is not None and (
            not source["answer_scoring_applicable"]
            or all(case[score] is not None for score in SCORES)
        )
        blank = (
            sufficiency is None
            and all(case[score] is None for score in SCORES)
            and case["needs_gold_review"] is False
            and case["review_note"] == ""
        )
        if complete:
            fully_reviewed += 1
            if source["answer_scoring_applicable"]:
                completed_answer_ids.append(qid)
        elif blank:
            untouched += 1
        else:
            partially_reviewed += 1
        if not partial:
            _require(complete, f"incomplete human review: {qid}")
    return {
        "fully_reviewed": fully_reviewed,
        "partially_reviewed": partially_reviewed,
        "untouched": untouched,
        "completed_answer_ids": completed_answer_ids,
    }


def summarize_review(
    review: dict[str, Any], expected: dict[str, Any], partial: bool = False
) -> dict[str, Any]:
    progress = validate_review(review, expected, partial=partial)
    completed_ids = set(progress["completed_answer_ids"])
    scored = [case for case in review["cases"] if case["question_id"] in completed_ids]
    metric_summary = {}
    for score in SCORES:
        values = [case[score] for case in scored]
        metric_summary[score] = {
            "mean": statistics.mean(values) if values else None,
            "score_2_count": values.count(2),
            "score_2_percent": 100 * values.count(2) / len(values) if values else None,
            "score_1_count": values.count(1),
            "score_0_count": values.count(0),
            "score_0_question_ids": [case["question_id"] for case in scored if case[score] == 0],
        }
    sufficiency = {
        label: sum(case["context_sufficiency"] == label for case in review["cases"])
        for label in RUBRIC["context_sufficiency"]
    }
    false_refusals = [
        case
        for case in review["cases"]
        if case["expected_status"] == "answered"
        and case["actual_status"] == "insufficient_evidence"
    ]
    return {
        "review_id": REVIEW_ID,
        "progress": {
            key: value for key, value in progress.items() if key != "completed_answer_ids"
        },
        "reviewed_answer_count": len(scored),
        "available_answer_count": 15,
        "metrics": metric_summary,
        "fully_clean_answer_count": sum(
            all(case[score] == 2 for score in SCORES) for case in scored
        ),
        "fully_clean_answer_ids": [
            case["question_id"] for case in scored if all(case[score] == 2 for score in SCORES)
        ],
        "any_zero_answer_ids": [
            case["question_id"] for case in scored if any(case[score] == 0 for score in SCORES)
        ],
        "context_sufficiency": {
            **sufficiency,
            "insufficient_ids": [
                case["question_id"]
                for case in review["cases"]
                if case["context_sufficiency"] == "insufficient"
            ],
            "unclear_ids": [
                case["question_id"]
                for case in review["cases"]
                if case["context_sufficiency"] == "unclear"
            ],
        },
        "refusals": {
            "expected_answer_cases": sum(
                case["expected_status"] == "answered" for case in review["cases"]
            ),
            "actual_answered_cases": sum(
                case["actual_status"] == "answered" for case in review["cases"]
            ),
            "false_refusal_cases": len(false_refusals),
            "false_refusal_ids": [case["question_id"] for case in false_refusals],
            "false_refusal_context_sufficiency": {
                case["question_id"]: case["context_sufficiency"] for case in false_refusals
            },
        },
        "needs_gold_review_count": sum(case["needs_gold_review"] for case in review["cases"]),
        "needs_gold_review_ids": [
            case["question_id"] for case in review["cases"] if case["needs_gold_review"]
        ],
    }


def _fence(value: str) -> str:
    longest = max((len(part) for part in re.findall(r"`+", value)), default=0)
    return "`" * max(3, longest + 1)


def _block(value: str) -> str:
    fence = _fence(value)
    return f"{fence}\n{value}\n{fence}"


def render_worksheet(packet: dict[str, Any]) -> str:
    lines = [
        "# Ручная оценка ответов Task018",
        "",
        (
            "Оценщик использует только эталонные факты, примечание датасета, точный "
            "ответ, пять официальных контекстов и цитаты. Веб-поиск и внешние знания "
            "не заменяют эти свидетельства. Ни одна оценка не заполнена заранее."
        ),
        "",
        "## Рубрика",
        "",
    ]
    labels = {
        "factual_correctness": "Фактическая правильность",
        "faithfulness": "Опора на контекст",
        "completeness": "Полнота",
        "citation_support": "Поддержка цитатами",
    }
    for score, title in labels.items():
        lines.extend([f"### {title}", ""])
        lines.extend(f"- {value}: {RUBRIC[score][str(value)]}" for value in (2, 1, 0))
        lines.append("")
    lines.extend(["### Достаточность контекста", ""])
    lines.extend(f"- {key}: {value}" for key, value in RUBRIC["context_sufficiency"].items())
    lines.extend(
        [
            "",
            (
                "Если эталон конфликтует с предоставленным официальным свидетельством, "
                "отметьте needs_gold_review и объясните в заметке. Это не меняет "
                "замороженный датасет."
            ),
            "",
        ]
    )
    for case in packet["cases"]:
        lines.extend(
            [
                f"## {case['question_id']} — {case['question']}",
                "",
                f"- Категория: {case['category']}",
                f"- Сложность: {case['difficulty']}",
                f"- Ожидаемый статус: {case['expected_status']}",
                f"- Фактический статус: {case['actual_status']}",
                f"- Основной источник: {case['primary_source_id']}",
                f"- Допустимые источники: {', '.join(case['acceptable_source_ids'])}",
                f"- Ожидаемые страницы: {case['expected_pages']}",
                "",
                "### Эталонные факты",
                "",
            ]
        )
        for fact in case["reference_facts"]:
            lines.extend([_block(fact), ""])
        lines.extend(["### Примечание из датасета", "", _block(case["manual_review_note"]), ""])
        lines.extend(["### Ответ Task018", "", _block(case["answer"]), ""])
        lines.extend(["### Цитаты", ""])
        if case["citations"]:
            for citation in case["citations"]:
                lines.append(_block(json.dumps(citation, ensure_ascii=False, indent=2)))
                lines.append("")
        else:
            lines.extend(["Цитат нет.", ""])
        for context in case["retrieved_contexts"]:
            lines.extend(
                [
                    f"### Контекст {context['context_id']}",
                    "",
                    f"- Источник: {context['source_title']} (`{context['source_id']}`)",
                    f"- Страница: {context['page']}",
                    f"- Представление: {context['representation']}",
                    f"- Процитирован: {'да' if context['cited'] else 'нет'}",
                    f"- ID: `{context['chunk_id']}`",
                    "",
                    _block(context["text"]),
                    "",
                ]
            )
        lines.extend(["### Оценка человека", ""])
        if not case["answer_scoring_applicable"]:
            lines.extend(
                [
                    "Semantic answer scores are not applicable because no substantive "
                    "answer was produced.",
                    "",
                ]
            )
        lines.extend(
            [
                "- Достаточность контекста:",
                "- Фактическая правильность:",
                "- Опора на контекст:",
                "- Полнота:",
                "- Поддержка цитатами:",
                "- Нужна проверка эталона:",
                "- Примечание оценщика:",
                "",
            ]
        )
    return "\n".join(lines)
