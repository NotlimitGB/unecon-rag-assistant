"""Offline metrics, single-call runner, and human review for generation."""

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Protocol

from app.generation.models import AnswerResponse
from app.retrieval.service import RetrievalResponse

SCORES = ("factual_correctness", "faithfulness", "completeness", "citation_support")


class Answerer(Protocol):
    def answer(self, question: str) -> AnswerResponse: ...


class RecordingRetrieval:
    """Capture the canonical result used by AnswerService, without searching again."""

    def __init__(self, service: Any):
        self.service = service
        self.last: RetrievalResponse | None = None

    def retrieve(self, question: str) -> RetrievalResponse:
        self.last = self.service.retrieve(question)
        return self.last


def latency_metrics(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": ordered[math.ceil(0.95 * len(values)) - 1],
        "min": ordered[0],
        "max": ordered[-1],
    }


def automatic_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    supported = [r for r in rows if r["expected_status"] == "answered"]
    unsupported = [r for r in rows if r["expected_status"] == "insufficient_evidence"]
    successful = [r for r in rows if r["success"]]
    answered = [r for r in supported if r["success"] and r["actual_status"] == "answered"]
    page_rows = [r for r in answered if r["expected_pages"] is not None]
    confusion = {
        "answered_to_answered": 0,
        "answered_to_insufficient_evidence": 0,
        "insufficient_evidence_to_answered": 0,
        "insufficient_evidence_to_insufficient_evidence": 0,
    }
    for row in successful:
        confusion[f"{row['expected_status']}_to_{row['actual_status']}"] += 1

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "reliability": {
            "total_questions": total,
            "successful_structured_responses": len(successful),
            "generation_errors": total - len(successful),
            "structured_success_rate": rate(len(successful), total),
        },
        "status": {
            "overall_status_accuracy": rate(sum(r["status_correct"] is True for r in rows), total),
            "supported_answer_rate": rate(len(answered), len(supported)),
            "unsupported_refusal_rate": rate(
                sum(
                    r["success"] and r["actual_status"] == "insufficient_evidence"
                    for r in unsupported
                ),
                len(unsupported),
            ),
            "denominators": {
                "overall": total,
                "supported": len(supported),
                "unsupported": len(unsupported),
            },
            "confusion": confusion,
        },
        "citations": {
            "answered_supported_count": len(answered),
            "page_labeled_answered_supported_count": len(page_rows),
            "expected_source_hit_rate": rate(
                sum(r["acceptable_source_hit"] is True for r in answered), len(answered)
            ),
            "primary_source_hit_rate": rate(
                sum(r["primary_source_hit"] is True for r in answered), len(answered)
            ),
            "expected_page_hit_rate": rate(
                sum(r["expected_page_hit"] is True for r in page_rows), len(page_rows)
            ),
            "answered_with_citation": sum(
                r["success"] and r["actual_status"] == "answered" and bool(r["citations"])
                for r in rows
            ),
            "insufficient_without_citation": sum(
                r["success"]
                and r["actual_status"] == "insufficient_evidence"
                and not r["citations"]
                for r in rows
            ),
            "invalid_citation_mapping_errors": sum(
                r["error"] == "invalid_citation_mapping" for r in rows
            ),
        },
        "latency_seconds": latency_metrics([r["elapsed_seconds"] for r in rows]),
    }


def _error_code(exc: Exception) -> str:
    message = str(exc).lower()
    if any(
        fragment in message
        for fragment in ("cited context", "duplicate context", "citations", "citation")
    ):
        return "invalid_citation_mapping"
    if "retrieval" in message or "index" in message:
        return "retrieval_error"
    if "timed out" in message:
        return "timeout"
    if "connect" in message or "ollama" in message:
        return "ollama_error"
    return "generation_error"


def evaluate_generation(
    dataset: dict[str, Any],
    retrieval_gold: dict[str, Any],
    answerer: Answerer,
    recording: RecordingRetrieval,
    config: dict[str, Any],
    clock=time.perf_counter,
) -> dict[str, Any]:
    gold = {q["question_id"]: q for q in retrieval_gold["questions"]}
    rows: list[dict[str, Any]] = []
    for item in dataset["questions"]:
        linked = gold[item["retrieval_question_id"]] if item["retrieval_question_id"] else None
        row = {
            "question_id": item["question_id"],
            "question": item["question"],
            "expected_status": item["expected_status"],
            "actual_status": None,
            "success": False,
            "error": None,
            "category": linked["category"] if linked else None,
            "difficulty": linked["difficulty"] if linked else None,
            "retrieval_question_id": item["retrieval_question_id"],
            "primary_source_id": linked["primary_source_id"] if linked else None,
            "acceptable_source_ids": linked["acceptable_source_ids"] if linked else [],
            "expected_pages": linked["expected_pages"] if linked else None,
            "reference_facts": item["reference_facts"],
            "manual_review_note": item["manual_review_note"],
            "answer": None,
            "retrieval_mode": None,
            "citations": [],
            "cited_contexts": [],
            "status_correct": False,
            "primary_source_hit": None,
            "acceptable_source_hit": None,
            "expected_page_hit": None,
            "elapsed_seconds": 0.0,
        }
        recording.last = None
        start = clock()
        try:
            response = answerer.answer(item["question"])
            row["success"] = True
            row["actual_status"] = response.status
            row["answer"] = response.answer
            row["retrieval_mode"] = response.retrieval_mode
            row["citations"] = [c.model_dump() for c in response.citations]
            row["status_correct"] = response.status == item["expected_status"]
            if linked and response.status == "answered":
                row["primary_source_hit"] = any(
                    c.source_id == linked["primary_source_id"] for c in response.citations
                )
                row["acceptable_source_hit"] = any(
                    c.source_id in linked["acceptable_source_ids"] for c in response.citations
                )
                if linked["expected_pages"] is not None:
                    row["expected_page_hit"] = any(
                        c.source_id == linked["primary_source_id"]
                        and c.page in linked["expected_pages"]
                        for c in response.citations
                    )
            if response.status == "answered":
                if recording.last is None:
                    raise ValueError("missing captured retrieval response")
                chunks = {f"C{i}": chunk for i, chunk in enumerate(recording.last.results, 1)}
                row["cited_contexts"] = [
                    {"context_id": c.context_id, "text": chunks[c.context_id].text}
                    for c in response.citations
                ]
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            row.update(
                success=False,
                error=_error_code(exc),
                actual_status=None,
                answer=None,
                retrieval_mode=None,
                citations=[],
                cited_contexts=[],
                status_correct=False,
                primary_source_hit=None,
                acceptable_source_hit=None,
                expected_page_hit=None,
            )
        finally:
            row["elapsed_seconds"] = max(0.0, clock() - start)
        rows.append(row)
    return {
        "schema_version": 1,
        "dataset_id": dataset["dataset_id"],
        "configuration": config,
        "metrics": automatic_metrics(rows),
        "questions": rows,
    }


def render_generation_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = ["# Оценка генерации", "", "## Конфигурация", ""]
    lines.extend(f"- {key}: {value}" for key, value in report["configuration"].items())
    lines.extend(
        [
            f"- dataset_id: {report['dataset_id']}",
            f"- total_questions: {len(report['questions'])}",
            "",
        ]
    )
    for heading, key in (
        ("Надёжность", "reliability"),
        ("Статус", "status"),
        ("Цитаты", "citations"),
        ("Время, секунды", "latency_seconds"),
    ):
        lines.extend([f"## {heading}", ""])
        for name, value in metrics[key].items():
            lines.append(f"- {name}: {value}")
        lines.append("")
    groups = (
        ("Ошибки генерации", lambda r: not r["success"]),
        (
            "Неверные отказы",
            lambda r: (
                r["expected_status"] == "answered" and r["actual_status"] == "insufficient_evidence"
            ),
        ),
        (
            "Неверные ответы",
            lambda r: (
                r["expected_status"] == "insufficient_evidence" and r["actual_status"] == "answered"
            ),
        ),
        ("Без ожидаемого источника", lambda r: r["acceptable_source_hit"] is False),
        ("Без ожидаемой страницы", lambda r: r["expected_page_hit"] is False),
    )
    lines.extend(["## Промахи", ""])
    for heading, predicate in groups:
        selected = [r for r in report["questions"] if predicate(r)]
        lines.append(f"### {heading} ({len(selected)})")
        lines.append("")
        lines.extend(f"- {r['question_id']}: {r['question']}" for r in selected)
        if not selected:
            lines.append("Нет.")
        lines.append("")
    return "\n".join(lines)


def draft_manual_review(report: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in report["questions"]:
        if (
            row["expected_status"] == "answered"
            and row["success"]
            and row["actual_status"] == "answered"
        ):
            rows.append(
                {
                    key: row[key]
                    for key in (
                        "question_id",
                        "question",
                        "reference_facts",
                        "manual_review_note",
                        "answer",
                        "citations",
                        "cited_contexts",
                    )
                }
                | {score: None for score in SCORES}
                | {"review_note": ""}
            )
    return {"schema_version": 1, "dataset_id": report["dataset_id"], "reviews": rows}


def validate_manual_review(
    review: dict[str, Any], report: dict[str, Any], partial: bool = False
) -> list[dict[str, Any]]:
    expected = draft_manual_review(report)
    if (
        not isinstance(review, dict)
        or set(review) != set(expected)
        or (
            review["schema_version"] != 1
            or review["dataset_id"] != expected["dataset_id"]
            or not isinstance(review["reviews"], list)
            or len(review["reviews"]) != len(expected["reviews"])
        )
    ):
        raise ValueError("manual review does not match current report")
    scored = []
    for actual, original in zip(review["reviews"], expected["reviews"], strict=True):
        if not isinstance(actual, dict) or set(actual) != set(original):
            raise ValueError("invalid manual review row")
        for key in (
            "question_id",
            "question",
            "reference_facts",
            "manual_review_note",
            "answer",
            "citations",
            "cited_contexts",
        ):
            if actual[key] != original[key]:
                raise ValueError(f"manual review changed source data: {original['question_id']}")
        if not isinstance(actual["review_note"], str):
            raise ValueError("invalid review note")
        values = [actual[key] for key in SCORES]
        if all(type(value) is int and value in (0, 1, 2) for value in values):
            scored.append(actual)
        elif not partial or any(value is not None for value in values):
            raise ValueError(f"incomplete or invalid scores: {original['question_id']}")
    return scored


def summarize_manual_review(
    review: dict[str, Any], report: dict[str, Any], partial: bool = False
) -> dict[str, Any]:
    rows = validate_manual_review(review, report, partial=partial)
    if not rows:
        raise ValueError("no completed manual reviews")
    return {
        "reviewed_count": len(rows),
        "available_count": len(review["reviews"]),
        "means": {key: statistics.mean(row[key] for row in rows) for key in SCORES},
        "score_2_percent": {
            key: 100 * sum(row[key] == 2 for row in rows) / len(rows)
            for key in ("factual_correctness", "faithfulness", "citation_support")
        },
        "any_zero_count": sum(any(row[key] == 0 for key in SCORES) for row in rows),
        "any_zero_ids": [
            row["question_id"] for row in rows if any(row[key] == 0 for key in SCORES)
        ],
    }


def write_generation_reports(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "generation_report.json",
        output_dir / "generation_report.md",
        output_dir / "generation_manual_review.json",
    )
    paths[0].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths[1].write_text(render_generation_markdown(report), encoding="utf-8")
    paths[2].write_text(
        json.dumps(draft_manual_review(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths
