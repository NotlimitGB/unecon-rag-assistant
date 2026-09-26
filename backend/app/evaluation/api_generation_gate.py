"""Acceptance gates over the existing full-generation metrics, without semantic scoring."""

import hashlib
import json
from pathlib import Path

from app.evaluation.generation_runner import automatic_metrics
from app.generation.models import AnswerResponse

EXPECTED_CONFIG = {
    "generation_provider": "ollama",
    "model": "qwen3.5:9b",
    "retrieval_mode": "reranked",
    "retrieval_top_k": 5,
    "prompt_version": "grounded-answer-v2",
    "temperature": 0.1,
    "max_tokens": 512,
    "supported_questions": 42,
    "unsupported_questions": 18,
}


BASELINE_SHA256 = "1355a0c13a7a641c2c4f4b7a8700463d0be50e44c25f31efb12cbf674784681b"
BASELINE_PATH = (
    Path(__file__).resolve().parents[3]
    / "data/processed/evaluation/task023a/generation/generation_report.json"
)


def load_v1_baseline() -> dict:
    raw = BASELINE_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASELINE_SHA256:
        raise ValueError("Task023A v1 baseline SHA-256 mismatch")
    return json.loads(raw)


def compare_v1(report: dict, baseline: dict) -> dict:
    previous = {r["question_id"]: r for r in baseline["questions"]}
    supported = [r for r in report["questions"] if r["expected_status"] == "answered"]
    transitions = [
        {
            "question_id": r["question_id"],
            "before": previous[r["question_id"]]["actual_status"],
            "after": r["actual_status"],
        }
        for r in supported
        if r["actual_status"] != previous[r["question_id"]]["actual_status"]
    ]
    failures = [
        f"{r['question_id']}: new supported refusal"
        for r in transitions
        if r["before"] == "answered" and r["after"] == "insufficient_evidence"
    ]
    answered = [r for r in supported if r["success"] and r["actual_status"] == "answered"]
    counts = {
        "acceptable_source": sum(r["acceptable_source_hit"] is True for r in answered),
        "primary_source": sum(r["primary_source_hit"] is True for r in answered),
        "source_page": sum(r["expected_page_hit"] is True for r in answered),
    }
    for key, minimum in (("acceptable_source", 35), ("primary_source", 34), ("source_page", 23)):
        if counts[key] < minimum:
            failures.append(f"{key}: {counts[key]} below {minimum}")
    if len(answered) < 39:
        failures.append("supported answered below 39")
    refused = {r["question_id"] for r in supported if r["actual_status"] == "insufficient_evidence"}
    if not refused <= {"gen-012", "gen-030", "gen-031"}:
        failures.append("supported refusal outside accepted v1 set")
    return {
        "failures": failures,
        "transitions": transitions,
        "citation_hit_counts": counts,
        "answered_supported_count": len(answered),
        "page_labeled_answered_count": sum(r["expected_pages"] is not None for r in answered),
    }


def validate_generation(
    report: dict, generation: dict, retrieval: dict, baseline: dict | None = None
) -> dict:
    failures = []
    if report.get("configuration") != EXPECTED_CONFIG:
        failures.append("configuration mismatch")
    if report.get("dataset_id") != generation["dataset_id"] or report.get("schema_version") != 1:
        failures.append("dataset identity mismatch")
    rows = report.get("questions", [])
    expected = generation["questions"]
    ids = [q["question_id"] for q in expected]
    if len(rows) != 60 or [q.get("question_id") for q in rows] != ids:
        return {"passed": False, "failures": failures + ["expected 60 ordered unique questions"]}
    gold = {q["question_id"]: q for q in retrieval["questions"]}
    unsupported = []
    refused = []
    for row, item in zip(rows, expected, strict=True):
        identity = item["question_id"]
        linked = gold.get(item["retrieval_question_id"])
        frozen = {
            **{
                key: item[key]
                for key in (
                    "question",
                    "expected_status",
                    "retrieval_question_id",
                    "reference_facts",
                    "manual_review_note",
                )
            },
            "primary_source_id": linked["primary_source_id"] if linked else None,
            "acceptable_source_ids": linked["acceptable_source_ids"] if linked else [],
            "expected_pages": linked["expected_pages"] if linked else None,
        }
        if any(row.get(key) != value for key, value in frozen.items()):
            failures.append(f"{identity}: frozen reference mismatch")
        if row.get("success") is not True or row.get("error") is not None:
            failures.append(f"{identity}: generation error")
        try:
            answer = AnswerResponse.model_validate(
                {
                    "query": row["question"],
                    "status": row["actual_status"],
                    "answer": row["answer"],
                    "retrieval_mode": row["retrieval_mode"],
                    "citations": row["citations"],
                }
            )
            if answer.retrieval_mode != "reranked":
                raise ValueError("incorrect retrieval mode")
            context_ids = [c.context_id for c in answer.citations]
            contexts = row["cited_contexts"]
            if (
                len(set(context_ids)) != len(context_ids)
                or [c["context_id"] for c in contexts] != context_ids
                or any(not isinstance(c["text"], str) or not c["text"].strip() for c in contexts)
            ):
                raise ValueError("invalid captured citation mapping")
        except (ValueError, KeyError, TypeError) as exc:
            failures.append(
                f"{identity}: invalid response or citation mapping ({type(exc).__name__})"
            )
        status = row.get("actual_status")
        if row.get("status_correct") != (status == item["expected_status"]):
            failures.append(f"{identity}: inconsistent status metric")
        if item["expected_status"] == "insufficient_evidence":
            unsupported.append({"question_id": identity, "status": status})
            if status != "insufficient_evidence":
                failures.append(f"{identity}: mandatory unsupported refusal missing")
        elif status == "insufficient_evidence":
            refused.append(identity)
    try:
        metrics = automatic_metrics(rows)
        if metrics != report.get("metrics"):
            failures.append("stored metrics differ from existing metric implementation")
        if metrics["reliability"]["generation_errors"] != 0:
            failures.append("generation errors must be zero")
        if metrics["citations"]["invalid_citation_mapping_errors"] != 0:
            failures.append("invalid citation mappings must be zero")
    except (ValueError, KeyError, TypeError) as exc:
        metrics = None
        failures.append(f"invalid metric input: {type(exc).__name__}")
    comparison = compare_v1(report, baseline) if baseline is not None else None
    if comparison is not None:
        failures.extend(comparison["failures"])
    critical = next(r for r in rows if r["question_id"] == "gen-048")
    return {
        "passed": not failures,
        "failures": failures,
        "metrics": metrics,
        "v1_comparison": comparison,
        "gen_059_refused": next(r for r in rows if r["question_id"] == "gen-059")["actual_status"]
        == "insufficient_evidence",
        "unsupported_statuses": unsupported,
        "supported_refusals": refused,
        "gen_048_refused": critical["actual_status"] == "insufficient_evidence",
        "gold_review": {"gen-008": {"needs_gold_review": True}},
    }
