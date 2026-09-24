"""Offline tests for the fixed generation evaluation experiment."""

import copy
import json

import pytest

from app.evaluation.dataset import DatasetError
from app.evaluation.generation_cli import ROOT, load_datasets, main
from app.evaluation.generation_dataset import validate_generation_dataset
from app.evaluation.generation_runner import (
    RecordingRetrieval,
    automatic_metrics,
    draft_manual_review,
    evaluate_generation,
    latency_metrics,
    render_generation_markdown,
    summarize_manual_review,
    validate_manual_review,
    write_generation_reports,
)
from app.generation.models import AnswerCitation, AnswerResponse
from app.retrieval.service import RetrievalResponse, RetrievedChunk


@pytest.fixture(scope="module")
def datasets():
    return load_datasets(
        ROOT / "data/evaluation/generation_questions.json",
        ROOT / "data/evaluation/retrieval_questions.json",
    )


def test_production_dataset_contract(datasets):
    generation, retrieval = datasets
    assert len(generation["questions"]) == 60
    assert [item["question_id"] for item in generation["questions"]] == [
        f"gen-{i:03d}" for i in range(1, 61)
    ]
    assert sum(q["expected_status"] == "answered" for q in generation["questions"]) == 42
    assert (
        sum(q["expected_status"] == "insufficient_evidence" for q in generation["questions"]) == 18
    )
    gold = {q["question_id"]: q for q in retrieval["questions"]}
    counts = {}
    for item in generation["questions"][:42]:
        linked = gold[item["retrieval_question_id"]]
        assert item["question"] == linked["question"]
        counts.setdefault(linked["category"], {}).setdefault(linked["difficulty"], 0)
        counts[linked["category"]][linked["difficulty"]] += 1
    assert len(counts) == 7
    assert all(group == {"easy": 2, "medium": 2, "hard": 2} for group in counts.values())


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["questions"].pop(),
        lambda d: d["questions"][1].update(question=d["questions"][0]["question"]),
        lambda d: d["questions"][0].update(expected_status="other"),
        lambda d: d["questions"][0].update(retrieval_question_id="ret-999"),
        lambda d: d["questions"][0].update(question="Другой текст"),
        lambda d: d["questions"][0].update(reference_facts=[]),
        lambda d: d["questions"][42].update(retrieval_question_id="ret-001"),
        lambda d: d["questions"][42].update(reference_facts=["Несуществующий факт"]),
        lambda d: d["questions"][0].update(extra=1),
        lambda d: d.update(extra=1),
        lambda d: d["questions"][0].update(question_id="gen-060"),
    ],
)
def test_invalid_dataset_rejected(datasets, change):
    generation, retrieval = datasets
    altered = copy.deepcopy(generation)
    change(altered)
    with pytest.raises(DatasetError):
        validate_generation_dataset(altered, retrieval)


def test_frozen_retrieval_sha_guard(datasets, tmp_path):
    generation, _ = datasets
    path = tmp_path / "altered.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(DatasetError, match="SHA-256"):
        load_datasets(ROOT / "data/evaluation/generation_questions.json", path)
    assert len(generation["questions"]) == 60


def _row(expected, actual, success=True, primary=False, accepted=False, page=None):
    return {
        "expected_status": expected,
        "actual_status": actual,
        "success": success,
        "status_correct": success and expected == actual,
        "expected_pages": [1] if page is not None else None,
        "primary_source_hit": primary,
        "acceptable_source_hit": accepted,
        "expected_page_hit": page,
        "citations": [{}] if actual == "answered" else [],
        "error": None if success else "generation_error",
        "elapsed_seconds": 1.0,
    }


def test_exact_automatic_metrics_and_denominators():
    rows = [
        _row("answered", "answered", primary=True, accepted=True, page=True),
        _row("answered", "answered", primary=False, accepted=True, page=False),
        _row("answered", "insufficient_evidence"),
        _row("insufficient_evidence", "insufficient_evidence"),
        _row("insufficient_evidence", "answered"),
        _row("insufficient_evidence", None, success=False),
    ]
    metrics = automatic_metrics(rows)
    assert metrics["reliability"] == {
        "total_questions": 6,
        "successful_structured_responses": 5,
        "generation_errors": 1,
        "structured_success_rate": 5 / 6,
    }
    assert metrics["status"]["overall_status_accuracy"] == 3 / 6
    assert metrics["status"]["supported_answer_rate"] == 2 / 3
    assert metrics["status"]["unsupported_refusal_rate"] == 1 / 3
    assert metrics["status"]["confusion"] == {
        "answered_to_answered": 2,
        "answered_to_insufficient_evidence": 1,
        "insufficient_evidence_to_answered": 1,
        "insufficient_evidence_to_insufficient_evidence": 1,
    }
    assert metrics["citations"]["expected_source_hit_rate"] == 1.0
    assert metrics["citations"]["primary_source_hit_rate"] == 0.5
    assert metrics["citations"]["expected_page_hit_rate"] == 0.5
    assert metrics["citations"]["answered_supported_count"] == 2
    assert metrics["citations"]["page_labeled_answered_supported_count"] == 2


def test_latency_nearest_rank():
    values = list(range(1, 21))
    assert latency_metrics(values) == {
        "count": 20,
        "mean": 10.5,
        "median": 10.5,
        "p95": 19,
        "min": 1,
        "max": 20,
    }
    assert latency_metrics([])["p95"] is None


def _retrieved(question):
    return RetrievalResponse(
        query=question,
        mode="reranked",
        top_k=5,
        results=[
            RetrievedChunk(
                rank=1,
                chunk_id="source:0001:abcdef123456",
                text="Точный текст чанка.",
                source_id="admission-deadlines-pdf",
                source_title="Сроки приема",
                source_url="https://unecon.ru/document.pdf",
                source_type="pdf",
                category="admission_deadlines",
                admission_year=2026,
                page=1,
                score=2.0,
                dense_score=0.7,
                rerank_score=2.0,
            )
        ],
    )


def test_runner_one_call_error_isolation_and_reports(datasets, tmp_path):
    generation, retrieval = datasets
    sample = copy.deepcopy(generation)
    sample["questions"] = [
        generation["questions"][0],
        generation["questions"][1],
        generation["questions"][42],
    ]

    class Search:
        def __init__(self):
            self.calls = []

        def retrieve(self, question):
            self.calls.append(question)
            return _retrieved(question)

    search = Search()
    recording = RecordingRetrieval(search)

    class Answerer:
        def __init__(self):
            self.calls = []

        def answer(self, question):
            self.calls.append(question)
            recording.retrieve(question)
            if len(self.calls) == 2:
                raise ValueError("synthetic transport failure")
            if len(self.calls) == 3:
                return AnswerResponse(
                    query=question,
                    status="insufficient_evidence",
                    answer=(
                        "В предоставленных официальных материалах недостаточно информации, "
                        "чтобы уверенно ответить на этот вопрос. Рекомендуется проверить "
                        "актуальную информацию на официальном сайте СПбГЭУ или обратиться "
                        "в приёмную комиссию."
                    ),
                    retrieval_mode="reranked",
                    citations=[],
                )
            return AnswerResponse(
                query=question,
                status="answered",
                answer="Ответ без изменений.",
                retrieval_mode="reranked",
                citations=[
                    AnswerCitation(
                        context_id="C1",
                        chunk_id="source:0001:abcdef123456",
                        source_id="admission-deadlines-pdf",
                        source_title="Сроки приема",
                        source_url="https://unecon.ru/document.pdf",
                        source_type="pdf",
                        page=1,
                    )
                ],
            )

    answerer = Answerer()
    times = iter([0, 1, 2, 4, 5, 8])
    report = evaluate_generation(
        sample, retrieval, answerer, recording, {"model": "fake"}, clock=lambda: next(times)
    )
    assert len(answerer.calls) == len(search.calls) == 3
    assert len(report["questions"]) == 3
    assert report["questions"][0]["answer"] == "Ответ без изменений."
    assert report["questions"][0]["cited_contexts"] == [
        {"context_id": "C1", "text": "Точный текст чанка."}
    ]
    assert report["questions"][1]["success"] is False
    assert report["questions"][2]["primary_source_id"] is None
    assert report["metrics"]["reliability"]["generation_errors"] == 1
    paths = write_generation_reports(report, tmp_path)
    assert all(path.exists() for path in paths)
    assert len(json.loads(paths[0].read_text(encoding="utf-8"))["questions"]) == 3
    assert "## Промахи" in render_generation_markdown(report)


def test_manual_review_validation_and_partial_summary(datasets):
    generation, _ = datasets
    question = generation["questions"][0]
    row = {
        "question_id": question["question_id"],
        "question": question["question"],
        "expected_status": "answered",
        "success": True,
        "actual_status": "answered",
        "reference_facts": question["reference_facts"],
        "manual_review_note": question["manual_review_note"],
        "answer": "Точный ответ",
        "citations": [{"context_id": "C1", "source_id": "source"}],
        "cited_contexts": [{"context_id": "C1", "text": "Доказательство"}],
    }
    report = {"dataset_id": generation["dataset_id"], "questions": [row]}
    review = draft_manual_review(report)
    assert len(review["reviews"]) == 1
    assert review["reviews"][0]["answer"] == "Точный ответ"
    assert review["reviews"][0]["citations"] == row["citations"]
    with pytest.raises(ValueError):
        validate_manual_review(review, report)
    assert validate_manual_review(review, report, partial=True) == []
    with pytest.raises(ValueError):
        summarize_manual_review(review, report, partial=True)
    for key in ("factual_correctness", "faithfulness", "completeness", "citation_support"):
        review["reviews"][0][key] = 2
    review["reviews"][0]["faithfulness"] = 0
    summary = summarize_manual_review(review, report)
    assert summary["reviewed_count"] == 1
    assert summary["means"]["faithfulness"] == 0
    assert summary["any_zero_ids"] == [question["question_id"]]
    for changed in ("answer", "citations", "cited_contexts", "question_id"):
        altered = copy.deepcopy(review)
        altered["reviews"][0][changed] = "tampered"
        with pytest.raises(ValueError):
            validate_manual_review(altered, report)
    for altered_rows in ([], review["reviews"] + [review["reviews"][0]]):
        altered = copy.deepcopy(review)
        altered["reviews"] = altered_rows
        with pytest.raises(ValueError):
            validate_manual_review(altered, report)
    altered = copy.deepcopy(review)
    altered["reviews"][0]["factual_correctness"] = 3
    with pytest.raises(ValueError):
        validate_manual_review(altered, report)
    altered["reviews"][0]["factual_correctness"] = None
    with pytest.raises(ValueError):
        validate_manual_review(altered, report, partial=True)

    second = copy.deepcopy(row)
    second["question_id"] = "gen-002"
    two_row_report = {"dataset_id": generation["dataset_id"], "questions": [row, second]}
    partial_review = draft_manual_review(two_row_report)
    for key in ("factual_correctness", "faithfulness", "completeness", "citation_support"):
        partial_review["reviews"][0][key] = 2
    assert (
        summarize_manual_review(partial_review, two_row_report, partial=True)["reviewed_count"] == 1
    )
    with pytest.raises(ValueError):
        summarize_manual_review(partial_review, two_row_report)


def test_validate_dataset_cli_is_offline(capsys):
    assert main(["validate-dataset"]) == 0
    assert "valid=true" in capsys.readouterr().out
