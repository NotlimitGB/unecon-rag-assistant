import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from app.evaluation.cli import run_evaluation
from app.evaluation.dataset import (
    DatasetError,
    load_dataset,
    validate_dataset,
    validate_page_labels,
)
from app.evaluation.metrics import grouped_primary_metrics, rank_metrics
from app.evaluation.runner import evaluate_retrieval, render_markdown, write_reports
from app.ingestion.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "data/source_manifest.json")
DATASET_PATH = ROOT / "data/evaluation/retrieval_questions.json"


@pytest.fixture
def raw_dataset():
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def test_production_dataset_contract(raw_dataset):
    validate_dataset(raw_dataset, MANIFEST)
    questions = raw_dataset["questions"]
    assert len(questions) == 80
    assert [item["question_id"] for item in questions] == [f"ret-{n:03d}" for n in range(1, 81)]
    assert Counter(item["difficulty"] for item in questions) == {
        "easy": 25,
        "medium": 35,
        "hard": 20,
    }
    categories = Counter(item["category"] for item in questions)
    assert all(5 <= count <= 20 for count in categories.values())
    assert load_dataset(DATASET_PATH, MANIFEST) == raw_dataset


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["questions"][1].update(question_id="ret-001"),
        lambda d: d["questions"][1].update(question="  " + d["questions"][0]["question"] + " "),
        lambda d: d["questions"][0].update(primary_source_id="missing"),
        lambda d: d["questions"][0].update(primary_source_id="admission-rules"),
        lambda d: d["questions"][0].update(acceptable_source_ids=["admissions-faq"]),
        lambda d: d["questions"][0].update(acceptable_source_ids=["admission-deadlines-pdf"] * 2),
        lambda d: d["questions"][0].update(difficulty="unknown"),
        lambda d: d["questions"][0].update(expected_pages=[0]),
        lambda d: d["questions"][72].update(expected_pages=[1]),
        lambda d: d["questions"][0].update(unexpected=True),
        lambda d: d["questions"][0].update(category="unknown"),
    ],
)
def test_invalid_dataset_rejected(raw_dataset, change):
    change(raw_dataset)
    with pytest.raises(DatasetError):
        validate_dataset(raw_dataset, MANIFEST)


def test_inactive_primary_and_distribution_rejected(raw_dataset):
    manifest = copy.deepcopy(MANIFEST)
    manifest.sources[8].status = "draft"
    with pytest.raises(DatasetError):
        validate_dataset(raw_dataset, manifest)
    raw_dataset["questions"][0]["difficulty"] = "hard"
    with pytest.raises(DatasetError, match="distribution"):
        validate_dataset(raw_dataset, MANIFEST)


def test_stale_page_label_rejected(raw_dataset):
    with pytest.raises(DatasetError, match="missing indexed page"):
        validate_page_labels(raw_dataset, [])


def test_dataset_category_coverage_and_root_schema(raw_dataset):
    for item in raw_dataset["questions"][-4:]:
        item["category"] = "admission_rules"
    with pytest.raises(DatasetError, match="category"):
        validate_dataset(raw_dataset, MANIFEST)
    raw_dataset["unexpected"] = True
    with pytest.raises(DatasetError, match="top-level"):
        validate_dataset(raw_dataset, MANIFEST)


def test_rank_metric_formulas():
    metrics = rank_metrics([1, 3, None])
    assert metrics["recall_at_1"] == pytest.approx(1 / 3)
    assert metrics["recall_at_3"] == pytest.approx(2 / 3)
    assert metrics["recall_at_5"] == pytest.approx(2 / 3)
    assert metrics["mrr_at_5"] == pytest.approx((1 + 1 / 3) / 3)
    assert all(value is None for value in rank_metrics([]).values())


def test_grouped_metrics():
    rows = [
        {"category": "a", "difficulty": "easy", "primary_rank": 1},
        {"category": "a", "difficulty": "hard", "primary_rank": None},
        {"category": "b", "difficulty": "hard", "primary_rank": 3},
    ]
    categories = grouped_primary_metrics(rows, "category")
    difficulties = grouped_primary_metrics(rows, "difficulty")
    assert categories["a"]["question_count"] == 2
    assert categories["a"]["primary_recall_at_1"] == 0.5
    assert categories["b"]["primary_mrr_at_5"] == pytest.approx(1 / 3)
    assert difficulties["hard"]["primary_recall_at_5"] == 0.5


class FakeSession:
    metadata = {
        "embedding": {"model": "fake"},
        "index": {"type": "IndexFlatIP", "vector_count": 3},
    }

    def __init__(self):
        self.calls = []

    def search(self, question, top_k=5):
        self.calls.append((question, top_k))
        return [
            {
                "source_id": "other",
                "chunk_id": "o:1",
                "score": 0.9,
                "page_start": 2,
                "text": "x" * 300,
            },
            {
                "source_id": "pdf",
                "chunk_id": "p:1",
                "score": 0.8,
                "page_start": 2,
                "text": "PDF page text",
            },
        ]


def test_runner_metrics_and_reports(tmp_path):
    dataset = {
        "dataset_id": "synthetic",
        "questions": [
            {
                "question_id": "ret-001",
                "question": "Вопрос один?",
                "category": "a",
                "difficulty": "easy",
                "primary_source_id": "pdf",
                "acceptable_source_ids": ["pdf", "other"],
                "expected_pages": [2],
                "evidence_note": "evidence",
            },
            {
                "question_id": "ret-002",
                "question": "Вопрос два?",
                "category": "b",
                "difficulty": "hard",
                "primary_source_id": "missing",
                "acceptable_source_ids": ["missing"],
                "expected_pages": None,
                "evidence_note": "evidence",
            },
        ],
    }
    session = FakeSession()
    report = evaluate_retrieval(dataset, session)
    assert session.calls == [("Вопрос один?", 5), ("Вопрос два?", 5)]
    assert report["questions"][0]["primary_rank"] == 2
    assert report["questions"][0]["accepted_rank"] == 1
    assert report["questions"][0]["page_rank"] == 2
    assert report["questions"][1]["primary_rank"] is None
    assert len(report["questions"][0]["top_results"][0]["text_excerpt"]) == 240
    assert report["metrics"]["primary_recall_at_5"] == 0.5
    assert report["metrics"]["accepted_recall_at_1"] == 0.5
    assert report["metrics"]["page_recall_at_3"] == 1.0
    assert report["metrics"]["page_labeled_questions"] == 1
    assert report["category_metrics"]["a"]["question_count"] == 1
    assert report["difficulty_metrics"]["hard"]["primary_recall_at_5"] == 0
    assert set(report) == {
        "schema_version",
        "dataset",
        "retrieval",
        "metrics",
        "category_metrics",
        "difficulty_metrics",
        "questions",
    }
    json_path, md_path = write_reports(report, tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    markdown = md_path.read_text(encoding="utf-8")
    assert "Промахи Primary top-5" in markdown
    assert "ret-002" in markdown
    assert "N/A" not in markdown
    assert "timestamp" not in json_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in json_path.read_text(encoding="utf-8")
    assert render_markdown(report) == markdown
    report["questions"][0]["primary_rank"] = 4
    assert "ret-001: позиция 4" in render_markdown(report)


def test_bad_page_label_fails_before_model(tmp_path, monkeypatch):
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    dataset["questions"][0]["expected_pages"] = [999]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    records = [{"source_id": "admission-deadlines-pdf", "page_start": 1}]
    metadata = {
        "corpus": {"sources": []},
        "records": records,
    }
    monkeypatch.setattr("app.evaluation.cli._validated_index", lambda *args: (None, metadata))
    monkeypatch.setattr("app.evaluation.cli.load_corpus", lambda *args: ([], records))
    created = []
    monkeypatch.setattr("app.evaluation.cli.RetrievalSession", lambda *args: created.append(1))
    with pytest.raises(DatasetError, match="missing indexed page"):
        run_evaluation(path, ROOT / "data/source_manifest.json", tmp_path, tmp_path, tmp_path)
    assert created == []
