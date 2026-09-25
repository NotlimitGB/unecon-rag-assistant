"""All review tests use synthetic answers and local tracked datasets only."""

import copy
import json
import shutil

import pytest

from app.config import PROJECT_ROOT
from app.evaluation.dataset import load_dataset
from app.evaluation.generation_dataset import load_generation_dataset
from app.evaluation.generation_runner import SCORES
from app.experiments.table_aware import human_review, human_review_cli
from app.ingestion.manifest import load_manifest


@pytest.fixture
def packet(tmp_path):
    dataset_dir = tmp_path / "data/evaluation"
    report_dir = tmp_path / "data/processed/evaluation"
    dataset_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    for relative in (
        "data/evaluation/generation_questions.json",
        "data/evaluation/retrieval_questions.json",
        "data/source_manifest.json",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative, destination)
    manifest = load_manifest(tmp_path / "data/source_manifest.json")
    sources = {source.id: source for source in manifest.sources}
    retrieval = load_dataset(dataset_dir / "retrieval_questions.json", manifest)
    generation = load_generation_dataset(dataset_dir / "generation_questions.json", retrieval)
    gold = {row["question_id"]: row for row in retrieval["questions"]}
    selections = []
    selection_by_id = {}
    for row in retrieval["questions"]:
        source = sources[row["primary_source_id"]]
        page = (row["expected_pages"] or [1])[0] if source.source_type == "pdf" else None
        top5 = [
            {
                "chunk_id": f"{row['question_id']}:context-{n}",
                "representation": "production_chunk",
                "source_id": source.id,
                "page": page,
            }
            for n in range(1, 6)
        ]
        selected = {"question_id": row["question_id"], "diversity_top5": top5}
        selections.append(selected)
        selection_by_id[row["question_id"]] = selected
    (report_dir / "pdf_page_diversity_retrieval_experiment.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": human_review.EXPERIMENT_ID,
                "phase_a_verdict": "diversity_retrieval_gate_passed",
                "corpus_counts": {"production_chunks": 338, "table_rows": 250, "questions": 80},
                "gate": {str(n): True for n in range(17)},
                "selection_diagnostics": selections,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rows = []
    source_items = {row["question_id"]: row for row in generation["questions"]}
    for qid in human_review.QUESTION_IDS:
        item = source_items[qid]
        label = gold[item["retrieval_question_id"]]
        selected = selection_by_id[item["retrieval_question_id"]]["diversity_top5"]
        source = sources[selected[0]["source_id"]]
        contexts = [
            {
                "context_id": f"C{n}",
                "chunk_id": hit["chunk_id"],
                "representation": hit["representation"],
                "text": f"Точный официальный контекст {qid} {n}. `код`",
            }
            for n, hit in enumerate(selected, 1)
        ]
        answered = qid != "gen-012"
        citations = (
            [
                {
                    "context_id": "C1",
                    "chunk_id": selected[0]["chunk_id"],
                    "source_id": source.id,
                    "source_title": source.title,
                    "source_url": source.url,
                    "source_type": source.source_type,
                    "page": selected[0]["page"],
                }
            ]
            if answered
            else []
        )
        result = {
            "success": True,
            "error": None,
            "question_id": qid,
            "question": item["question"],
            "retrieval_question_id": item["retrieval_question_id"],
            "expected_status": item["expected_status"],
            "actual_status": "answered" if answered else "insufficient_evidence",
            "reference_facts": item["reference_facts"],
            "manual_review_note": item["manual_review_note"],
            "category": label["category"],
            "difficulty": label["difficulty"],
            "primary_source_id": label["primary_source_id"],
            "acceptable_source_ids": label["acceptable_source_ids"],
            "expected_pages": label["expected_pages"],
            "answer": f"Синтетический ответ {qid}" if answered else "Синтетический отказ",
            "citations": citations,
            "cited_contexts": [{"context_id": "C1", "text": contexts[0]["text"]}]
            if answered
            else [],
        }
        rows.append({"question_id": qid, "augmented": result, "diversity_contexts": contexts})
    (report_dir / "pdf_page_diversity_generation_experiment.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": human_review.EXPERIMENT_ID,
                "verdict": "diversity_integration_promising",
                "generation_question_ids": list(human_review.QUESTION_IDS),
                "status_movement": {"error": 0, "new_refusal": 0},
                "critical_probes": [{"question_id": qid} for qid in sorted(human_review.PROBE_IDS)],
                "questions": rows,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return human_review.expected_packet(tmp_path)


def test_blank_packet_exact_evidence_and_order(packet):
    assert set(packet) == human_review.TOP_KEYS
    assert [case["question_id"] for case in packet["cases"]] == list(human_review.QUESTION_IDS)
    assert sum(case["answer_scoring_applicable"] for case in packet["cases"]) == 15
    assert [
        case["question_id"] for case in packet["cases"] if not case["answer_scoring_applicable"]
    ] == ["gen-012"]
    assert packet["reviewer_name"] == packet["reviewed_at"] == packet["reviewer_note"] == ""
    assert (
        len(packet["source_generation_report_sha256"])
        == len(packet["source_retrieval_report_sha256"])
        == 64
    )
    assert all(len(case["retrieved_contexts"]) == 5 for case in packet["cases"])
    assert all(
        case["context_sufficiency"] is None and all(case[score] is None for score in SCORES)
        for case in packet["cases"]
    )
    assert packet["cases"][0]["answer"] == "Синтетический ответ gen-007"
    assert packet["cases"][0]["reference_facts"]
    assert packet["cases"][0]["retrieved_contexts"][0]["cited"] is True
    assert packet["cases"][0]["retrieved_contexts"][1]["cited"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question", "изменённый вопрос"),
        ("answer", "изменённый ответ"),
        ("reference_facts", ["выдуманный факт"]),
        ("retrieved_contexts", []),
        ("citations", []),
        ("primary_source_id", "другой-источник"),
        ("expected_pages", [999]),
        ("factual_correctness", 3),
        ("factual_correctness", -1),
        ("factual_correctness", "2"),
        ("context_sufficiency", "maybe"),
        ("needs_gold_review", 1),
    ],
)
def test_rejects_evidence_mutation_and_invalid_human_value(packet, field, value):
    edited = copy.deepcopy(packet)
    edited["cases"][0][field] = value
    with pytest.raises(ValueError):
        human_review.validate_review(edited, packet, partial=True)


def test_rejects_changed_context_text_and_citation_provenance(packet):
    for field in ("text", "source_id", "page"):
        edited = copy.deepcopy(packet)
        edited["cases"][0]["retrieved_contexts"][0][field] = "wrong"
        with pytest.raises(ValueError):
            human_review.validate_review(edited, packet, partial=True)
    edited = copy.deepcopy(packet)
    edited["cases"][0]["citations"][0]["chunk_id"] = "wrong"
    with pytest.raises(ValueError):
        human_review.validate_review(edited, packet, partial=True)


def test_immutable_json_types_are_strict(packet):
    edited = copy.deepcopy(packet)
    edited["schema_version"] = True
    with pytest.raises(ValueError):
        human_review.validate_review(edited, packet, partial=True)
    edited = copy.deepcopy(packet)
    edited["cases"][0]["answer_scoring_applicable"] = 1
    with pytest.raises(ValueError):
        human_review.validate_review(edited, packet, partial=True)


def test_partial_full_and_refusal_rules(packet):
    assert human_review.validate_review(packet, packet, partial=True)["untouched"] == 16
    with pytest.raises(ValueError):
        human_review.validate_review(packet, packet)
    edited = copy.deepcopy(packet)
    edited["cases"][0]["context_sufficiency"] = "sufficient"
    for score in SCORES:
        edited["cases"][0][score] = 2
    progress = human_review.validate_review(edited, packet, partial=True)
    assert (progress["fully_reviewed"], progress["untouched"]) == (1, 15)
    edited["cases"][1]["factual_correctness"] = 1
    assert human_review.validate_review(edited, packet, partial=True)["partially_reviewed"] == 1
    edited["cases"][5]["factual_correctness"] = 0
    with pytest.raises(ValueError, match="refusal"):
        human_review.validate_review(edited, packet, partial=True)


def test_manual_summary_uses_only_completed_answer_rows(packet):
    edited = copy.deepcopy(packet)
    edited["reviewer_name"] = "Рецензент"
    edited["reviewed_at"] = "2026-09-25"
    for i, scores in ((0, [2, 2, 2, 2]), (1, [0, 1, 2, 1])):
        case = edited["cases"][i]
        case["context_sufficiency"] = "sufficient" if i == 0 else "insufficient"
        for score, value in zip(SCORES, scores, strict=True):
            case[score] = value
    edited["cases"][1]["needs_gold_review"] = True
    edited["cases"][5]["context_sufficiency"] = "unclear"
    edited["cases"][2]["factual_correctness"] = 2  # incomplete: excluded
    summary = human_review.summarize_review(edited, packet, partial=True)
    assert summary["reviewed_answer_count"] == 2
    assert summary["available_answer_count"] == 15
    assert summary["metrics"]["factual_correctness"] == {
        "mean": 1,
        "score_2_count": 1,
        "score_2_percent": 50,
        "score_1_count": 0,
        "score_0_count": 1,
        "score_0_question_ids": ["gen-008"],
    }
    assert summary["fully_clean_answer_ids"] == ["gen-007"]
    assert summary["any_zero_answer_ids"] == ["gen-008"]
    assert summary["metrics"]["faithfulness"]["mean"] == 1.5
    assert summary["metrics"]["faithfulness"]["score_1_count"] == 1
    assert summary["metrics"]["completeness"]["score_2_percent"] == 100
    assert summary["metrics"]["citation_support"]["score_0_count"] == 0
    assert summary["context_sufficiency"] == {
        "sufficient": 1,
        "insufficient": 1,
        "unclear": 1,
        "insufficient_ids": ["gen-008"],
        "unclear_ids": ["gen-012"],
    }
    assert summary["refusals"]["false_refusal_ids"] == ["gen-012"]
    assert summary["refusals"]["false_refusal_context_sufficiency"] == {"gen-012": "unclear"}
    assert summary["needs_gold_review_ids"] == ["gen-008"]
    with pytest.raises(ValueError):
        human_review.summarize_review(edited, packet)


def test_complete_human_fixture_validates_without_scoring_refusal(packet):
    edited = copy.deepcopy(packet)
    edited["reviewer_name"] = "Рецензент"
    edited["reviewed_at"] = "2026-09-25"
    for case in edited["cases"]:
        case["context_sufficiency"] = "sufficient"
        if case["answer_scoring_applicable"]:
            for score in SCORES:
                case[score] = 2
    progress = human_review.validate_review(edited, packet)
    assert progress["fully_reviewed"] == 16
    assert progress["completed_answer_ids"] == [
        case["question_id"] for case in edited["cases"] if case["answer_scoring_applicable"]
    ]
    summary = human_review.summarize_review(edited, packet)
    assert summary["fully_clean_answer_count"] == 15
    assert summary["refusals"]["false_refusal_ids"] == ["gen-012"]


def test_markdown_contains_exact_evidence_without_prefilled_scores(packet):
    markdown = human_review.render_worksheet(packet)
    for case in packet["cases"]:
        assert markdown.count(f"## {case['question_id']} — ") == 1
        assert case["question"] in markdown
        assert case["answer"] in markdown
        assert all(fact in markdown for fact in case["reference_facts"])
        assert all(context["text"] in markdown for context in case["retrieved_contexts"])
    assert "Semantic answer scores are not applicable" in markdown
    assert "Процитирован: да" in markdown
    assert "Процитирован: нет" in markdown
    assert "Фактическая правильность:" in markdown
    assert "Ни одна оценка не заполнена заранее" in markdown


def test_prepare_refuses_overwriting_human_work(packet, tmp_path, monkeypatch):
    review_path = tmp_path / "review.json"
    worksheet_path = tmp_path / "review.md"
    monkeypatch.setattr(human_review_cli, "expected_packet", lambda: packet)
    monkeypatch.setattr(human_review_cli, "REVIEW_PATH", review_path)
    monkeypatch.setattr(human_review_cli, "WORKSHEET_PATH", worksheet_path)
    assert human_review_cli.prepare()["state"] == "created"
    original = review_path.read_bytes()
    assert human_review_cli.prepare()["state"] == "already_exists"
    assert review_path.read_bytes() == original
    modified = json.loads(review_path.read_text(encoding="utf-8"))
    modified["cases"][0]["factual_correctness"] = 2
    review_path.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing overwrite"):
        human_review_cli.prepare()
    assert (
        json.loads(review_path.read_text(encoding="utf-8"))["cases"][0]["factual_correctness"] == 2
    )
