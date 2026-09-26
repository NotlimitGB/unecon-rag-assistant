"""Offline evidence-policy and current-production regression gates."""

import copy
import json
from types import SimpleNamespace

import pytest

from app.evaluation import api_smoke as smoke
from app.evaluation.api_generation_gate import EXPECTED_CONFIG, validate_generation
from app.evaluation.api_smoke_spec import validate_evidence_pages, validate_spec
from app.evaluation.generation_cli import ROOT, load_datasets
from app.evaluation.generation_runner import automatic_metrics
from app.generation.models import REFUSAL_MESSAGE
from app.ingestion.manifest import load_manifest


def inputs():
    generation, retrieval = load_datasets(
        ROOT / "data/evaluation/generation_questions.json",
        ROOT / "data/evaluation/retrieval_questions.json",
    )
    specification = json.loads(smoke.SPEC_PATH.read_text(encoding="utf-8"))
    manifest = load_manifest(ROOT / "data/source_manifest.json")
    return specification, manifest, generation, retrieval


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "version",
        "order",
        "count",
        "purpose",
        "duplicate",
        "unknown",
        "inactive",
        "pages",
        "bool_page",
        "repeat_page",
        "html_pages",
        "empty",
        "status",
        "unsupported_evidence",
    ],
)
def test_invalid_spec(mutation):
    spec, manifest, generation, _ = inputs()
    case = spec["cases"][0]
    if mutation == "extra":
        spec["extra"] = True
    elif mutation == "version":
        spec["schema_version"] = True
    elif mutation == "order":
        spec["cases"].reverse()
    elif mutation == "count":
        spec["cases"].pop()
    elif mutation == "purpose":
        case["purpose"] = " "
    elif mutation == "duplicate":
        case["acceptable_evidence"] *= 2
    elif mutation == "unknown":
        case["acceptable_evidence"][0]["source_id"] = "unknown"
    elif mutation == "inactive":
        manifest = SimpleNamespace(
            sources=[s.model_copy(update={"active": False}) for s in manifest.sources]
        )
    elif mutation == "pages":
        case["acceptable_evidence"][0]["pages"] = []
    elif mutation == "bool_page":
        case["acceptable_evidence"][0]["pages"] = [True]
    elif mutation == "repeat_page":
        case["acceptable_evidence"][0]["pages"] = [1, 1]
    elif mutation == "html_pages":
        spec["cases"][3]["acceptable_evidence"][0]["pages"] = [1]
    elif mutation == "empty":
        case["acceptable_evidence"] = []
    elif mutation == "status":
        case["expected_status"] = "insufficient_evidence"
    elif mutation == "unsupported_evidence":
        spec["cases"][5]["acceptable_evidence"] = []
    with pytest.raises(ValueError):
        validate_spec(spec, manifest, generation)


def test_pages_and_alternate_evidence():
    spec, manifest, generation, _ = inputs()
    validate_spec(spec, manifest, generation)
    records = [
        {"source_id": e["source_id"], "page_start": p}
        for c in spec["cases"]
        for e in c.get("acceptable_evidence", [])
        for p in e.get("pages", [None])
    ]
    validate_evidence_pages(spec["cases"], records)
    with pytest.raises(ValueError):
        validate_evidence_pages(spec["cases"], records[1:])
    case = smoke.load_cases()[2]
    payload = {
        "query": case["question"],
        "status": "answered",
        "answer": "Математика",
        "retrieval_mode": "reranked",
        "citations": [
            {
                "context_id": "C1",
                "chunk_id": "rules",
                "source_id": "admission-rules-pdf",
                "source_title": "Правила",
                "source_url": "https://unecon.ru/rules/",
                "source_type": "pdf",
                "page": 51,
            }
        ],
    }
    for page in (51, 52, 56):
        payload["citations"][0]["page"] = page
        result = smoke.check_case(case, 200, payload)
        assert result["passed"] and result["evidence_citation_hit"]
        assert not result["primary_citation_hit"] and not result["expected_page_citation_hit"]
    payload["citations"][0]["page"] = 53
    assert not smoke.check_case(case, 200, payload)["passed"]


def full_report():
    _, _, generation, retrieval = inputs()
    gold = {q["question_id"]: q for q in retrieval["questions"]}
    rows = []
    for item in generation["questions"]:
        linked = gold.get(item["retrieval_question_id"])
        # Supported refusals are allowed: this synthetic fixture needs no production chunks.
        rows.append(
            {
                **item,
                "success": True,
                "error": None,
                "actual_status": "insufficient_evidence",
                "answer": REFUSAL_MESSAGE,
                "retrieval_mode": "reranked",
                "citations": [],
                "cited_contexts": [],
                "primary_source_id": linked["primary_source_id"] if linked else None,
                "acceptable_source_ids": linked["acceptable_source_ids"] if linked else [],
                "expected_pages": linked["expected_pages"] if linked else None,
                "primary_source_hit": None,
                "acceptable_source_hit": None,
                "expected_page_hit": None,
                "elapsed_seconds": 1.0,
                "status_correct": item["expected_status"] == "insufficient_evidence",
            }
        )
    report = {
        "schema_version": 1,
        "dataset_id": generation["dataset_id"],
        "configuration": copy.deepcopy(EXPECTED_CONFIG),
        "questions": rows,
        "metrics": automatic_metrics(rows),
    }
    return report, generation, retrieval


def test_supported_refusal_and_gold_review():
    report, generation, retrieval = full_report()
    before = copy.deepcopy(generation)
    result = validate_generation(report, generation, retrieval)
    assert result["passed"] and result["gen_048_refused"]
    assert "gen-012" in result["supported_refusals"]
    assert result["gold_review"]["gen-008"]["needs_gold_review"] is True
    assert len(result["unsupported_statuses"]) == 18
    assert generation == before


@pytest.mark.parametrize(
    "mutation",
    [
        "config",
        "identity",
        "count",
        "order",
        "question",
        "error",
        "invalid_citation_mapping",
        "mapping",
        "refusal",
        "metrics",
        "gen-048",
        "gen-060",
    ],
)
def test_generation_gate_failures(mutation):
    report, generation, retrieval = full_report()
    row = report["questions"][0]
    if mutation == "config":
        report["configuration"]["temperature"] = 0
    elif mutation == "identity":
        report["dataset_id"] = "wrong"
    elif mutation == "count":
        report["questions"].pop()
    elif mutation == "order":
        report["questions"].reverse()
    elif mutation == "question":
        row["question"] = "changed"
    elif mutation in ("error", "invalid_citation_mapping"):
        row.update(success=False, error=mutation, actual_status=None)
    elif mutation == "mapping":
        row["cited_contexts"] = [{"context_id": "C1", "text": "x"}]
    elif mutation == "refusal":
        row["answer"] = "different refusal"
    elif mutation == "metrics":
        report["metrics"]["status"]["overall_status_accuracy"] = 1.0
    else:
        row = next(r for r in report["questions"] if r["question_id"] == mutation)
        row.update(
            actual_status="answered",
            answer="wrong",
            status_correct=False,
            citations=[
                {
                    "context_id": "C1",
                    "chunk_id": "x",
                    "source_id": "tuition",
                    "source_title": "Стоимость",
                    "source_type": "html",
                    "page": None,
                    "source_url": "https://unecon.ru/tuition/",
                }
            ],
            cited_contexts=[{"context_id": "C1", "text": "x"}],
        )
        report["metrics"] = automatic_metrics(report["questions"])
    assert not validate_generation(report, generation, retrieval)["passed"]


def test_protect_old_reports(tmp_path):
    with pytest.raises(ValueError, match="preserve Task023"):
        smoke.run(ROOT / "data/processed/evaluation")
    path = tmp_path / "api_e2e_smoke.json"
    path.write_bytes(b"original")
    with pytest.raises(ValueError, match="already exists"):
        smoke.run(tmp_path)
    assert path.read_bytes() == b"original"


def test_generation_validation_cli(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("app.evaluation.api_smoke_cli.load_v1_baseline", lambda: None)
    from app.evaluation.api_smoke_cli import main

    report, _, _ = full_report()
    path = tmp_path / "generation_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert main(["validate-generation", "--report", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["gold_review"]["gen-008"]["needs_gold_review"]
    report["configuration"]["model"] = "other"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert main(["validate-generation", "--report", str(path)]) == 1
