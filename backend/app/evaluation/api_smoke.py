"""Explicit real HTTP evaluation; never invoked by ordinary pytest."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.corpus.paths import CorpusPaths
from app.evaluation.api_generation_gate import load_v1_baseline
from app.evaluation.api_smoke_spec import CASE_IDS, validate_evidence_pages, validate_spec
from app.evaluation.generation_cli import (
    ROOT,
    check_baseline,
    load_datasets,
    preflight_corpus,
    preflight_ollama,
)
from app.evaluation.generation_runner import latency_metrics
from app.generation.models import AnswerResponse
from app.generation.prompt import PROMPT_VERSION
from app.ingestion.manifest import load_manifest
from app.ingestion.writer import write_document
from app.main import create_app
from app.retrieval.index import _validated_index
from app.retrieval.table_index import validated_table_index

HASHES = {
    "retrieval_questions.json": "52de939e1ba13d1558c3e96fa6cec2cabdb69fca9158996aa1a4282d4167511e",
    "generation_questions.json": "11a284b4279f627ae787d0ec4777e0884733b4ec937d6658be3857c2dd15d0a4",
}
HEALTH = {"status": "ok", "service": "unecon-rag-assistant"}


def latency(values: list[float]) -> dict:
    return {key: value for key, value in latency_metrics(values).items() if key != "count"}


SPEC_PATH = ROOT / "data/evaluation/api_smoke_cases.json"
DEFAULT_OUTPUT = ROOT / "data/processed/evaluation/task023b/smoke"


def load_cases() -> list[dict]:
    for name, expected in HASHES.items():
        if hashlib.sha256((ROOT / "data/evaluation" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen dataset hash mismatch: {name}")
    generation, retrieval = load_datasets(
        ROOT / "data/evaluation/generation_questions.json",
        ROOT / "data/evaluation/retrieval_questions.json",
    )
    questions = {q["question_id"]: q for q in generation["questions"]}
    gold = {q["question_id"]: q for q in retrieval["questions"]}
    specification = validate_spec(
        json.loads(SPEC_PATH.read_text(encoding="utf-8")),
        load_manifest(ROOT / "data/source_manifest.json"),
        generation,
    )
    cases = []
    for spec in specification:
        item = questions[spec["generation_question_id"]]
        reference = gold.get(item["retrieval_question_id"])
        cases.append(
            {
                **item,
                "purpose": spec["purpose"],
                "acceptable_evidence": spec.get("acceptable_evidence", []),
                "expected_primary_source": reference["primary_source_id"] if reference else None,
                "expected_pages": (reference["expected_pages"] or []) if reference else [],
                "reference_metadata": reference,
            }
        )
    return cases


SAFETY_IDS = ("gen-048", "gen-059")


def load_safety_cases() -> list[dict]:
    load_cases()  # Verify the same frozen datasets and evidence contract.
    generation, _ = load_datasets(
        ROOT / "data/evaluation/generation_questions.json",
        ROOT / "data/evaluation/retrieval_questions.json",
    )
    questions = {q["question_id"]: q for q in generation["questions"]}
    return [
        {
            **questions[identity],
            "acceptable_evidence": [],
            "expected_primary_source": None,
            "expected_pages": [],
        }
        for identity in SAFETY_IDS
    ]


def preflight() -> CorpusPaths:
    check_baseline("grounded-answer-v2")
    load_v1_baseline()
    expected = {
        "embedding_model": "BAAI/bge-m3",
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "reranker_candidate_k": 20,
        "reranker_max_length": 512,
        "reranker_batch_size": 8,
    }
    if any(getattr(settings, key) != value for key, value in expected.items()):
        raise ValueError("canonical retrieval configuration differs")
    _, retrieval = load_datasets(
        ROOT / "data/evaluation/generation_questions.json",
        ROOT / "data/evaluation/retrieval_questions.json",
    )
    try:
        paths = preflight_corpus(retrieval)
        _, metadata = _validated_index(paths.index, settings.embedding_model)
    except (ValueError, OSError) as exc:
        raise ValueError(
            "Published corpus/index invalid; inspect python -m app.corpus.cli list, "
            "then explicitly validate/publish a candidate or rollback to a verified release"
        ) from exc
    try:
        validated_table_index(
            paths.manifest,
            paths.pdf,
            paths.tables,
            metadata,
            settings.embedding_model,
        )
    except (ValueError, OSError) as exc:
        raise ValueError(
            "Table index invalid in selected release; "
            "validate a candidate or rollback via app.corpus"
        ) from exc
    validate_evidence_pages(load_cases(), metadata["records"])
    preflight_ollama()
    return paths


def check_case(case: dict, http_status: int, body: object) -> dict:
    result = {
        **case,
        "http_status": http_status,
        "actual_status": None,
        "answer": None,
        "citations": [],
        "primary_citation_hit": False,
        "expected_page_citation_hit": False,
        "evidence_citation_hit": False,
        "passed": False,
        "error": None,
    }
    try:
        if http_status != 200:
            raise ValueError(f"HTTP {http_status}")
        answer = AnswerResponse.model_validate(body)
        result.update(
            actual_status=answer.status,
            answer=answer.answer,
            citations=[c.model_dump() for c in answer.citations],
        )
        if answer.query != case["question"].strip() or answer.retrieval_mode != "reranked":
            raise ValueError("query or retrieval mode mismatch")
        if answer.status != case["expected_status"]:
            raise ValueError("unexpected answer status")
        if answer.status == "answered":
            matching = [
                c for c in answer.citations if c.source_id == case["expected_primary_source"]
            ]
            result["primary_citation_hit"] = bool(matching)
            pages = case["expected_pages"]
            result["expected_page_citation_hit"] = any(
                (c.source_type == "pdf" and c.page in pages)
                if pages
                else (c.source_type == "html" and c.page is None)
                for c in matching
            )
            result["evidence_citation_hit"] = any(
                c.source_id == evidence["source_id"]
                and (
                    c.source_type == "pdf"
                    and c.page in evidence.get("pages", [])
                    or c.source_type == "html"
                    and c.page is None
                    and "pages" not in evidence
                )
                for c in answer.citations
                for evidence in case["acceptable_evidence"]
            )
            if not result["evidence_citation_hit"]:
                raise ValueError("verified source/page evidence citation missing")
        result["passed"] = True
    except (ValueError, TypeError) as exc:
        result["error"] = str(exc)
    return result


def session_identity(application) -> tuple:
    service = application.state.answer_service
    retrieval = service.retrieval_service
    return service, retrieval._dense_session, retrieval._reranked_session, service.generator.client


def exercise(client, application, cases: list[dict], clock=time.perf_counter) -> dict:
    health = client.get("/api/v1/health")
    invalid = client.post("/api/v1/answer", json={"question": " \t\n"})
    initial = session_identity(application)
    checks = {
        "health_result": {
            "http_status": health.status_code,
            "passed": health.status_code == 200 and health.json() == HEALTH,
        },
        "validation_result": {
            "http_status": invalid.status_code,
            "passed": invalid.status_code == 422 and initial[1:3] == (None, None),
        },
        "case_results": [],
        "service_reused": True,
    }
    if not checks["health_result"]["passed"] or not checks["validation_result"]["passed"]:
        raise ValueError("health or validation precheck failed")
    previous = None
    for case in cases:
        start = clock()
        try:
            response = client.post("/api/v1/answer", json={"question": case["question"]})
            elapsed = clock() - start
            result = check_case(case, response.status_code, response.json())
        except Exception as exc:
            # Evaluation records failures; the production route still propagates defects.
            elapsed = clock() - start
            result = check_case(case, 0, None)
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_seconds"] = elapsed
        checks["case_results"].append(result)
        current = session_identity(application)
        reused = current[0] is initial[0] and current[3] is initial[3]
        if previous is not None:
            reused = reused and all(a is b for a, b in zip(current, previous, strict=True))
        checks["service_reused"] &= reused
        previous = current
    return checks


def write_reports(report: dict, output_dir: Path, stem="api_e2e_smoke") -> None:
    write_document(output_dir, stem, report)
    lines = [
        "# Application API E2E smoke",
        "",
        f"Verdict: {report['verdict']}",
        f"Passed: {report['pass_count']}/{len(report['case_ids'])}",
        "",
        "```json",
        json.dumps(report, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    (output_dir / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")


def run(output_dir: Path, *, safety: bool = False) -> dict:
    stem = "api_answerability_safety" if safety else "api_e2e_smoke"
    identities = SAFETY_IDS if safety else CASE_IDS
    if output_dir.resolve() == (ROOT / "data/processed/evaluation").resolve():
        raise ValueError("preserve Task023 reports: choose a separate output directory")
    if (output_dir / f"{stem}.json").exists():
        raise ValueError("smoke report already exists; preserve previous runtime evidence")
    # Must precede deferred transformers imports; missing cached models are blockers.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    report = {
        "schema_version": 1,
        "evaluation_id": "application-api-answerability-safety-v1"
        if safety
        else "application-api-e2e-smoke-v1",
        "evaluated_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dataset_hashes": HASHES,
        "smoke_spec_sha256": hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
        "endpoint": "POST /api/v1/answer",
        "model": settings.ollama_model,
        "prompt_version": PROMPT_VERSION,
        "retrieval_mode": settings.retrieval_mode,
        "temperature": settings.generation_temperature,
        "max_tokens": settings.generation_max_tokens,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "case_ids": list(identities),
        "case_results": [],
        "pass_count": 0,
        "health_result": None,
        "validation_result": None,
        "ollama_preflight_result": None,
        "corpus_preflight_passed": False,
        "verdict": "api_e2e_smoke_failed",
        "error": None,
    }
    try:
        cases = load_safety_cases() if safety else load_cases()
        selected_corpus = preflight()
        report.update(ollama_preflight_result={"passed": True}, corpus_preflight_passed=True)
        application = create_app()
        with TestClient(application) as client:
            if selected_corpus is not None:
                retrieval_service = application.state.answer_service.retrieval_service
                if (
                    retrieval_service.manifest_path != selected_corpus.manifest
                    or retrieval_service.index_dir != selected_corpus.index
                    or retrieval_service.table_index_dir != selected_corpus.tables
                ):
                    raise ValueError("corpus changed between preflight and application startup")
            report.update(exercise(client, application, cases))
        report["pass_count"] = sum(c["passed"] for c in report["case_results"])
        if report["pass_count"] == len(identities) and report["service_reused"]:
            report["verdict"] = "api_e2e_smoke_passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    values = [c["elapsed_seconds"] for c in report["case_results"]]
    report["latency_summary"] = {
        **latency(values),
        "cold_request_seconds": values[0] if values else None,
        "warm": latency(values[1:]),
    }
    write_reports(report, output_dir, stem)
    return report
