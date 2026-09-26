"""Offline tests of the real smoke harness using synthetic HTTP responses."""

from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app.evaluation import api_smoke as smoke
from app.evaluation import api_smoke_cli as cli
from app.generation.models import REFUSAL_MESSAGE
from app.main import create_app

EXPECTED = (
    ("admission-deadlines-pdf", 1),
    ("admission-capacity-pdf", 1),
    ("entrance-exams-list-pdf", 2),
    ("tuition", None),
    ("tuition-order-128-pdf", 2),
    (None, None),
)


def cases():
    return [
        {
            "question_id": identity,
            "question": identity,
            "reference_facts": ["fact"],
            "acceptable_evidence": [{"source_id": source, **({"pages": [page]} if page else {})}]
            if source
            else [],
            "expected_primary_source": source,
            "expected_pages": [page] if page else [],
            "expected_status": "answered" if source else "insufficient_evidence",
        }
        for identity, (source, page) in zip(smoke.CASE_IDS, EXPECTED, strict=True)
    ]


def body(case):
    source, pages = case["expected_primary_source"], case["expected_pages"]
    return {
        "query": case["question"],
        "status": case["expected_status"],
        "answer": "Ответ" if source else REFUSAL_MESSAGE,
        "retrieval_mode": "reranked",
        "citations": [
            {
                "context_id": "C1",
                "chunk_id": "chunk",
                "source_id": source,
                "source_title": "Источник",
                "source_url": "https://unecon.ru/doc/",
                "source_type": "pdf" if pages else "html",
                "page": pages[0] if pages else None,
            }
        ]
        if source
        else [],
    }


@pytest.mark.parametrize("case", cases())
def test_all_acceptance_cases(case):
    assert smoke.check_case(case, 200, body(case))["passed"]


@pytest.mark.parametrize(
    "change", ["source", "page", "url", "status", "query", "mode", "extra", "refusal", "html_page"]
)
def test_strict_gates(change):
    case = cases()[5 if change == "refusal" else 3 if change == "html_page" else 0]
    payload = body(case)
    if change == "source":
        payload["citations"][0]["source_id"] = "different"
    elif change in ("page", "html_page"):
        payload["citations"][0]["page"] = 99
    elif change == "url":
        payload["citations"][0]["source_url"] = "https://external.example/doc/"
    elif change == "status":
        payload.update(status="insufficient_evidence", answer=REFUSAL_MESSAGE, citations=[])
    elif change == "query":
        payload["query"] = "wrong"
    elif change == "mode":
        payload["retrieval_mode"] = "dense"
    elif change == "extra":
        payload["debug"] = True
    elif change == "refusal":
        payload["answer"] = "different refusal"
    assert not smoke.check_case(case, 200, payload)["passed"]


def test_http_and_nonobject_failure():
    assert not smoke.check_case(cases()[0], 503, {"detail": "unavailable"})["passed"]
    assert not smoke.check_case(cases()[0], 200, [])["passed"]


def test_latency_nearest_rank():
    assert smoke.latency([1, 2, 3, 4, 5, 6]) == {
        "mean": 3.5,
        "median": 3.5,
        "p95": 6,
        "min": 1,
        "max": 6,
    }
    assert smoke.latency(list(range(1, 21)))["p95"] == 19
    assert smoke.latency([])["mean"] is None


class Service:
    def __init__(self):
        self.retrieval_service = SimpleNamespace(_dense_session=None, _reranked_session=None)
        self.generator = SimpleNamespace(client=object())
        self.calls = []
        self.closed = 0

    def answer(self, question):
        self.calls.append(question)
        if self.retrieval_service._dense_session is None:
            self.retrieval_service._dense_session = object()
            self.retrieval_service._reranked_session = object()
        return body(next(c for c in cases() if c["question"] == question))

    def close(self):
        self.closed += 1


def test_one_lifespan_order_reports_and_cli(monkeypatch, tmp_path):
    service = Service()
    monkeypatch.setattr(smoke, "load_cases", cases)
    monkeypatch.setattr(smoke, "preflight", lambda: None)
    monkeypatch.setattr(smoke, "create_app", lambda: create_app(lambda: service))
    assert cli.main(["run", "--output-dir", str(tmp_path)]) == 0
    import json

    report = json.loads((tmp_path / "api_e2e_smoke.json").read_text(encoding="utf-8"))
    assert report["pass_count"] == 6
    assert service.calls == list(smoke.CASE_IDS)
    assert service.closed == 1 and report["service_reused"]
    assert report["validation_result"]["passed"]
    assert report["latency_summary"]["cold_request_seconds"] >= 0
    assert report["latency_summary"]["warm"]["mean"] >= 0
    assert (tmp_path / "api_e2e_smoke.md").exists()


def test_preflight_failure_no_application(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "load_cases", cases)

    def fail():
        raise ValueError("missing table index")

    monkeypatch.setattr(smoke, "preflight", fail)
    monkeypatch.setattr(smoke, "create_app", lambda: pytest.fail("must not create app"))
    assert cli.main(["run", "--output-dir", str(tmp_path)]) == 1
    assert "missing table index" in (tmp_path / "api_e2e_smoke.json").read_text()


@pytest.mark.parametrize("failure", ["json", "http", "exception", "citation"])
def test_failed_case_continues_without_retry(failure):
    service = Service()
    application = create_app(lambda: service)
    with TestClient(application) as actual:

        class Client:
            count = 0

            def get(self, *args, **kwargs):
                return actual.get(*args, **kwargs)

            def post(self, url, **kwargs):
                if kwargs["json"]["question"].strip():
                    self.count += 1
                    if self.count == 2:
                        if failure == "exception":
                            raise RuntimeError("defect")
                        if failure == "json":
                            return httpx.Response(200, text="invalid JSON")
                        if failure == "http":
                            return httpx.Response(503, json={"detail": "unavailable"})
                        payload = body(cases()[1])
                        payload["citations"][0]["source_id"] = "wrong"
                        return httpx.Response(200, json=payload)
                return actual.post(url, **kwargs)

        client = Client()
        report = smoke.exercise(client, application, cases())
    assert client.count == 6
    assert [c["question_id"] for c in report["case_results"]] == list(smoke.CASE_IDS)
    assert sum(c["passed"] for c in report["case_results"]) == 5
    assert report["case_results"][1]["error"]


def test_frozen_case_selection():
    selected = smoke.load_cases()
    assert [c["question_id"] for c in selected] == list(smoke.CASE_IDS)
    assert selected[3]["reference_metadata"]["expected_pages"] is None
    assert [c["expected_primary_source"] for c in selected] == [s for s, _ in EXPECTED]


def test_one_failed_gate_cli_is_nonzero(monkeypatch, tmp_path):
    service = Service()
    original = service.answer

    def wrong_source(question):
        result = original(question)
        if question == smoke.CASE_IDS[0]:
            result["citations"][0]["source_id"] = "other"
        return result

    service.answer = wrong_source
    monkeypatch.setattr(smoke, "load_cases", cases)
    monkeypatch.setattr(smoke, "preflight", lambda: None)
    monkeypatch.setattr(smoke, "create_app", lambda: create_app(lambda: service))
    assert cli.main(["run", "--output-dir", str(tmp_path)]) == 1
    assert service.calls == list(smoke.CASE_IDS)
