"""Offline HTTP contract and lifecycle checks; no RAG models or runtime required."""

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_answer_service
from app.api.routes.answer import UNAVAILABLE_MESSAGE
from app.config import settings
from app.generation.models import (
    REFUSAL_MESSAGE,
    AnswerCitation,
    AnswerResponse,
    GenerationError,
)
from app.main import create_app


def response(status="answered", source_type="pdf"):
    return AnswerResponse(
        query="Какие документы нужны?",
        status=status,
        answer="Нужен документ об образовании." if status == "answered" else REFUSAL_MESSAGE,
        retrieval_mode="reranked",
        citations=[
            AnswerCitation(
                context_id="C1",
                chunk_id="documents:0001:abc",
                source_id="admission-documents",
                source_title="Документы",
                source_url="https://unecon.ru/documents/",
                source_type=source_type,
                page=4 if source_type == "pdf" else None,
            )
        ]
        if status == "answered"
        else [],
    )


class FakeAnswerService:
    def __init__(self, result=None, error=None):
        self.result = result if result is not None else response()
        self.error = error
        self.calls = []
        self.closed = 0

    def answer(self, question):
        self.calls.append(question)
        if self.error is not None:
            raise self.error
        return self.result

    def close(self):
        self.closed += 1


@pytest.mark.parametrize("source_type", ["pdf", "html"])
def test_answer_response_and_native_dependency_override(source_type):
    owned = FakeAnswerService()
    fake = FakeAnswerService(response(source_type=source_type))
    application = create_app(lambda: owned)
    application.dependency_overrides[get_answer_service] = lambda: fake
    with TestClient(application) as client:
        result = client.post("/api/v1/answer", json={"question": " Какие документы нужны? "})
    assert result.status_code == 200
    assert result.json() == fake.result.model_dump(mode="json")
    assert fake.calls == ["Какие документы нужны?"]
    assert owned.calls == []
    assert owned.closed == 1 and fake.closed == 0
    assert set(result.json()) == {"query", "status", "answer", "retrieval_mode", "citations"}
    assert set(result.json()["citations"][0]) == {
        "context_id",
        "chunk_id",
        "source_id",
        "source_title",
        "source_url",
        "source_type",
        "page",
    }


def test_insufficient_evidence_is_success():
    fake = FakeAnswerService(response(status="insufficient_evidence"))
    with TestClient(create_app(lambda: fake)) as client:
        result = client.post("/api/v1/answer", json={"question": "Какие документы нужны?"})
    assert result.status_code == 200
    assert result.json() == fake.result.model_dump(mode="json")
    assert result.json()["answer"] == REFUSAL_MESSAGE
    assert result.json()["citations"] == []
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": None},
        {"question": ""},
        {"question": " \t\n"},
        {"question": 123},
        {"question": True},
        {"question": {}},
        {"question": []},
        {"question": "я" * 2001},
        {"question": " " + "я" * 2000},
        {"question": "Вопрос", "mode": "dense"},
        [],
        None,
    ],
)
def test_invalid_requests_never_call_service(payload):
    fake = FakeAnswerService()
    with TestClient(create_app(lambda: fake)) as client:
        result = client.post("/api/v1/answer", json=payload)
    assert result.status_code == 422
    assert fake.calls == []


def test_maximum_length_is_accepted():
    fake = FakeAnswerService()
    with TestClient(create_app(lambda: fake)) as client:
        result = client.post("/api/v1/answer", json={"question": "я" * 2000})
    assert result.status_code == 200
    assert fake.calls == ["я" * 2000]


def test_expected_failure_safe_response_and_data_free_logging(caplog):
    secret = "internal diagnostic text and document body"
    fake = FakeAnswerService(error=GenerationError(secret))
    with TestClient(create_app(lambda: fake)) as client:
        with caplog.at_level("ERROR", logger="app.api.routes.answer"):
            result = client.post("/api/v1/answer", json={"question": "private question"})
    assert result.status_code == 503
    assert result.json() == {"detail": UNAVAILABLE_MESSAGE}
    assert secret not in result.text
    assert fake.calls == ["private question"]
    assert "GenerationError" in caplog.text
    assert "frames=" in caplog.text
    assert secret not in caplog.text and "private question" not in caplog.text


def test_unexpected_error_is_not_swallowed():
    fake = FakeAnswerService(error=RuntimeError("programming defect"))
    with TestClient(create_app(lambda: fake)) as client:
        with pytest.raises(RuntimeError, match="programming defect"):
            client.post("/api/v1/answer", json={"question": "Вопрос"})
    assert len(fake.calls) == 1
    assert fake.closed == 1


def test_lifecycle_constructs_once_reuses_and_closes_once():
    created = []

    def factory():
        service = FakeAnswerService()
        created.append(service)
        return service

    application = create_app(factory)
    assert created == []
    with TestClient(application) as client:
        assert len(created) == 1 and created[0].calls == []
        assert client.get("/api/v1/health").json() == {
            "status": "ok",
            "service": "unecon-rag-assistant",
        }
        for _ in range(2):
            assert client.post("/api/v1/answer", json={"question": "Вопрос"}).status_code == 200
        assert created[0].calls == ["Вопрос", "Вопрос"]
        assert created[0].closed == 0
    assert len(created) == 1 and created[0].closed == 1


def test_default_service_startup_does_not_load_models_or_call_ollama(monkeypatch):
    import httpx

    from app.retrieval.service import RetrievalService

    def forbidden(*_args, **_kwargs):
        pytest.fail("startup attempted retrieval or external HTTP")

    monkeypatch.setattr(RetrievalService, "_session", forbidden)
    monkeypatch.setattr(httpx.Client, "post", forbidden)
    with TestClient(create_app()) as client:
        result = client.get("/api/v1/health")
    assert result.status_code == 200


def test_cors_preflight_and_origin_policy():
    with TestClient(create_app(FakeAnswerService)) as client:
        allowed = client.options(
            "/api/v1/answer",
            headers={
                "Origin": settings.frontend_origin,
                "Access-Control-Request-Method": "POST",
            },
        )
        denied = client.options(
            "/api/v1/answer",
            headers={
                "Origin": "https://not-allowed.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        wrong_method = client.options(
            "/api/v1/answer",
            headers={
                "Origin": settings.frontend_origin,
                "Access-Control-Request-Method": "DELETE",
            },
        )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == settings.frontend_origin
    assert set(allowed.headers["access-control-allow-methods"].split(", ")) == {"GET", "POST"}
    assert "access-control-allow-credentials" not in allowed.headers
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers
    assert wrong_method.status_code == 400


def test_openapi_reuses_canonical_answer_schema():
    schema = create_app(FakeAnswerService).openapi()
    operation = schema["paths"]["/api/v1/answer"]["post"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AnswerResponse",
    }
    request = schema["components"]["schemas"]["AnswerRequest"]
    assert set(request["properties"]) == {"question"}
    assert request["additionalProperties"] is False
    assert request["properties"]["question"]["maxLength"] == 2000
