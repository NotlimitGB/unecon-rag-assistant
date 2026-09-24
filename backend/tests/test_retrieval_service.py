"""Offline checks for the canonical application retrieval boundary."""

import json

import pytest

from app.config import Settings
from app.retrieval.cli import main
from app.retrieval.corpus import RetrievalError
from app.retrieval.service import RetrievalResponse, RetrievalService


def hit(**changes):
    record = {
        "vector_id": 12,
        "chunk_id": "rules:0001:abcdef123456",
        "text": "Точный текст чанка.\nВторая строка.",
        "source_id": "rules",
        "source_title": "Правила приема",
        "final_url": "https://unecon.ru/rules.pdf",
        "url": "https://unecon.ru/original.pdf",
        "source_type": "pdf",
        "category": "admission_rules",
        "admission_year": 2026,
        "page_start": 8,
        "score": 0.7,
    }
    record.update(changes)
    return record


class FakeSession:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        if isinstance(self.results, Exception):
            raise self.results
        return self.results[:top_k]


def config(**values):
    return Settings(_env_file=None).model_copy(update=values)


def test_dense_default_top_k_provenance_lazy_reuse():
    dense = FakeSession(
        [hit(), hit(chunk_id="faq:0001:abcdef123456", source_type="html", page_start=None)]
    )
    created = []
    rerank_created = []
    service = RetrievalService(
        config=config(retrieval_mode="dense"),
        dense_factory=lambda: created.append(dense) or dense,
        reranked_factory=lambda session: rerank_created.append(session),
    )
    assert created == rerank_created == []
    first = service.retrieve("  Сроки приема?  ")
    second = service.retrieve("Сроки приема?", 1)
    assert created == [dense]
    assert rerank_created == []
    assert dense.calls == [("Сроки приема?", 5), ("Сроки приема?", 1)]
    assert first.mode == "dense" and first.top_k == 5 and first.query == "Сроки приема?"
    assert [part.chunk_id for part in first.results] == [
        "rules:0001:abcdef123456",
        "faq:0001:abcdef123456",
    ]
    assert [part.rank for part in first.results] == [1, 2]
    assert first.results[0].text == hit()["text"]
    assert first.results[0].source_url == hit()["final_url"]
    assert first.results[0].page == 8 and first.results[1].page is None
    assert first.results[0].score == first.results[0].dense_score == 0.7
    assert first.results[0].rerank_score is None
    assert "vector_id" not in first.model_dump_json()
    assert len(second.results) == 1


def test_reranked_default_order_scores_and_reuse():
    dense = FakeSession([])
    reranked = FakeSession(
        [
            hit(chunk_id="second", score=3.0, dense_score=0.4, rerank_score=3.0),
            hit(chunk_id="first", score=2.0, dense_score=0.9, rerank_score=2.0),
        ]
    )
    dense_created = []
    rerank_created = []
    service = RetrievalService(
        config=config(),
        dense_factory=lambda: dense_created.append(dense) or dense,
        reranked_factory=lambda session: rerank_created.append(session) or reranked,
    )
    assert dense_created == rerank_created == []
    response = service.retrieve(" Экзамены? ")
    service.retrieve("Экзамены?", 1)
    assert response.mode == "reranked" and response.top_k == 5
    assert [part.chunk_id for part in response.results] == ["second", "first"]
    assert response.results[0].score == response.results[0].rerank_score == 3.0
    assert response.results[0].dense_score == 0.4
    assert dense_created == [dense] and rerank_created == [dense]
    assert reranked.calls == [("Экзамены?", 5), ("Экзамены?", 1)]
    assert dense.calls == []


@pytest.mark.parametrize("failure", ["init", "score"])
def test_reranked_never_falls_back(failure):
    dense = FakeSession([hit()])
    if failure == "init":

        def fail_factory(_session):
            raise RuntimeError("model unavailable")
    else:

        def fail_factory(_session):
            return FakeSession(RuntimeError("scoring failed"))

    service = RetrievalService(dense_factory=lambda: dense, reranked_factory=fail_factory)
    with pytest.raises(RetrievalError, match="reranked"):
        service.retrieve("Экзамены?")
    assert dense.calls == []
    explicit_dense = RetrievalService(
        mode="dense",
        dense_factory=lambda: dense,
        reranked_factory=lambda _session: pytest.fail("reranker instantiated"),
    )
    assert explicit_dense.retrieve("Экзамены?").mode == "dense"


@pytest.mark.parametrize(
    "question,top_k",
    [
        ("", None),
        (" \t ", None),
        (None, None),
        (3, None),
        ("question", 0),
        ("question", -1),
        ("question", True),
        ("question", 21),
        ("question", 6),
    ],
)
def test_bad_requests_fail_before_session_creation(question, top_k):
    created = []
    service = RetrievalService(
        config=config(reranker_candidate_k=5),
        dense_factory=lambda: created.append(1),
    )
    with pytest.raises(RetrievalError):
        service.retrieve(question, top_k)
    assert created == []


@pytest.mark.parametrize(
    "change",
    [
        {"text": " "},
        {"chunk_id": ""},
        {"source_id": " "},
        {"source_title": ""},
        {"final_url": "http://unecon.ru/rules.pdf"},
        {"final_url": "https://example.com/rules.pdf"},
        {"source_type": "pdf", "page_start": None},
        {"source_type": "html", "page_start": 2},
        {"source_type": "other"},
        {"admission_year": 0},
        {"score": float("nan")},
        {"score": float("inf")},
    ],
)
def test_invalid_internal_provenance_and_scores(change):
    service = RetrievalService(mode="dense", dense_factory=lambda: FakeSession([hit(**change)]))
    with pytest.raises(RetrievalError, match="invalid dense retrieval result"):
        service.retrieve("Правила?")


@pytest.mark.parametrize("missing", ["dense_score", "rerank_score"])
def test_reranked_missing_score_rejected(missing):
    record = hit(score=2.0, dense_score=0.7, rerank_score=2.0)
    del record[missing]
    service = RetrievalService(
        dense_factory=lambda: FakeSession([]),
        reranked_factory=lambda _session: FakeSession([record]),
    )
    with pytest.raises(RetrievalError):
        service.retrieve("Правила?")


def test_response_json_contract_and_settings():
    response = RetrievalService(mode="dense", dense_factory=lambda: FakeSession([hit()])).retrieve(
        "Правила?"
    )
    data = json.loads(response.model_dump_json())
    assert set(data) == {"query", "mode", "top_k", "results"}
    assert set(data["results"][0]) == {
        "rank",
        "chunk_id",
        "text",
        "source_id",
        "source_title",
        "source_url",
        "source_type",
        "category",
        "admission_year",
        "page",
        "score",
        "dense_score",
        "rerank_score",
    }
    assert data["results"][0]["rerank_score"] is None
    assert not any(
        key in response.model_dump_json()
        for key in ("vector_id", "timestamp", "cache", "C:\\", "D:\\")
    )
    assert RetrievalResponse.model_validate(data) == response
    assert Settings(_env_file=None).retrieval_mode == "reranked"
    assert Settings(_env_file=None).retrieval_top_k == 5
    assert Settings(_env_file=None, RETRIEVAL_TOP_K="5").retrieval_top_k == 5
    for values in (
        {"RETRIEVAL_MODE": "other"},
        {"RETRIEVAL_TOP_K": 0},
        {"RETRIEVAL_TOP_K": 21},
        {"RETRIEVAL_TOP_K": True},
        {"RETRIEVAL_TOP_K": 5.0},
        {"RETRIEVAL_TOP_K": "5.5"},
    ):
        with pytest.raises(ValueError):
            Settings(_env_file=None, **values)


def test_canonical_cli_human_json_and_mode_override(monkeypatch, capsys):
    calls = []

    class FakeService:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def retrieve(self, question, top_k):
            mode = calls[-1]["mode"] or "reranked"
            record = hit(
                score=2.0 if mode == "reranked" else 0.7,
                dense_score=0.7,
                rerank_score=2.0 if mode == "reranked" else None,
            )
            return RetrievalResponse(
                query=question,
                mode=mode,
                top_k=top_k or 5,
                results=[
                    {
                        "rank": 1,
                        "chunk_id": record["chunk_id"],
                        "text": record["text"],
                        "source_id": record["source_id"],
                        "source_title": record["source_title"],
                        "source_url": record["final_url"],
                        "source_type": record["source_type"],
                        "category": record["category"],
                        "admission_year": record["admission_year"],
                        "page": record["page_start"],
                        "score": record["score"],
                        "dense_score": record["dense_score"],
                        "rerank_score": record["rerank_score"],
                    }
                ],
            )

    monkeypatch.setattr("app.retrieval.cli.RetrievalService", FakeService)
    assert main(["retrieve", "Экзамены?"]) == 0
    assert "mode=reranked top_k=5" in capsys.readouterr().out
    assert calls[-1]["mode"] is None
    assert main(["retrieve", "Экзамены?", "--mode", "dense", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["mode"] == "dense" and "vector_id" not in data["results"][0]
    assert calls[-1]["mode"] == "dense"
    assert main(["retrieve", "Экзамены?", "--mode", "reranked", "--top-k", "2"]) == 0
    assert "mode=reranked top_k=2" in capsys.readouterr().out
    assert calls[-1]["mode"] == "reranked"


def test_cli_json_failure_is_not_mixed_with_stdout(monkeypatch, capsys):
    class FailedService:
        def __init__(self, **_kwargs):
            pass

        def retrieve(self, _question, _top_k):
            raise RetrievalError("model unavailable")

    monkeypatch.setattr("app.retrieval.cli.RetrievalService", FailedService)
    assert main(["retrieve", "question", "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "model unavailable" in captured.err
