import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from app.chunking.core import build_chunk_artifact
from app.config import Settings
from app.ingestion.models import Source
from app.retrieval.cli import main
from app.retrieval.corpus import RetrievalError, load_corpus
from app.retrieval.index import build_index, search


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


HTML = Source(
    id="faq", title="Вопросы", url="https://unecon.ru/faq/", source_type="html",
    category="faq", admission_year=2026, active=True,
)
PDF = Source(
    id="rules", title="Правила", url="https://unecon.ru/rules.pdf", source_type="pdf",
    category="rules", admission_year=2026, active=True,
)


def normalized(source: Source, pages: list[str]) -> dict:
    text = pages[0] if source.source_type == "html" else "\n\n\f\n\n".join(p for p in pages if p)
    document = {"title": source.title, "text": text, "content_sha256": digest(text)}
    if source.source_type == "pdf":
        document.update({
            "page_count": len(pages),
            "pages": [{"page_number": i, "text": p} for i, p in enumerate(pages, 1)],
            "file_sha256": digest("pdf bytes"),
        })
    return {
        "schema_version": 1,
        "source": {**source.model_dump(exclude={"active"}), "final_url": source.url},
        "document": document,
    }


@pytest.fixture
def corpus(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    manifest.write_text(json.dumps({
        "schema_version": 1, "sources": [HTML.model_dump(), PDF.model_dump()]
    }), encoding="utf-8")
    artifacts = {
        "faq": build_chunk_artifact(HTML, normalized(HTML, ["Ответ про поступление."])),
        "rules": build_chunk_artifact(PDF, normalized(PDF, ["", "Правила приема.", "Стоимость."])),
    }
    for source_id, artifact in artifacts.items():
        (chunks_dir / f"{source_id}.json").write_text(
            json.dumps(artifact, ensure_ascii=False), encoding="utf-8"
        )
    return manifest, chunks_dir, tmp_path / "index", artifacts


class FakeEmbedder:
    def __init__(self, vectors=None):
        self.vectors = vectors
        self.document_calls = 0
        self.query_calls = 0

    def encode_documents(self, texts):
        self.document_calls += 1
        return self.vectors if self.vectors is not None else np.eye(len(texts), dtype=np.float32)

    def encode_query(self, text):
        self.query_calls += 1
        return np.array([0, 1, 0], dtype=np.float32)


def build(corpus, embedder=None):
    manifest, chunks_dir, index_dir, _ = corpus
    return build_index(
        manifest, chunks_dir, index_dir, "fake", embedder_factory=lambda: embedder or FakeEmbedder()
    )


def query(corpus, embedder=None, text="правила", top_k=5):
    manifest, chunks_dir, index_dir, _ = corpus
    return search(
        text, top_k, manifest, chunks_dir, index_dir, "fake",
        embedder_factory=lambda: embedder or FakeEmbedder(),
    )


def test_build_search_provenance_and_rebuild(corpus):
    fingerprints, records = load_corpus(corpus[0], corpus[1])
    assert len(fingerprints) == 2
    assert [record["page_start"] for record in records] == [None, 2, 3]
    assert [record["vector_id"] for record in records] == [0, 1, 2]
    metadata = build(corpus)
    assert metadata["embedding"] == {"model": "fake", "dimension": 3, "normalized": True}
    assert metadata["index"]["vector_count"] == 3
    assert metadata["index"]["file_sha256"] == hashlib.sha256(
        (corpus[2] / "index.faiss").read_bytes()
    ).hexdigest()
    assert query(corpus)[0]["source_id"] == "rules"
    assert query(corpus)[0]["page_start"] == 2
    assert len(query(corpus, top_k=10)) == 3
    assert build(corpus)["records"] == metadata["records"]


@pytest.mark.parametrize("mutation", [
    lambda a: a["chunks"][0].update(text="tampered"),
    lambda a: a["chunks"][0].update(chunk_id="bad"),
    lambda a: a["chunks"][0].update(content_sha256="0" * 64),
    lambda a: a["source"].update(url="https://example.com/"),
    lambda a: a["source"].update(content_sha256="0" * 64),
    lambda a: a["chunks"][0].update(page_start=1),
])
def test_invalid_chunks_fail_before_model(corpus, mutation):
    artifact = copy.deepcopy(corpus[3]["rules"])
    mutation(artifact)
    (corpus[1] / "rules.json").write_text(json.dumps(artifact), encoding="utf-8")
    calls = []
    with pytest.raises(RetrievalError):
        build_index(
            corpus[0], corpus[1], corpus[2], "fake", embedder_factory=lambda: calls.append(1)
        )
    assert calls == []


def test_missing_and_manifest_mismatch_fail_before_model(corpus):
    (corpus[1] / "rules.json").unlink()
    with pytest.raises(RetrievalError):
        build(corpus)
    (corpus[1] / "rules.json").write_text(json.dumps(corpus[3]["rules"]), encoding="utf-8")
    manifest = json.loads(corpus[0].read_text(encoding="utf-8"))
    manifest["sources"][1]["title"] = "Изменено"
    corpus[0].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RetrievalError):
        build(corpus)


@pytest.mark.parametrize("vectors", [
    np.array([1, 0, 0], dtype=np.float32),
    np.ones((2, 3), dtype=np.float32),
    np.zeros((3, 3), dtype=np.float32),
    np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]]),
    np.array([[np.inf, 0, 0], [0, 1, 0], [0, 0, 1]]),
    np.ones((3, 3), dtype=np.float32),
    np.empty((3, 0), dtype=np.float32),
])
def test_bad_document_vectors_rejected(corpus, vectors):
    with pytest.raises(RetrievalError):
        build(corpus, FakeEmbedder(vectors))


def test_stale_corpus_and_corrupt_index_fail_before_query_encoding(corpus):
    build(corpus)
    artifact = copy.deepcopy(corpus[3]["faq"])
    artifact = build_chunk_artifact(HTML, normalized(HTML, ["Другой ответ."]))
    (corpus[1] / "faq.json").write_text(json.dumps(artifact), encoding="utf-8")
    embedder = FakeEmbedder()
    with pytest.raises(RetrievalError):
        query(corpus, embedder)
    assert embedder.query_calls == 0
    (corpus[1] / "faq.json").write_text(json.dumps(corpus[3]["faq"]), encoding="utf-8")
    (corpus[2] / "index.faiss").write_bytes(b"bad")
    with pytest.raises(RetrievalError):
        query(corpus, embedder)
    assert embedder.query_calls == 0


def test_model_mismatch_bad_query_and_bad_metadata(corpus):
    build(corpus)
    with pytest.raises(RetrievalError):
        query(corpus, text=" ")
    with pytest.raises(RetrievalError):
        query(corpus, top_k=0)
    with pytest.raises(RetrievalError):
        search("q", 1, corpus[0], corpus[1], corpus[2], "other", embedder_factory=FakeEmbedder)
    metadata_path = corpus[2] / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["records"][0]["text"] = "changed"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RetrievalError):
        query(corpus)


def test_tied_scores_are_ordered_by_vector_id(corpus):
    vectors = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    build(corpus, FakeEmbedder(vectors))
    hits = query(corpus)
    assert [hit["vector_id"] for hit in hits] == [0, 1, 2]
    assert hits[0]["score"] == 1.0


def test_provenance_change_with_unchanged_fingerprint_is_stale(corpus):
    build(corpus)
    artifact = copy.deepcopy(corpus[3]["faq"])
    artifact["source"]["final_url"] = "https://unecon.ru/faq/updated/"
    (corpus[1] / "faq.json").write_text(json.dumps(artifact), encoding="utf-8")
    embedder = FakeEmbedder()
    with pytest.raises(RetrievalError):
        query(corpus, embedder)
    assert embedder.query_calls == 0


def test_invalid_query_vector_rejected(corpus):
    build(corpus)

    class BadQuery(FakeEmbedder):
        def encode_query(self, text):
            return np.ones((1, 3), dtype=np.float32)

    with pytest.raises(RetrievalError):
        query(corpus, BadQuery())


def test_failed_metadata_publish_leaves_detectable_pair(corpus, monkeypatch):
    build(corpus)
    original_replace = os.replace

    def fail_metadata_publish(source, destination):
        if Path(destination).name == "metadata.json":
            raise OSError("synthetic publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr("app.retrieval.index.os.replace", fail_metadata_publish)
    changed = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    with pytest.raises(OSError):
        build(corpus, FakeEmbedder(changed))
    assert not list(corpus[2].glob("tmp*"))
    with pytest.raises(RetrievalError, match="hash mismatch"):
        query(corpus)


@pytest.mark.parametrize("values", [
    {"EMBEDDING_MODEL": " "},
    {"EMBEDDING_DEVICE": "other"},
    {"EMBEDDING_BATCH_SIZE": 0},
    {"EMBEDDING_BATCH_SIZE": 129},
])
def test_invalid_embedding_settings(values):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **values)


def test_cli_build_and_search_output(corpus, monkeypatch, capsys):
    monkeypatch.setattr("app.retrieval.cli.build_index", lambda *args: {
        "index": {"vector_count": 3},
        "embedding": {"dimension": 3},
        "corpus": {"sources": [{}, {}]},
    })
    assert main(["build-index"]) == 0
    assert "vectors=3 dimension=3 sources=2" in capsys.readouterr().out
    monkeypatch.setattr("app.retrieval.cli.search", lambda *args: [{
        "chunk_id": "faq:0001:abc", "score": 0.5, "page_start": None,
        "source_id": "faq", "final_url": "https://unecon.ru/faq/", "text": "Текст ответа",
    }])
    assert main(["search", "вопрос", "--top-k", "5"]) == 0
    assert "score=0.500000 page=- source=faq" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["search", "вопрос", "--top-k", "51"])


def test_metadata_rejects_boolean_page_number(corpus):
    build(corpus)
    path = corpus[2] / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["records"][1]["page_start"] = True
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RetrievalError, match="invalid vector records"):
        query(corpus)
