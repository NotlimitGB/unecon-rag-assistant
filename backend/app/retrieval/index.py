"""Build, publish, validate, and search a local exact FAISS index."""

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.retrieval.corpus import RetrievalError, load_corpus
from app.retrieval.embeddings import DenseEmbedder, SentenceTransformerEmbedder

INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"
RECORD_KEYS = {
    "vector_id",
    "chunk_id",
    "ordinal",
    "text",
    "content_sha256",
    "source_id",
    "source_title",
    "url",
    "final_url",
    "source_type",
    "category",
    "admission_year",
    "page_start",
    "page_end",
}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _vectors(raw: Any, count: int, dimension: int | None = None) -> np.ndarray:
    try:
        vectors = np.ascontiguousarray(raw, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RetrievalError(f"invalid embedding matrix: {exc}") from exc
    if vectors.ndim != 2 or vectors.shape[0] != count or vectors.shape[1] < 1:
        raise RetrievalError("embedding matrix has invalid shape")
    if dimension is not None and vectors.shape[1] != dimension:
        raise RetrievalError("embedding dimension mismatch")
    if not np.isfinite(vectors).all():
        raise RetrievalError("embedding matrix has non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if (norms == 0).any() or not np.allclose(norms, 1, atol=1e-3, rtol=0):
        raise RetrievalError("embedding vectors must have unit L2 norm")
    return vectors


def _default_factory(model_name: str, device: str, batch_size: int) -> Callable[[], DenseEmbedder]:
    return lambda: SentenceTransformerEmbedder(model_name, device, batch_size)


def build_index(
    manifest_path: Path,
    chunks_dir: Path,
    index_dir: Path,
    model_name: str,
    device: str = "auto",
    batch_size: int = 16,
    embedder_factory: Callable[[], DenseEmbedder] | None = None,
) -> dict[str, Any]:
    if (
        not model_name.strip()
        or device not in {"auto", "cpu", "cuda"}
        or not 1 <= batch_size <= 128
    ):
        raise RetrievalError("invalid embedding configuration")
    fingerprints, records = load_corpus(manifest_path, chunks_dir)
    embedder = (embedder_factory or _default_factory(model_name, device, batch_size))()
    vectors = _vectors(
        embedder.encode_documents([record["text"] for record in records]), len(records)
    )
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    metadata = {
        "schema_version": 2,
        "embedding": {"model": model_name, "dimension": dimension, "normalized": True},
        "index": {
            "type": "IndexFlatIP",
            "metric": "inner_product",
            "vector_count": len(records),
            "file": INDEX_FILE,
            "file_sha256": "",
        },
        "corpus": {"sources": fingerprints},
        "records": records,
    }
    index_dir.mkdir(parents=True, exist_ok=True)
    index_temp: Path | None = None
    metadata_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=index_dir, suffix=".faiss", delete=False) as handle:
            index_temp = Path(handle.name)
        faiss.write_index(index, str(index_temp))
        metadata["index"]["file_sha256"] = _hash(index_temp.read_bytes())
        with tempfile.NamedTemporaryFile(
            dir=index_dir, suffix=".json", mode="w", encoding="utf-8", delete=False
        ) as handle:
            metadata_temp = Path(handle.name)
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(index_temp, index_dir / INDEX_FILE)
        index_temp = None
        os.replace(metadata_temp, index_dir / METADATA_FILE)
        metadata_temp = None
    finally:
        for path in (index_temp, metadata_temp):
            if path is not None:
                path.unlink(missing_ok=True)
    return metadata


def _validated_index(
    index_dir: Path,
    model_name: str,
) -> tuple[faiss.IndexFlatIP, dict[str, Any]]:
    try:
        with (index_dir / METADATA_FILE).open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict) or set(metadata) != {
            "schema_version",
            "embedding",
            "index",
            "corpus",
            "records",
        }:
            raise RetrievalError("invalid metadata schema")
        if type(metadata["schema_version"]) is not int or metadata["schema_version"] != 2:
            raise RetrievalError("invalid metadata version")
        embedding = metadata["embedding"]
        info = metadata["index"]
        corpus = metadata["corpus"]
        if not isinstance(embedding, dict) or set(embedding) != {
            "model",
            "dimension",
            "normalized",
        }:
            raise RetrievalError("invalid embedding metadata")
        if (
            embedding["model"] != model_name
            or embedding["normalized"] is not True
            or type(embedding["dimension"]) is not int
            or embedding["dimension"] < 1
        ):
            raise RetrievalError("configured embedding model or dimension mismatch")
        if not isinstance(info, dict) or set(info) != {
            "type",
            "metric",
            "vector_count",
            "file",
            "file_sha256",
        }:
            raise RetrievalError("invalid index metadata")
        if (info["type"], info["metric"], info["file"]) != (
            "IndexFlatIP",
            "inner_product",
            INDEX_FILE,
        ):
            raise RetrievalError("unsupported FAISS index")
        if not isinstance(metadata["records"], list) or not metadata["records"]:
            raise RetrievalError("invalid vector records")
        if type(info["vector_count"]) is not int or info["vector_count"] != len(
            metadata["records"]
        ):
            raise RetrievalError("vector count mismatch")
        if (
            not isinstance(corpus, dict)
            or set(corpus) != {"sources"}
            or not isinstance(corpus["sources"], list)
        ):
            raise RetrievalError("invalid corpus metadata")
        if any(
            not isinstance(source, dict)
            or set(source)
            != {
                "source_id",
                "logical_document_id",
                "version",
                "snapshot_sha256",
                "content_sha256",
                "file_sha256",
                "chunk_count",
            }
            or not isinstance(source["logical_document_id"], str)
            or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", source["logical_document_id"])
            or type(source["version"]) is not int
            or source["version"] < 1
            or not isinstance(source["snapshot_sha256"], str)
            or not HEX_SHA256.fullmatch(source["snapshot_sha256"])
            or not isinstance(source["source_id"], str)
            or not isinstance(source["content_sha256"], str)
            or not HEX_SHA256.fullmatch(source["content_sha256"])
            or (
                source["file_sha256"] is not None
                and (
                    not isinstance(source["file_sha256"], str)
                    or not HEX_SHA256.fullmatch(source["file_sha256"])
                )
            )
            or type(source["chunk_count"]) is not int
            or source["chunk_count"] < 1
            for source in corpus["sources"]
        ):
            raise RetrievalError("invalid source fingerprints")
        if any(
            not isinstance(record, dict)
            or set(record) != RECORD_KEYS
            or type(record["vector_id"]) is not int
            or record["vector_id"] != i
            or type(record["ordinal"]) is not int
            or record["ordinal"] < 1
            or type(record["admission_year"]) is not int
            or any(
                page is not None and (type(page) is not int or page < 1)
                for page in (record["page_start"], record["page_end"])
            )
            for i, record in enumerate(metadata["records"])
        ):
            raise RetrievalError("invalid vector records")
        index_bytes = (index_dir / INDEX_FILE).read_bytes()
        if not isinstance(info["file_sha256"], str) or info["file_sha256"] != _hash(index_bytes):
            raise RetrievalError("FAISS index hash mismatch")
        index = faiss.read_index(str(index_dir / INDEX_FILE))
        if (
            not isinstance(index, faiss.IndexFlatIP)
            or index.d != embedding["dimension"]
            or index.ntotal != len(metadata["records"])
        ):
            raise RetrievalError("FAISS index type, dimension, or vector count mismatch")
        return index, metadata
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, KeyError, RuntimeError) as exc:
        raise RetrievalError(f"invalid local index: {exc}") from exc


class RetrievalSession:
    """Reuse a validated index and one embedder across multiple queries."""

    def __init__(
        self,
        manifest_path: Path,
        chunks_dir: Path,
        index_dir: Path,
        model_name: str,
        device: str = "auto",
        batch_size: int = 16,
        embedder_factory: Callable[[], DenseEmbedder] | None = None,
    ):
        if (
            not model_name.strip()
            or device not in {"auto", "cpu", "cuda"}
            or not 1 <= batch_size <= 128
        ):
            raise RetrievalError("invalid embedding configuration")
        self.index, self.metadata = _validated_index(index_dir, model_name)
        fingerprints, self.records = load_corpus(manifest_path, chunks_dir)
        if self.metadata["corpus"]["sources"] != fingerprints:
            raise RetrievalError("index is stale relative to the source corpus")
        if self.metadata["records"] != self.records:
            raise RetrievalError("index records are stale relative to the chunks")
        self.embedder = (embedder_factory or _default_factory(model_name, device, batch_size))()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("query must not be empty")
        if type(top_k) is not int or top_k < 1:
            raise RetrievalError("top_k must be a positive integer")
        raw_query = np.asarray(self.embedder.encode_query(query))
        if raw_query.ndim != 1:
            raise RetrievalError("query embedding must be a one-dimensional vector")
        query_vector = _vectors(raw_query.reshape(1, -1), 1, self.index.d)
        scores, ids = self.index.search(query_vector, self.index.ntotal)
        ranked = sorted(
            zip(scores[0].tolist(), ids[0].tolist(), strict=True),
            key=lambda pair: (-pair[0], pair[1]),
        )
        return [
            {**self.records[vector_id], "score": float(score)}
            for score, vector_id in ranked[:top_k]
        ]


def search(
    query: str,
    top_k: int,
    manifest_path: Path,
    chunks_dir: Path,
    index_dir: Path,
    model_name: str,
    device: str = "auto",
    batch_size: int = 16,
    embedder_factory: Callable[[], DenseEmbedder] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(query, str) or not query.strip():
        raise RetrievalError("query must not be empty")
    if type(top_k) is not int or top_k < 1:
        raise RetrievalError("top_k must be a positive integer")
    session = RetrievalSession(
        manifest_path, chunks_dir, index_dir, model_name, device, batch_size, embedder_factory
    )
    return session.search(query, top_k)
