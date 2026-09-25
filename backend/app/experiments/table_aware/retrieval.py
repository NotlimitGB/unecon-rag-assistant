"""An isolated FAISS index of table rows using the accepted model wrappers."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.experiments.table_aware.extract import ExperimentError
from app.retrieval.embeddings import DenseEmbedder
from app.retrieval.index import _vectors
from app.retrieval.reranker import Reranker, rerank_candidates


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_row_index(
    artifacts: list[dict[str, Any]], embedder: DenseEmbedder
) -> tuple[Any, list[dict[str, Any]], bytes]:
    records = []
    for artifact in artifacts:
        for table in artifact["tables"]:
            for row in table["rows"]:
                records.append(
                    {
                        "vector_id": len(records),
                        "chunk_id": row["row_id"],
                        "source_id": artifact["source"]["id"],
                        "page_start": table["page"],
                        "text": row["text"],
                    }
                )
    if not records or len({record["chunk_id"] for record in records}) != len(records):
        raise ExperimentError("empty or duplicate table-row corpus")
    vectors = _vectors(
        embedder.encode_documents([record["text"] for record in records]), len(records)
    )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, records, faiss.serialize_index(index).tobytes()


def save_row_index(
    output_dir: Path, index_bytes: bytes, records: list[dict[str, Any]], model: str
) -> None:
    metadata = {
        "schema_version": 1,
        "experiment_id": "table-aware-pdf-v1",
        "model": model,
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "records": records,
    }
    _atomic_bytes(output_dir / "index.faiss", index_bytes)
    _atomic_bytes(
        output_dir / "metadata.json",
        (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def dense_search(
    index: Any,
    records: list[dict[str, Any]],
    embedder: DenseEmbedder,
    question: str,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    query = _vectors(np.asarray(embedder.encode_query(question)).reshape(1, -1), 1, index.d)
    scores, ids = index.search(query, index.ntotal)
    ranked = sorted(
        zip(scores[0].tolist(), ids[0].tolist(), strict=True), key=lambda item: (-item[0], item[1])
    )
    return [{**records[vector_id], "score": float(score)} for score, vector_id in ranked[:top_k]]


def reranked_search(
    question: str, dense_hits: list[dict[str, Any]], reranker: Reranker, top_k: int = 5
) -> list[dict[str, Any]]:
    return rerank_candidates(question, dense_hits, reranker, min(top_k, len(dense_hits)))
