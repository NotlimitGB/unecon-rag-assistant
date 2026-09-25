"""Offline construction and strict validation of the approved PDF table channel."""

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import faiss
import httpx
import pymupdf

from app.chunking.core import validate_normalized_document
from app.ingestion.fetcher import fetch_pdf
from app.ingestion.manifest import load_manifest
from app.retrieval.corpus import RetrievalError, load_corpus
from app.retrieval.embeddings import DenseEmbedder
from app.retrieval.index import _default_factory, _validated_index, _vectors
from app.retrieval.table_extract import STRATEGY, extract_pdf_tables, row_id

# Only these PDFs participated in the accepted Task014–018 evaluation.
TABLE_SOURCE_IDS = (
    "admission-capacity-pdf",
    "entrance-exams-list-pdf",
    "tuition-order-128-pdf",
)
ARCHITECTURE = "dual-channel-pdf-page-diversity-v1"
RECORD_KEYS = {
    "vector_id",
    "chunk_id",
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
    "table_index",
    "row_index",
}
SHA = re.compile(r"[0-9a-f]{64}")


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sources(manifest_path: Path) -> list[Any]:
    manifest = load_manifest(manifest_path)
    by_id = {source.id: source for source in manifest.sources}
    if any(
        source_id not in by_id
        or not by_id[source_id].active
        or by_id[source_id].source_type != "pdf"
        or by_id[source_id].admission_year != 2026
        for source_id in TABLE_SOURCE_IDS
    ):
        raise RetrievalError("approved table PDF sources are missing, inactive, or changed")
    return [by_id[source_id] for source_id in TABLE_SOURCE_IDS]


def _normalized(source: Any, pdf_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        artifact = json.loads((pdf_root / f"{source.id}.json").read_text(encoding="utf-8"))
        document = validate_normalized_document(source, artifact)
        return artifact, document
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        raise RetrievalError(f"invalid normalized PDF for {source.id}: {exc}") from exc


def _production(manifest_path: Path, chunks_dir: Path, index_dir: Path, model: str):
    index, metadata = _validated_index(index_dir, model)
    fingerprints, records = load_corpus(manifest_path, chunks_dir)
    if metadata["corpus"]["sources"] != fingerprints or metadata["records"] != records:
        raise RetrievalError("production index is stale relative to the chunk corpus")
    return index, metadata


def _records(
    artifacts: list[dict[str, Any]], sources: list[Any], offset: int
) -> list[dict[str, Any]]:
    records = []
    for source, artifact in zip(sources, artifacts, strict=True):
        meta = artifact["source"]
        for table in artifact["tables"]:
            for row in table["rows"]:
                records.append(
                    {
                        "vector_id": offset + len(records),
                        "chunk_id": row["row_id"],
                        "text": row["text"],
                        "content_sha256": row["content_sha256"],
                        "source_id": source.id,
                        "source_title": source.title,
                        "url": source.url,
                        "final_url": meta["final_url"],
                        "source_type": "pdf",
                        "category": source.category,
                        "admission_year": source.admission_year,
                        "page_start": row["page"],
                        "page_end": row["page"],
                        "table_index": row["table_index"],
                        "row_index": row["row_index"],
                    }
                )
    if not records or len({r["chunk_id"] for r in records}) != len(records):
        raise RetrievalError("empty or duplicate PDF table rows")
    return records


def build_table_index(
    manifest_path: Path,
    chunks_dir: Path,
    production_dir: Path,
    pdf_root: Path,
    table_dir: Path,
    model_name: str,
    device: str = "auto",
    batch_size: int = 16,
    embedder_factory: Callable[[], DenseEmbedder] | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch verified PDFs once, then embed rows and publish the index pair."""
    if (
        not model_name.strip()
        or device not in {"auto", "cpu", "cuda"}
        or not 1 <= batch_size <= 128
    ):
        raise RetrievalError("invalid embedding configuration")
    production, production_meta = _production(manifest_path, chunks_dir, production_dir, model_name)
    sources = _sources(manifest_path)
    originals = [_normalized(source, pdf_root) for source in sources]
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=60)
    try:
        fetched = [fetch_pdf(source, client) for source in sources]
    finally:
        if own_client:
            client.close()
    for source, (normalized, document), pdf in zip(sources, originals, fetched, strict=True):
        if _digest(pdf.content) != document["file_sha256"]:
            raise RetrievalError(
                f"official PDF changed for {source.id}; rerun ingestion, "
                "chunking, and index preparation"
            )
        if pdf.final_url != normalized["source"]["final_url"]:
            raise RetrievalError(f"official PDF final URL changed for {source.id}")
    artifacts = [
        extract_pdf_tables(source, pdf.final_url, pdf.content)
        for source, pdf in zip(sources, fetched, strict=True)
    ]
    records = _records(artifacts, sources, len(production_meta["records"]))
    if {record["chunk_id"] for record in records} & {
        record["chunk_id"] for record in production_meta["records"]
    }:
        raise RetrievalError("table row ID conflicts with the production corpus")
    embedder = (embedder_factory or _default_factory(model_name, device, batch_size))()
    vectors = _vectors(
        embedder.encode_documents([r["text"] for r in records]), len(records), production.d
    )
    index = faiss.IndexFlatIP(production.d)
    index.add(vectors)
    metadata = {
        "schema_version": 1,
        "architecture": ARCHITECTURE,
        "embedding": {"model": model_name, "dimension": production.d, "normalized": True},
        "index": {
            "type": "IndexFlatIP",
            "metric": "inner_product",
            "vector_count": len(records),
            "file": "index.faiss",
            "file_sha256": "",
        },
        "production": {
            "index_sha256": production_meta["index"]["file_sha256"],
            "record_count": len(production_meta["records"]),
        },
        "sources": [
            {
                "source_id": source.id,
                "file_sha256": document["file_sha256"],
                "content_sha256": document["content_sha256"],
                "final_url": normalized["source"]["final_url"],
                "row_count": sum(len(t["rows"]) for t in artifact["tables"]),
                "page_count": document["page_count"],
            }
            for source, (normalized, document), artifact in zip(
                sources, originals, artifacts, strict=True
            )
        ],
        "extraction": artifacts[0]["extraction"],
        "records": records,
    }
    table_dir.mkdir(parents=True, exist_ok=True)
    index_temp = metadata_temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=table_dir, suffix=".faiss", delete=False) as handle:
            index_temp = Path(handle.name)
        faiss.write_index(index, str(index_temp))
        metadata["index"]["file_sha256"] = _digest(index_temp.read_bytes())
        with tempfile.NamedTemporaryFile(
            dir=table_dir, suffix=".json", mode="w", encoding="utf-8", delete=False
        ) as handle:
            metadata_temp = Path(handle.name)
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(index_temp, table_dir / "index.faiss")
        index_temp = None
        os.replace(metadata_temp, table_dir / "metadata.json")
        metadata_temp = None
    finally:
        for path in (index_temp, metadata_temp):
            if path is not None:
                path.unlink(missing_ok=True)
    return metadata


def validated_table_index(
    manifest_path: Path,
    pdf_root: Path,
    table_dir: Path,
    production_meta: dict[str, Any],
    model_name: str,
) -> tuple[faiss.IndexFlatIP, list[dict[str, Any]]]:
    """Use only local files; reject an index built for another corpus or PDF version."""
    try:
        meta = json.loads((table_dir / "metadata.json").read_text(encoding="utf-8"))
        if (
            not isinstance(meta, dict)
            or set(meta)
            != {
                "schema_version",
                "architecture",
                "embedding",
                "index",
                "production",
                "sources",
                "extraction",
                "records",
            }
            or type(meta["schema_version"]) is not int
            or meta["schema_version"] != 1
            or meta["architecture"] != ARCHITECTURE
        ):
            raise RetrievalError("invalid table index metadata schema")
        embedding, info = meta["embedding"], meta["index"]
        if (
            not isinstance(embedding, dict)
            or set(embedding) != {"model", "dimension", "normalized"}
            or (
                embedding["model"] != model_name
                or embedding["dimension"] != production_meta["embedding"]["dimension"]
                or embedding["normalized"] is not True
            )
        ):
            raise RetrievalError("table index embedding model or dimension mismatch")
        if (
            not isinstance(info, dict)
            or set(info) != {"type", "metric", "vector_count", "file", "file_sha256"}
            or (info["type"], info["metric"], info["file"])
            != ("IndexFlatIP", "inner_product", "index.faiss")
        ):
            raise RetrievalError("invalid table FAISS metadata")
        if meta["production"] != {
            "index_sha256": production_meta["index"]["file_sha256"],
            "record_count": len(production_meta["records"]),
        }:
            raise RetrievalError("table index is stale against production index")
        if (
            not isinstance(meta["extraction"], dict)
            or set(meta["extraction"]) != {"library", "version", "strategy"}
            or (
                meta["extraction"]["library"] != "PyMuPDF"
                or meta["extraction"]["strategy"] != STRATEGY
                or meta["extraction"]["version"] != pymupdf.VersionBind
            )
        ):
            raise RetrievalError("invalid table extraction metadata")
        sources = _sources(manifest_path)
        expected = []
        for source in sources:
            normalized, document = _normalized(source, pdf_root)
            expected.append((source, normalized, document))
        source_meta = meta["sources"]
        if not isinstance(source_meta, list) or len(source_meta) != len(expected):
            raise RetrievalError("table index approved sources mismatch")
        for saved, (source, normalized, document) in zip(source_meta, expected, strict=True):
            if (
                not isinstance(saved, dict)
                or set(saved)
                != {
                    "source_id",
                    "file_sha256",
                    "content_sha256",
                    "final_url",
                    "row_count",
                    "page_count",
                }
                or any(
                    (
                        saved["source_id"] != source.id,
                        saved["file_sha256"] != document["file_sha256"],
                        saved["content_sha256"] != document["content_sha256"],
                        saved["final_url"] != normalized["source"]["final_url"],
                        saved["page_count"] != document["page_count"],
                        type(saved["row_count"]) is not int or saved["row_count"] < 1,
                    )
                )
            ):
                raise RetrievalError(f"table source artifact is stale or invalid: {source.id}")
        records = meta["records"]
        if (
            not isinstance(records, list)
            or not records
            or type(info["vector_count"]) is not int
            or info["vector_count"] != len(records)
        ):
            raise RetrievalError("table vector count mismatch")
        if sum(source["row_count"] for source in source_meta) != len(records):
            raise RetrievalError("table source row counts mismatch")
        seen: set[str] = set()
        production_ids = {record["chunk_id"] for record in production_meta["records"]}
        source_by_id = {
            source.id: (source, saved) for source, saved in zip(sources, source_meta, strict=True)
        }
        source_positions = {source.id: position for position, source in enumerate(sources)}
        counts = {source.id: 0 for source in sources}
        previous_position = -1
        previous_location: tuple[int, int] | None = None
        previous_row = 0
        for position, record in enumerate(records):
            if not isinstance(record, dict) or set(record) != RECORD_KEYS:
                raise RetrievalError("invalid table vector record schema")
            if (
                type(record["vector_id"]) is not int
                or record["vector_id"] != len(production_meta["records"]) + position
            ):
                raise RetrievalError("duplicate or nonsequential table vector ID")
            source_id = record["source_id"]
            if source_id not in source_by_id:
                raise RetrievalError("unexpected table source")
            source, saved = source_by_id[source_id]
            source_position = source_positions[source_id]
            if source_position < previous_position:
                raise RetrievalError("table source record order mismatch")
            if source_position != previous_position:
                previous_location = None
                previous_row = 0
            previous_position = source_position
            page = record["page_start"]
            if (
                type(page) is not int
                or not 1 <= page <= saved["page_count"]
                or record["page_end"] != page
            ):
                raise RetrievalError("invalid table PDF page")
            if (
                type(record["table_index"]) is not int
                or record["table_index"] < 1
                or (type(record["row_index"]) is not int or record["row_index"] < 1)
            ):
                raise RetrievalError("invalid table row provenance")
            location = (page, record["table_index"])
            if previous_location is not None and location < previous_location:
                raise RetrievalError("table row order mismatch")
            expected_row = previous_row + 1 if location == previous_location else 1
            if record["row_index"] != expected_row:
                raise RetrievalError("nonsequential table row")
            previous_location = location
            previous_row = record["row_index"]
            text = record["text"]
            if (
                not isinstance(text, str)
                or not text
                or record["content_sha256"] != _digest(text.encode("utf-8"))
            ):
                raise RetrievalError("table row text or hash mismatch")
            if (
                record["chunk_id"]
                != row_id(
                    source_id,
                    saved["file_sha256"],
                    page,
                    record["table_index"],
                    record["row_index"],
                    text,
                )
                or record["chunk_id"] in seen
                or record["chunk_id"] in production_ids
            ):
                raise RetrievalError("duplicate or invalid table row ID")
            seen.add(record["chunk_id"])
            if any(
                (
                    record["source_title"] != source.title,
                    record["url"] != source.url,
                    record["final_url"] != saved["final_url"],
                    record["source_type"] != "pdf",
                    record["category"] != source.category,
                    record["admission_year"] != source.admission_year,
                )
            ):
                raise RetrievalError("table row source provenance mismatch")
            counts[source_id] += 1
        if any(
            counts[source.id] != saved["row_count"]
            for source, saved in zip(sources, source_meta, strict=True)
        ):
            raise RetrievalError("table row source counts mismatch")
        content = (table_dir / "index.faiss").read_bytes()
        if (
            not isinstance(info["file_sha256"], str)
            or not SHA.fullmatch(info["file_sha256"])
            or _digest(content) != info["file_sha256"]
        ):
            raise RetrievalError("table FAISS index hash mismatch")
        index = faiss.read_index(str(table_dir / "index.faiss"))
        if (
            not isinstance(index, faiss.IndexFlatIP)
            or index.metric_type != faiss.METRIC_INNER_PRODUCT
            or index.d != embedding["dimension"]
            or index.ntotal != len(records)
        ):
            raise RetrievalError("table FAISS type, metric, dimension, or count mismatch")
        _vectors(index.reconstruct_n(0, index.ntotal), index.ntotal, index.d)
        return index, records
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, KeyError, RuntimeError) as exc:
        raise RetrievalError(f"invalid local table index: {exc}") from exc
