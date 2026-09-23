"""Validate and load a complete, ordered corpus without network access."""

import hashlib
import json
from pathlib import Path
from typing import Any

from app.chunking.core import (
    PDF_PAGE_SEPARATOR,
    ChunkingError,
    validate_chunk_artifact,
)
from app.ingestion.manifest import load_manifest


class RetrievalError(ValueError):
    """The local retrieval inputs or index are invalid."""


def _text_units(source_type: str, chunks: list[dict[str, Any]]) -> list[tuple[int | None, str]]:
    units: list[tuple[int | None, str]] = []
    current_page: int | None = None
    current_text = ""
    previous_text = ""
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise RetrievalError("chunk must be an object")
        page = chunk.get("page_start")
        if source_type == "pdf":
            if type(page) is not int or page < 1:
                raise RetrievalError("PDF page number must be positive")
            if current_page is not None and page < current_page:
                raise RetrievalError("PDF pages must be ordered")
        if page != current_page and current_page is not None:
            units.append((current_page, current_text))
            current_text = ""
            previous_text = ""
        current_page = page
        overlap = chunk.get("overlap_prefix_chars")
        text = chunk.get("text")
        if type(overlap) is not int or not isinstance(text, str) or overlap < 0:
            raise RetrievalError("invalid chunk overlap or text")
        if overlap and (not previous_text or text[:overlap] != previous_text[-overlap:]):
            raise RetrievalError("chunk overlap does not match the previous chunk")
        if previous_text and overlap == 0:
            raise RetrievalError("continued chunks must record overlap")
        current_text += text[overlap:]
        previous_text = text
    if chunks:
        units.append((current_page, current_text))
    return units


def load_corpus(
    manifest_path: Path, chunks_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ordered source fingerprints and vector records after strict validation."""
    try:
        manifest = load_manifest(manifest_path)
        fingerprints: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for source in manifest.sources:
            if not source.active:
                continue
            path = chunks_dir / f"{source.id}.json"
            with path.open(encoding="utf-8") as handle:
                artifact = json.load(handle)
            if not isinstance(artifact, dict) or not isinstance(artifact.get("chunks"), list):
                raise RetrievalError(f"invalid chunk artifact: {path}")
            chunks = artifact["chunks"]
            units = _text_units(source.source_type, chunks)
            if source.source_type == "html" and len(units) != 1:
                raise RetrievalError("HTML chunks must form one document")
            validate_chunk_artifact(source, artifact, units)
            reconstructed = (
                units[0][1]
                if source.source_type == "html"
                else PDF_PAGE_SEPARATOR.join(text for _, text in units if text)
            )
            if (
                hashlib.sha256(reconstructed.encode("utf-8")).hexdigest()
                != artifact["source"]["content_sha256"]
            ):
                raise RetrievalError("reconstructed source content hash mismatch")
            metadata = artifact["source"]
            fingerprints.append(
                {
                    "source_id": source.id,
                    "content_sha256": metadata["content_sha256"],
                    "file_sha256": metadata["file_sha256"],
                    "chunk_count": len(chunks),
                }
            )
            for chunk in chunks:
                chunk_id = chunk["chunk_id"]
                if chunk_id in seen_ids:
                    raise RetrievalError(f"duplicate chunk_id: {chunk_id}")
                seen_ids.add(chunk_id)
                records.append(
                    {
                        "vector_id": len(records),
                        "chunk_id": chunk_id,
                        "ordinal": chunk["ordinal"],
                        "text": chunk["text"],
                        "content_sha256": chunk["content_sha256"],
                        "source_id": source.id,
                        "source_title": source.title,
                        "url": source.url,
                        "final_url": metadata["final_url"],
                        "source_type": source.source_type,
                        "category": source.category,
                        "admission_year": source.admission_year,
                        "page_start": chunk["page_start"],
                        "page_end": chunk["page_end"],
                    }
                )
        if not records:
            raise RetrievalError("active corpus contains no chunks")
        return fingerprints, records
    except (OSError, UnicodeError, json.JSONDecodeError, ChunkingError, ValueError) as exc:
        if isinstance(exc, RetrievalError):
            raise
        raise RetrievalError(f"invalid local corpus: {exc}") from exc
