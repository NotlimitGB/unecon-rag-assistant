"""Build deterministic, provenance-aware chunks from normalized documents."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from app.ingestion.manifest import load_manifest
from app.ingestion.models import Source, validate_official_url
from app.ingestion.writer import write_document

ALGORITHM = "paragraph-aware-v1"
MAX_CHARS = 1200
TARGET_OVERLAP_CHARS = 150
PDF_PAGE_SEPARATOR = "\n\n\f\n\n"
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_PARAGRAPH_BOUNDARY = re.compile(r"\n{2,}")
_SENTENCE_BOUNDARY = re.compile(r"[.!?…][»”\"')\]]*\s+")
_WHITESPACE_BOUNDARY = re.compile(r"\s+")


class ChunkingError(ValueError):
    """A normalized document cannot safely be chunked."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _required_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ChunkingError(f"{label} must be an object")
    return value


def _require_exact_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        missing = sorted(keys - set(value))
        extra = sorted(set(value) - keys)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unknown {', '.join(extra)}")
        raise ChunkingError(f"invalid {label} schema: {'; '.join(details)}")


def _validate_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HASH_PATTERN.fullmatch(value):
        raise ChunkingError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def validate_normalized_document(source: Source, artifact: Any) -> dict[str, Any]:
    """Validate the Task002/003 artifact contract and return its document object."""
    root = _required_dict(artifact, "artifact")
    _require_exact_keys(root, {"schema_version", "source", "document"}, "artifact")
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise ChunkingError("unsupported normalized document schema_version")

    source_data = _required_dict(root["source"], "source")
    expected_source_keys = {
        "id",
        "title",
        "url",
        "source_type",
        "category",
        "admission_year",
        "final_url",
    }
    _require_exact_keys(source_data, expected_source_keys, "source")
    expected_source = source.model_dump(exclude={"active"})
    for key, expected_value in expected_source.items():
        if source_data.get(key) != expected_value:
            raise ChunkingError(f"source metadata mismatch for {key}")
    if not isinstance(source_data["final_url"], str):
        raise ChunkingError("source.final_url must be a string")
    try:
        validate_official_url(source_data["url"])
        validate_official_url(source_data["final_url"])
    except ValueError as exc:
        raise ChunkingError(f"source URL is outside the approved boundary: {exc}") from exc

    document = _required_dict(root["document"], "document")
    if source.source_type == "html":
        _require_exact_keys(document, {"title", "text", "content_sha256"}, "HTML document")
    else:
        _require_exact_keys(
            document,
            {"title", "page_count", "pages", "text", "content_sha256", "file_sha256"},
            "PDF document",
        )

    if not isinstance(document["title"], str):
        raise ChunkingError("document.title must be a string")
    if not isinstance(document["text"], str):
        raise ChunkingError("document.text must be a string")
    content_hash = _validate_hash(document["content_sha256"], "document.content_sha256")
    if content_hash != _sha256_text(document["text"]):
        raise ChunkingError("document content_sha256 does not match document.text")
    if not document["text"]:
        raise ChunkingError("document.text is empty")

    if source.source_type == "pdf":
        if document["title"] != source.title:
            raise ChunkingError("PDF document title does not match manifest title")
        _validate_hash(document["file_sha256"], "document.file_sha256")
        pages = document["pages"]
        if not isinstance(pages, list) or type(document["page_count"]) is not int:
            raise ChunkingError("PDF pages and page_count have invalid types")
        if document["page_count"] != len(pages) or not pages:
            raise ChunkingError("PDF page_count does not match pages")
        page_texts: list[str] = []
        for expected_number, raw_page in enumerate(pages, start=1):
            page = _required_dict(raw_page, f"pages[{expected_number - 1}]")
            _require_exact_keys(page, {"page_number", "text"}, f"pages[{expected_number - 1}]")
            if type(page["page_number"]) is not int or page["page_number"] != expected_number:
                raise ChunkingError("PDF page numbers must be sequential and start at 1")
            if not isinstance(page["text"], str):
                raise ChunkingError(f"PDF page {expected_number} text must be a string")
            page_texts.append(page["text"])
        recomposed = PDF_PAGE_SEPARATOR.join(text for text in page_texts if text)
        if document["text"] != recomposed:
            raise ChunkingError("PDF document.text does not match its non-empty pages")
    return document


def _preferred_cut(text: str, start: int, limit: int) -> int:
    """Choose the furthest useful paragraph, sentence, or word boundary."""
    if limit >= len(text):
        return len(text)
    capacity = limit - start
    minimum_preferred = start + max(1, capacity // 2)
    for pattern in (_PARAGRAPH_BOUNDARY, _SENTENCE_BOUNDARY, _WHITESPACE_BOUNDARY):
        candidates = [
            match.end()
            for match in pattern.finditer(text, start, limit + 1)
            if minimum_preferred <= match.end() <= limit
        ]
        if candidates:
            return candidates[-1]
    return limit


def _overlap_start(text: str, cursor: int, previous_core_start: int) -> int:
    earliest = max(previous_core_start, cursor - TARGET_OVERLAP_CHARS)
    for index in range(earliest, cursor):
        if text[index].isspace():
            return index
    return earliest


def _chunk_unit(
    text: str,
    source_id: str,
    document_hash: str,
    page_number: int | None,
    first_ordinal: int,
) -> list[dict[str, Any]]:
    if not text:
        return []
    if len(text) <= MAX_CHARS:
        pieces = [(0, len(text), text, 0)]
    else:
        pieces: list[tuple[int, int, str, int]] = []
        cursor = 0
        previous_core_start = 0
        previous_text = ""
        while cursor < len(text):
            overlap_start = _overlap_start(text, cursor, previous_core_start) if pieces else cursor
            overlap = text[overlap_start:cursor]
            if overlap and not previous_text.endswith(overlap):
                raise ChunkingError("internal overlap does not match preceding chunk")
            available = MAX_CHARS - len(overlap)
            end = _preferred_cut(text, cursor, min(cursor + available, len(text)))
            core = text[cursor:end]
            if not core:
                raise ChunkingError("chunking made no progress")
            chunk_text = overlap + core
            pieces.append((cursor, end, chunk_text, len(overlap)))
            previous_core_start = cursor
            previous_text = chunk_text
            cursor = end

    chunks: list[dict[str, Any]] = []
    for offset, (_, _, chunk_text, overlap_chars) in enumerate(pieces):
        ordinal = first_ordinal + offset
        page_start = page_number
        page_end = page_number
        id_parts = (
            source_id,
            document_hash,
            str(ordinal),
            "" if page_start is None else str(page_start),
            "" if page_end is None else str(page_end),
            chunk_text,
        )
        digest = hashlib.sha256("\0".join(id_parts).encode("utf-8")).hexdigest()
        chunks.append(
            {
                "id": f"{source_id}:{ordinal:04d}:{digest[:12]}",
                "ordinal": ordinal,
                "page_start": page_start,
                "page_end": page_end,
                "char_count": len(chunk_text),
                "text": chunk_text,
                "content_sha256": _sha256_text(chunk_text),
                "overlap_chars": overlap_chars,
            }
        )
    return chunks


def build_chunk_artifact(source: Source, normalized: Any) -> dict[str, Any]:
    """Validate and chunk one normalized HTML or PDF artifact."""
    document = validate_normalized_document(source, normalized)
    document_hash = document["content_sha256"]
    if source.source_type == "html":
        chunks = _chunk_unit(document["text"], source.id, document_hash, None, 1)
        text_units = [(None, document["text"])]
    else:
        chunks = []
        text_units = []
        for page in document["pages"]:
            page_text = page["text"]
            text_units.append((page["page_number"], page_text))
            chunks.extend(
                _chunk_unit(
                    page_text,
                    source.id,
                    document_hash,
                    page["page_number"],
                    len(chunks) + 1,
                )
            )

    artifact = {
        "schema_version": 1,
        "chunking": {
            "algorithm": ALGORITHM,
            "max_chars": MAX_CHARS,
            "target_overlap_chars": TARGET_OVERLAP_CHARS,
        },
        "source": normalized["source"],
        "document": {
            "title": document["title"],
            "content_sha256": document_hash,
            "file_sha256": document.get("file_sha256"),
        },
        "chunks": chunks,
    }
    validate_chunk_artifact(source, artifact, text_units)
    return artifact


def validate_chunk_artifact(
    source: Source, artifact: Any, text_units: list[tuple[int | None, str]]
) -> None:
    """Check chunk limits, hashes, IDs, page boundaries, and lossless source coverage."""
    root = _required_dict(artifact, "chunk artifact")
    _require_exact_keys(
        root, {"schema_version", "chunking", "source", "document", "chunks"}, "chunk artifact"
    )
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise ChunkingError("unsupported chunk artifact schema_version")
    chunking = _required_dict(root["chunking"], "chunking metadata")
    _require_exact_keys(
        chunking, {"algorithm", "max_chars", "target_overlap_chars"}, "chunking metadata"
    )
    if chunking != {
        "algorithm": ALGORITHM,
        "max_chars": MAX_CHARS,
        "target_overlap_chars": TARGET_OVERLAP_CHARS,
    }:
        raise ChunkingError("chunking metadata does not match the active algorithm")

    chunk_source = _required_dict(root["source"], "chunk artifact source")
    chunk_document = _required_dict(root["document"], "chunk artifact document")
    _require_exact_keys(
        chunk_document, {"title", "content_sha256", "file_sha256"}, "chunk artifact document"
    )
    _require_exact_keys(
        chunk_source,
        {"id", "title", "url", "source_type", "category", "admission_year", "final_url"},
        "chunk artifact source",
    )
    if chunk_source.get("id") != source.id:
        raise ChunkingError("chunk artifact source id mismatch")
    for key, expected_value in source.model_dump(exclude={"active"}).items():
        if chunk_source.get(key) != expected_value:
            raise ChunkingError(f"chunk artifact source metadata mismatch for {key}")
    try:
        validate_official_url(chunk_source["final_url"])
    except ValueError as exc:
        raise ChunkingError("chunk artifact final URL is outside the approved boundary") from exc

    document_text = (
        text_units[0][1]
        if source.source_type == "html"
        else PDF_PAGE_SEPARATOR.join(text for _, text in text_units if text)
    )
    content_hash = _validate_hash(
        chunk_document["content_sha256"], "chunk artifact document.content_sha256"
    )
    if content_hash != _sha256_text(document_text):
        raise ChunkingError("chunk artifact document hash does not match source text")
    if not isinstance(chunk_document["title"], str):
        raise ChunkingError("chunk artifact document title must be a string")
    if source.source_type == "html":
        if chunk_document["file_sha256"] is not None:
            raise ChunkingError("HTML chunk artifact file_sha256 must be null")
    else:
        _validate_hash(chunk_document["file_sha256"], "chunk artifact document.file_sha256")
    chunk_list = root["chunks"]
    if not isinstance(chunk_list, list) or not chunk_list:
        raise ChunkingError("chunk artifact must contain at least one chunk")

    grouped: dict[int | None, list[dict[str, Any]]] = {page: [] for page, _ in text_units}
    seen_ids: set[str] = set()
    for expected_ordinal, raw_chunk in enumerate(chunk_list, start=1):
        chunk = _required_dict(raw_chunk, f"chunks[{expected_ordinal - 1}]")
        _require_exact_keys(
            chunk,
            {
                "id",
                "ordinal",
                "page_start",
                "page_end",
                "char_count",
                "text",
                "content_sha256",
                "overlap_chars",
            },
            f"chunks[{expected_ordinal - 1}]",
        )
        if type(chunk["ordinal"]) is not int or chunk["ordinal"] != expected_ordinal:
            raise ChunkingError("chunk ordinals must be sequential and start at 1")
        text = chunk["text"]
        if not isinstance(text, str) or not text or len(text) > MAX_CHARS:
            raise ChunkingError("chunk text must contain 1–1200 characters")
        if type(chunk["char_count"]) is not int or chunk["char_count"] != len(text):
            raise ChunkingError("chunk char_count does not match its text")
        if (
            type(chunk["overlap_chars"]) is not int
            or not 0 <= chunk["overlap_chars"] <= TARGET_OVERLAP_CHARS
        ):
            raise ChunkingError("chunk overlap_chars is outside the supported range")
        expected_page = chunk["page_start"]
        if source.source_type == "html":
            if expected_page is not None or chunk["page_end"] is not None:
                raise ChunkingError("HTML chunks must not have page provenance")
            group_key = None
        else:
            if (
                type(expected_page) is not int
                or type(chunk["page_end"]) is not int
                or chunk["page_end"] != expected_page
                or expected_page not in grouped
            ):
                raise ChunkingError("PDF chunks must belong to exactly one source page")
            group_key = expected_page
        if group_key not in grouped:
            raise ChunkingError("chunk page provenance is not present in the source")

        content_hash = _validate_hash(chunk["content_sha256"], "chunk.content_sha256")
        if content_hash != _sha256_text(text):
            raise ChunkingError("chunk content_sha256 does not match chunk text")
        page_number = chunk["page_start"]
        id_parts = (
            source.id,
            root["document"]["content_sha256"],
            str(expected_ordinal),
            "" if page_number is None else str(page_number),
            "" if chunk["page_end"] is None else str(chunk["page_end"]),
            text,
        )
        digest = hashlib.sha256("\0".join(id_parts).encode("utf-8")).hexdigest()
        expected_id = f"{source.id}:{expected_ordinal:04d}:{digest[:12]}"
        if chunk["id"] != expected_id or chunk["id"] in seen_ids:
            raise ChunkingError("chunk id is invalid or duplicated")
        seen_ids.add(chunk["id"])
        grouped[group_key].append(chunk)

    for page_number, source_text in text_units:
        page_chunks = grouped[page_number]
        reconstructed: list[str] = []
        for index, chunk in enumerate(page_chunks):
            overlap = chunk["overlap_chars"]
            if index == 0:
                if overlap != 0:
                    raise ChunkingError("first chunk in a source unit cannot have overlap")
            else:
                previous = page_chunks[index - 1]["text"]
                if overlap and chunk["text"][:overlap] != previous[-overlap:]:
                    raise ChunkingError("chunk overlap does not match preceding chunk")
                if not overlap:
                    raise ChunkingError("continued chunks must record their overlap")
            reconstructed.append(chunk["text"][overlap:])
        if "".join(reconstructed) != source_text:
            raise ChunkingError("chunks do not cover source text exactly")


def build_source_chunks(source: Source, input_root: Path, output_dir: Path) -> Path:
    input_path = input_root / source.source_type / f"{source.id}.json"
    try:
        with input_path.open(encoding="utf-8") as handle:
            normalized = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ChunkingError(f"cannot read normalized document {input_path}: {exc}") from exc
    artifact = build_chunk_artifact(source, normalized)
    return write_document(output_dir, source.id, artifact)


def load_source_manifest(path: Path):
    """Expose manifest loading at the stage boundary with its existing strict schema."""
    return load_manifest(path)
