"""Deterministic, source-independent extraction of PDF table rows."""

import hashlib
import re
from collections import Counter
from typing import Any

import pymupdf

from app.ingestion.models import Source, validate_official_url

EXPERIMENT_ID = "table-aware-pdf-v1"
STRATEGY = "lines_strict"
MAX_CONTEXT_CHARS = 500


class ExperimentError(ValueError):
    """The isolated experiment cannot use the supplied evidence safely."""


def normalize(value: str | None) -> str:
    return " ".join(value.split()) if value else ""


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def row_id(source_id: str, file_sha256: str, page: int, table: int, row: int, text: str) -> str:
    payload = "\0".join((source_id, file_sha256, str(page), str(table), str(row), text))
    return f"{source_id}:p{page:04d}:t{table:03d}:r{row:04d}:{_digest(payload)[:12]}"


def _header_names(table: Any, data_row: Any) -> list[str | None]:
    """Copy extracted headers, including a merged header spanning another column."""
    header = table.header
    names = list(header.names) if header else []
    names.extend([None] * (table.col_count - len(names)))
    if not header:
        return names
    for column, name in enumerate(names):
        if normalize(name) or data_row is None:
            continue
        data_rect = data_row.cells[column]
        if data_rect is None:
            continue
        center_x = (data_rect[0] + data_rect[2]) / 2
        for other, cell in enumerate(header.cells):
            if cell is not None and cell[0] <= center_x <= cell[2] and normalize(names[other]):
                names[column] = names[other]
                break
    return names


def _continuation_count(rows: list[list[str | None]]) -> int:
    """Identify leading sparse rows that continue a multi-level header."""
    count = 1  # PyMuPDF includes its internal header as the first extracted row.
    while count < len(rows):
        row = rows[count]
        occupied = sum(bool(normalize(cell)) for cell in row)
        if not occupied or (not normalize(row[0]) and occupied <= max(2, len(row) // 3)):
            count += 1
        else:
            break
    return count


def _columns(
    header_names: list[str | None], continuation: list[list[str | None]]
) -> list[dict[str, Any]]:
    combined = []
    for column, original in enumerate(header_names):
        parts = [normalize(original)]
        parts.extend(normalize(row[column]) for row in continuation)
        combined.append(" / ".join(part for part in parts if part))
    counts = Counter(label for label in combined if label)
    return [
        {
            "column_id": f"column_{position}",
            "original_header": original,
            "label": label if label and counts[label] == 1 else f"column_{position}",
            "ambiguous": not label or counts[label] > 1,
        }
        for position, (original, label) in enumerate(zip(header_names, combined, strict=True), 1)
    ]


def _context(page: Any, table: Any) -> str:
    if table.header and table.header.external:
        external = normalize(" ".join(name or "" for name in table.header.names))
        if external:
            return external[:MAX_CONTEXT_CHARS]
    above = [
        block
        for block in page.get_text("blocks", sort=True)
        if block[3] <= table.bbox[1] and normalize(block[4])
    ]
    if not above:
        return ""
    nearest = min(above, key=lambda block: table.bbox[1] - block[3])
    return normalize(nearest[4])[:MAX_CONTEXT_CHARS]


def _serialize(source_title: str, page: int, context: str, pairs: list[dict[str, str]]) -> str:
    lines = [f"Источник: {source_title}", f"Страница: {page}"]
    if context:
        lines.append(f"Контекст таблицы: {context}")
    lines.extend(f"{pair['label']} = {pair['value']}" for pair in pairs if pair["value"])
    return "\n".join(lines)


def extract_table(
    source: Source, file_sha256: str, page: Any, page_number: int, table: Any, table_index: int
) -> dict[str, Any]:
    raw_rows = table.extract()
    if not raw_rows or table.col_count < 2:
        raise ExperimentError("detected table has no usable rows or columns")
    if any(len(row) != table.col_count for row in raw_rows):
        raise ExperimentError("detected table has inconsistent column count")
    header_rows = _continuation_count(raw_rows)
    data_row = table.rows[header_rows] if header_rows < len(table.rows) else None
    columns = _columns(_header_names(table, data_row), raw_rows[1:header_rows])
    context = _context(page, table)
    rows = []
    for raw in raw_rows[header_rows:]:
        if not any(normalize(cell) for cell in raw):
            continue
        pairs = [
            {"column_id": column["column_id"], "label": column["label"], "value": normalize(value)}
            for column, value in zip(columns, raw, strict=True)
        ]
        text = _serialize(source.title, page_number, context, pairs)
        index = len(rows) + 1
        rows.append(
            {
                "row_id": row_id(source.id, file_sha256, page_number, table_index, index, text),
                "page": page_number,
                "table_index": table_index,
                "row_index": index,
                "raw_cells": raw,
                "values": pairs,
                "text": text,
                "content_sha256": _digest(text),
            }
        )
    return {
        "page": page_number,
        "table_index": table_index,
        "context": context,
        "columns": columns,
        "rows": rows,
    }


def extract_pdf_tables(source: Source, final_url: str, content: bytes) -> dict[str, Any]:
    if source.source_type != "pdf":
        raise ExperimentError("table extractor requires a PDF source")
    validate_official_url(final_url)
    if not content.startswith(b"%PDF-"):
        raise ExperimentError("invalid PDF signature")
    file_sha256 = hashlib.sha256(content).hexdigest()
    document = pymupdf.open(stream=content, filetype="pdf")
    try:
        if document.needs_pass:
            raise ExperimentError("encrypted PDF cannot be inspected")
        tables = [
            extract_table(source, file_sha256, page, page_number, table, table_index)
            for page_number, page in enumerate(document, 1)
            for table_index, table in enumerate(page.find_tables(strategy=STRATEGY).tables, 1)
        ]
    finally:
        document.close()
    artifact = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source": {
            "id": source.id,
            "title": source.title,
            "final_url": final_url,
            "admission_year": source.admission_year,
            "file_sha256": file_sha256,
        },
        "extraction": {"library": "PyMuPDF", "version": pymupdf.VersionBind, "strategy": STRATEGY},
        "tables": tables,
    }
    validate_artifact(artifact, source, file_sha256, final_url)
    return artifact


def validate_artifact(
    artifact: dict[str, Any], source: Source, file_sha256: str, final_url: str
) -> None:
    """Validate the complete experimental schema and deterministic identities."""

    def exact(value: Any, keys: set[str]) -> None:
        if not isinstance(value, dict) or set(value) != keys:
            raise ExperimentError("invalid table artifact schema")

    exact(artifact, {"schema_version", "experiment_id", "source", "extraction", "tables"})
    if (
        type(artifact["schema_version"]) is not int
        or artifact["schema_version"] != 1
        or artifact["experiment_id"] != EXPERIMENT_ID
    ):
        raise ExperimentError("invalid table artifact identity")
    meta = artifact["source"]
    exact(meta, {"id", "title", "final_url", "admission_year", "file_sha256"})
    if meta != {
        "id": source.id,
        "title": source.title,
        "final_url": final_url,
        "admission_year": source.admission_year,
        "file_sha256": file_sha256,
    } or not re.fullmatch(r"[0-9a-f]{64}", meta["file_sha256"]):
        raise ExperimentError("table artifact source mismatch")
    validate_official_url(meta["final_url"])
    exact(artifact["extraction"], {"library", "version", "strategy"})
    if (
        artifact["extraction"]["library"] != "PyMuPDF"
        or artifact["extraction"]["strategy"] != STRATEGY
        or not isinstance(artifact["extraction"]["version"], str)
        or not artifact["extraction"]["version"]
    ):
        raise ExperimentError("invalid extraction metadata")
    tables = artifact["tables"]
    if not isinstance(tables, list) or not tables:
        raise ExperimentError("no tables detected")
    last_page, last_table = 0, 0
    seen_ids: set[str] = set()
    for table in tables:
        exact(table, {"page", "table_index", "context", "columns", "rows"})
        page, index = table["page"], table["table_index"]
        if (
            type(page) is not int
            or page < 1
            or type(index) is not int
            or index < 1
            or page < last_page
            or index != (last_table + 1 if page == last_page else 1)
        ):
            raise ExperimentError("nonsequential table provenance")
        last_page, last_table = page, index
        if not isinstance(table["context"], str) or len(table["context"]) > MAX_CONTEXT_CHARS:
            raise ExperimentError("invalid table context")
        columns = table["columns"]
        if not isinstance(columns, list) or len(columns) < 2 or not isinstance(table["rows"], list):
            raise ExperimentError("invalid table columns or rows")
        for col_index, col in enumerate(columns, 1):
            exact(col, {"column_id", "original_header", "label", "ambiguous"})
            if (
                col["column_id"] != f"column_{col_index}"
                or not isinstance(col["label"], str)
                or not col["label"]
                or type(col["ambiguous"]) is not bool
                or col["original_header"] is not None
                and not isinstance(col["original_header"], str)
            ):
                raise ExperimentError("invalid column metadata")
        for row_index, row in enumerate(table["rows"], 1):
            exact(
                row,
                {
                    "row_id",
                    "page",
                    "table_index",
                    "row_index",
                    "raw_cells",
                    "values",
                    "text",
                    "content_sha256",
                },
            )
            if (row["page"], row["table_index"], row["row_index"]) != (page, index, row_index):
                raise ExperimentError("invalid row provenance")
            if (
                not isinstance(row["raw_cells"], list)
                or len(row["raw_cells"]) != len(columns)
                or any(cell is not None and not isinstance(cell, str) for cell in row["raw_cells"])
            ):
                raise ExperimentError("invalid raw cells")
            pairs = row["values"]
            if not isinstance(pairs, list) or len(pairs) != len(columns):
                raise ExperimentError("invalid row values")
            for col, pair, raw in zip(columns, pairs, row["raw_cells"], strict=True):
                exact(pair, {"column_id", "label", "value"})
                if pair != {
                    "column_id": col["column_id"],
                    "label": col["label"],
                    "value": normalize(raw),
                }:
                    raise ExperimentError("row column/value association mismatch")
            text = _serialize(source.title, page, table["context"], pairs)
            if row["text"] != text or row["content_sha256"] != _digest(text):
                raise ExperimentError("row text or hash mismatch")
            if row["row_id"] != row_id(source.id, file_sha256, page, index, row_index, text):
                raise ExperimentError("row ID mismatch")
            if row["row_id"] in seen_ids:
                raise ExperimentError("duplicate row ID")
            seen_ids.add(row["row_id"])
