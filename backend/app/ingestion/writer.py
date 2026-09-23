"""Build deterministic normalized documents and write them atomically."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from app.ingestion.fetcher import IngestionError
from app.ingestion.html import ExtractedDocument
from app.ingestion.models import Source


def build_document(source: Source, final_url: str, extracted: ExtractedDocument) -> dict[str, Any]:
    if not extracted.text.strip():
        raise IngestionError("HTML page has no meaningful text")
    return {
        "schema_version": 1,
        "source": {
            **source.model_dump(exclude={"active"}),
            "final_url": final_url,
        },
        "document": {
            "title": extracted.title,
            "text": extracted.text,
            "content_sha256": hashlib.sha256(extracted.text.encode("utf-8")).hexdigest(),
        },
    }


def write_document(output_dir: Path, source_id: str, document: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{source_id}.json"
    serialized = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=output_dir,
            prefix=f".{source_id}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
