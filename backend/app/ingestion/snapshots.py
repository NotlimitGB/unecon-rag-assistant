"""Immutable original/normalized pairs; no network access."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from app.chunking.core import validate_normalized_document
from app.corpus.paths import protect_saved_release
from app.ingestion.errors import IngestionError
from app.ingestion.html import extract_html
from app.ingestion.models import Source

DEFAULT_ORIGINALS_ROOT = Path(__file__).resolve().parents[3] / "data/processed/originals"


def snapshot_path(root: Path, source: Source) -> Path:
    return root / source.source_type / f"{source.id}.{source.source_type}"


def validate_snapshot(source: Source, artifact: dict, content: bytes) -> None:
    document = validate_normalized_document(source, artifact)
    if hashlib.sha256(content).hexdigest() != artifact["source"]["snapshot_sha256"]:
        raise IngestionError("snapshot SHA-256 mismatch")
    if source.source_type == "html":
        extracted = extract_html(content.decode("utf-8"), source.title)
        if extracted.text != document["text"] or extracted.title != document["title"]:
            raise IngestionError("HTML snapshot does not reproduce normalized document")
    elif not content.startswith(b"%PDF-"):
        raise IngestionError("snapshot has invalid PDF signature")


def read_pair(source: Source, output_dir: Path, originals_root: Path) -> tuple[dict, bytes]:
    artifact = json.loads((output_dir / f"{source.id}.json").read_text(encoding="utf-8"))
    content = snapshot_path(originals_root, source).read_bytes()
    validate_snapshot(source, artifact, content)
    return artifact, content


def write_pair(
    source: Source, output_dir: Path, originals_root: Path, artifact: dict[str, Any], content: bytes
) -> Path:
    """Publish a new pair metadata-last; never overwrite an existing source version."""
    protect_saved_release(output_dir)
    protect_saved_release(originals_root)
    validate_snapshot(source, artifact, content)
    original = snapshot_path(originals_root, source)
    destination = output_dir / f"{source.id}.json"
    original.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = original.with_suffix(original.suffix + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise IngestionError("source version is being written") from exc
    os.close(descriptor)
    temporary: list[Path] = []
    published_original = False
    try:
        if original.exists() or destination.exists():
            if not original.exists() or not destination.exists():
                raise IngestionError("incomplete snapshot pair; explicit recovery required")
            old, old_bytes = read_pair(source, output_dir, originals_root)
            if old != artifact or old_bytes != content:
                raise IngestionError("immutable source version changed; a new version is required")
            return destination
        serialized = (json.dumps(artifact, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        for path, payload in ((original, content), (destination, serialized)):
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=".tmp", delete=False
            ) as handle:
                temporary.append(Path(handle.name))
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary[0], original)
        published_original = True
        os.replace(temporary[1], destination)
        return destination
    except BaseException:
        if published_original and not destination.exists():
            original.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
