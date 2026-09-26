"""Read whole incoming batches before mutation and recheck their exact bytes."""

from dataclasses import dataclass
from pathlib import Path

from app.corpus.models import IncomingRecord
from app.corpus.paths import CorpusError, regular_file, safe_id, safe_path
from app.corpus.storage import parse_json, sha
from app.ingestion.html import extract_html
from app.ingestion.models import Manifest, Source
from app.ingestion.pdf import extract_pdf
from app.ingestion.writer import build_document, build_pdf_document


@dataclass(frozen=True)
class Package:
    path: Path
    record: IncomingRecord
    metadata: bytes
    content: bytes
    source: Source
    artifact: dict


def read_package(path: Path, incoming_root: Path) -> Package:
    path = safe_path(path, incoming_root)
    if path.parent != incoming_root.absolute():
        raise CorpusError("package must be an immediate child of incoming root")
    safe_id(path.name)
    metadata_path = regular_file(path / "metadata.json", incoming_root)
    metadata = metadata_path.read_bytes()
    raw = parse_json(metadata)
    required = set(Source.model_fields) - {"status"}
    if not isinstance(raw, dict) or set(raw) != required | {"schema_version"}:
        raise CorpusError("incoming metadata fields differ from schema")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise CorpusError("incoming metadata schema must be 1")
    source = Source.model_validate({**{k: raw[k] for k in required}, "status": "active"})
    safe_id(source.id)
    safe_id(source.logical_document_id)
    filename = f"document.{source.source_type}"
    if {p.name for p in path.iterdir()} != {"metadata.json", filename}:
        raise CorpusError("package must contain only metadata and one document")
    document_path = regular_file(path / filename, incoming_root)
    if source.source_type == "pdf" and document_path.stat().st_size > 25 * 1024 * 1024:
        raise CorpusError("PDF exceeds 25 MiB")
    content = document_path.read_bytes()
    if source.source_type == "pdf":
        if len(content) > 25 * 1024 * 1024 or not content.startswith(b"%PDF-"):
            raise CorpusError("invalid PDF signature or PDF exceeds 25 MiB")
        artifact = build_pdf_document(source, source.url, content, extract_pdf(content))
    else:
        html = content.decode("utf-8", errors="strict")
        artifact = build_document(source, source.url, extract_html(html, source.title), content)
    record = IncomingRecord(
        package_id=path.name,
        source_id=source.id,
        metadata_sha256=sha(metadata),
        document_sha256=sha(content),
    )
    return Package(path, record, metadata, content, source, artifact)


def transition(base: Manifest, packages: list[Package]) -> Manifest:
    records = [s.model_dump() for s in base.sources]
    existing = {s.id: s for s in base.sources}
    seen_ids, seen_families = set(existing), set()
    for package in packages:
        source = package.source
        family = (source.logical_document_id, source.admission_year)
        if source.id in seen_ids or family in seen_families:
            raise CorpusError("duplicate ID or competing changes in incoming batch")
        seen_ids.add(source.id)
        seen_families.add(family)
        if source.supersedes is None:
            if source.version != 1 or any(
                (s.logical_document_id, s.admission_year) == family for s in base.sources
            ):
                raise CorpusError("new family requires version 1 and no predecessor")
            records.append(source.model_dump())
        else:
            prior = existing.get(source.supersedes)
            if (
                prior is None
                or not prior.is_active
                or source.version != prior.version + 1
                or family != (prior.logical_document_id, prior.admission_year)
            ):
                raise CorpusError(
                    "replacement requires current active predecessor and next version"
                )
            index = next(i for i, r in enumerate(records) if r["id"] == prior.id)
            records[index]["status"] = "superseded"
            records.insert(index + 1, source.model_dump())
    return Manifest.model_validate({"schema_version": 2, "sources": records})


def recheck(records: list[IncomingRecord], incoming_root: Path):
    packages = [
        read_package(incoming_root / record.package_id, incoming_root) for record in records
    ]
    if [package.record for package in packages] != records:
        raise CorpusError("incoming package changed since staging")
    return packages
