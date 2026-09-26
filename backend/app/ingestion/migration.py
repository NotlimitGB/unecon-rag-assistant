"""Explicit Task025 v1 migration into an empty staging directory, never in place."""

import argparse
import hashlib
import json
from pathlib import Path

import httpx

from app.chunking.core import (
    build_chunk_artifact,
    validate_chunk_artifact,
    validate_normalized_document,
)
from app.ingestion.fetcher import fetch_html, fetch_pdf
from app.ingestion.html import extract_html
from app.ingestion.models import Manifest, source_identity
from app.ingestion.pdf import extract_pdf
from app.ingestion.snapshots import write_pair
from app.ingestion.writer import build_document, build_pdf_document, write_document

LOGICAL_IDS = {
    "admissions-bachelor": "admissions-bachelor-page",
    "admission-documents": "admission-documents-page",
    "admission-rules": "admission-rules-page",
    "entrance-exams": "entrance-exams-page",
    "admissions-faq": "admissions-faq-page",
    "tuition": "tuition-page",
    "admission-rules-pdf": "admission-rules-document",
    "admission-capacity-pdf": "admission-capacity-document",
    "admission-deadlines-pdf": "admission-deadlines-document",
    "entrance-exams-list-pdf": "entrance-exams-list-document",
    "entrance-exam-regulations-pdf": "entrance-exam-regulations-document",
    "tuition-order-128-pdf": "tuition-fees-document",
}
TABLE_IDS = {"admission-capacity-pdf", "entrance-exams-list-pdf", "tuition-order-128-pdf"}
LEGACY_KEYS = {"id", "title", "url", "source_type", "category", "admission_year"}


def migrate_registry(raw: dict) -> Manifest:
    if set(raw) != {"schema_version", "sources"} or type(raw["schema_version"]) is not int:
        raise ValueError("invalid legacy manifest")
    if raw["schema_version"] != 1 or [s["id"] for s in raw["sources"]] != list(LOGICAL_IDS):
        raise ValueError("migration requires the accepted ordered Task024A registry")
    sources = []
    for source in raw["sources"]:
        if set(source) != LEGACY_KEYS | {"active"} or source["active"] is not True:
            raise ValueError("invalid legacy source")
        sources.append(
            {
                **{key: source[key] for key in LEGACY_KEYS},
                "logical_document_id": LOGICAL_IDS[source["id"]],
                "version": 1,
                "status": "active",
                "supersedes": None,
                "published_at": None,
                "effective_from": None,
                "effective_to": None,
                "processing": {"table_aware": source["id"] in TABLE_IDS},
            }
        )
    return Manifest.model_validate({"schema_version": 2, "sources": sources})


def prepare(legacy_manifest: Path, input_root: Path, staging: Path, client: httpx.Client) -> dict:
    if staging.exists():
        raise ValueError("staging directory must not exist")
    manifest = migrate_registry(json.loads(legacy_manifest.read_text(encoding="utf-8")))
    prepared = []
    comparisons = []
    for source in manifest.sources:
        old = json.loads(
            (input_root / source.source_type / f"{source.id}.json").read_text(encoding="utf-8")
        )
        if (
            set(old) != {"schema_version", "source", "document"}
            or type(old["schema_version"]) is not int
            or old["schema_version"] != 1
            or set(old["source"]) != LEGACY_KEYS | {"final_url"}
            or any(old["source"][key] != source_identity(source)[key] for key in LEGACY_KEYS)
        ):
            raise ValueError(f"invalid legacy normalized artifact: {source.id}")
        if source.source_type == "html":
            fetched = fetch_html(source, client)
            snapshot = fetched.html.encode("utf-8")
            document = build_document(
                source, fetched.final_url, extract_html(fetched.html, source.title), snapshot
            )
        else:
            fetched = fetch_pdf(source, client)
            snapshot = fetched.content
            document = build_pdf_document(
                source, fetched.final_url, snapshot, extract_pdf(snapshot)
            )
        upgraded_old = {
            "schema_version": 2,
            "source": {
                **source_identity(source),
                "final_url": old["source"]["final_url"],
                "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
            },
            "document": old["document"],
        }
        validate_normalized_document(source, upgraded_old)
        if document != upgraded_old:
            raise ValueError(f"source drift blocks migration: {source.id}")
        chunks = build_chunk_artifact(source, document)
        old_chunks = json.loads(
            (input_root / "chunks" / f"{source.id}.json").read_text(encoding="utf-8")
        )
        if type(old_chunks.get("schema_version")) is not int or old_chunks["schema_version"] != 1:
            raise ValueError(f"invalid legacy chunk schema: {source.id}")
        units = (
            [(None, document["document"]["text"])]
            if source.source_type == "html"
            else [(p["page_number"], p["text"]) for p in document["document"]["pages"]]
        )
        validate_chunk_artifact(
            source,
            {
                **old_chunks,
                "schema_version": 2,
                "source": {
                    **old_chunks["source"],
                    "logical_document_id": source.logical_document_id,
                    "version": source.version,
                    "snapshot_sha256": document["source"]["snapshot_sha256"],
                },
            },
            units,
        )
        expected_old_chunks = {
            **chunks,
            "schema_version": 1,
            "source": {
                key: value
                for key, value in chunks["source"].items()
                if key not in {"logical_document_id", "version", "snapshot_sha256"}
            },
        }
        if old_chunks != expected_old_chunks:
            raise ValueError(f"legacy chunks mismatch: {source.id}")
        prepared.append((source, snapshot, document, chunks))
        comparisons.append(
            {
                "source_id": source.id,
                "logical_document_id": source.logical_document_id,
                "content_sha256": document["document"]["content_sha256"],
                "snapshot_sha256": document["source"]["snapshot_sha256"],
                "file_sha256": document["document"].get("file_sha256"),
                "final_url": fetched.final_url,
                "matched": True,
                "chunks": len(chunks["chunks"]),
            }
        )
    # All remote and legacy comparisons completed before creating any candidate artifacts.
    staging.mkdir(parents=True)
    (staging / "source_manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    for source, snapshot, document, chunks in prepared:
        write_pair(source, staging / source.source_type, staging / "originals", document, snapshot)
        write_document(staging / "chunks", source.id, chunks)
    report = {
        "schema_version": 1,
        "sources": comparisons,
        "chunk_count": sum(row["chunks"] for row in comparisons),
    }
    write_document(staging, "migration_comparison", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    args = parser.parse_args()
    try:
        with httpx.Client(timeout=30) as client:
            report = prepare(args.legacy_manifest, args.input_root, args.staging, client)
        print(f"Prepared sources={len(report['sources'])} chunks={report['chunk_count']}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"BLOCKED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
