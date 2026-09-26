"""Read-only lifecycle audit and remote freshness comparison."""

import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from app.corpus.paths import protect_saved_release
from app.ingestion.fetcher import fetch_html, fetch_pdf
from app.ingestion.html import extract_html
from app.ingestion.models import Manifest, active_sources
from app.ingestion.snapshots import read_pair


def audit(manifest: Manifest, as_of: date) -> list[dict[str, str]]:
    warnings = []
    by_id = {s.id: s for s in manifest.sources}
    for source in manifest.sources:
        reasons = []
        if source.is_active:
            if source.effective_from and source.effective_from > as_of.isoformat():
                reasons.append("active_not_yet_effective")
            if source.effective_to and source.effective_to < as_of.isoformat():
                reasons.append("active_expired")
        if source.status == "superseded":
            if source.effective_from and source.effective_from > as_of.isoformat():
                reasons.append("superseded_future_effective_date")
            successors = [s for s in manifest.sources if s.supersedes == source.id]
            if source.effective_to and any(
                s.effective_from and s.effective_from <= source.effective_to for s in successors
            ):
                reasons.append("superseded_overlapping_interval")
        if source.status == "draft":
            cursor = source
            while cursor.supersedes:
                cursor = by_id[cursor.supersedes]
                if cursor.is_active:
                    reasons.append("draft_successor_of_active")
                    break
        warnings.extend({"source_id": source.id, "warning": reason} for reason in reasons)
    return warnings


def freshness(
    manifest: Manifest, input_root: Path, originals_root: Path, client: httpx.Client
) -> dict:
    results = []
    for source in active_sources(manifest):
        row = {
            "source_id": source.id,
            "logical_document_id": source.logical_document_id,
            "version": source.version,
            "source_type": source.source_type,
            "local": None,
            "remote": None,
            "status": "error",
            "error": None,
            "error_type": None,
            "changes": [],
        }
        try:
            local, _ = read_pair(source, input_root / source.source_type, originals_root)
            row["local"] = {
                "snapshot_sha256": local["source"]["snapshot_sha256"],
                "file_sha256": local["document"].get("file_sha256"),
                "content_sha256": local["document"]["content_sha256"],
                "final_url": local["source"]["final_url"],
            }
            if source.source_type == "html":
                fetched = fetch_html(source, client)
                raw = fetched.html.encode("utf-8")
                text = extract_html(fetched.html, source.title).text
                if not text.strip():
                    raise ValueError("HTML page has no meaningful text")
                remote = {
                    "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
                    "file_sha256": None,
                    "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "final_url": fetched.final_url,
                }
            else:
                fetched = fetch_pdf(source, client)
                digest = hashlib.sha256(fetched.content).hexdigest()
                remote = {
                    "snapshot_sha256": digest,
                    "file_sha256": digest,
                    "content_sha256": None,
                    "final_url": fetched.final_url,
                }
            row["remote"] = remote
            changes = row["changes"]
            for key in ("final_url", "snapshot_sha256", "file_sha256", "content_sha256"):
                if remote[key] is not None and remote[key] != row["local"][key]:
                    changes.append(key)
            if "final_url" in changes:
                row["status"] = "redirect_changed"
            elif source.source_type == "pdf" and "file_sha256" in changes:
                row["status"] = "file_changed"
            elif source.source_type == "html" and "content_sha256" in changes:
                row["status"] = "content_changed"
            elif "snapshot_sha256" in changes:
                row["status"] = "presentation_changed"
            else:
                row["status"] = "unchanged"
        except FileNotFoundError:
            row["status"] = "missing_local_artifact"
            row["error"] = "normalized document or original snapshot is missing"
            row["error_type"] = "FileNotFoundError"
        except (OSError, ValueError, RuntimeError) as exc:
            row["error"] = str(exc)
            row["error_type"] = type(exc).__name__
        results.append(row)
    return {
        "schema_version": 1,
        "registry_schema_version": manifest.schema_version,
        "registry_sha256": hashlib.sha256(manifest.model_dump_json().encode("utf-8")).hexdigest(),
        "checked_at": datetime.now(UTC).isoformat(),
        "active_count": len(results),
        "counts": dict(Counter(row["status"] for row in results)),
        "sources": results,
    }


def write_freshness_report(output_dir: Path, report: dict) -> Path:
    """The audit's only write: a report, independent of ingestion publication."""
    protect_saved_release(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "report.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_dir, suffix=".tmp", mode="w", encoding="utf-8", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
