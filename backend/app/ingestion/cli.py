"""Manual entry point for curated HTML and PDF ingestion."""

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.ingestion.errors import IngestionError
from app.ingestion.fetcher import fetch_html, fetch_pdf
from app.ingestion.html import extract_html
from app.ingestion.lifecycle import audit, freshness, write_freshness_report
from app.ingestion.manifest import ManifestError, load_manifest
from app.ingestion.models import active_sources, parse_date
from app.ingestion.pdf import extract_pdf
from app.ingestion.snapshots import DEFAULT_ORIGINALS_ROOT, write_pair
from app.ingestion.writer import build_document, build_pdf_document

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "source_manifest.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"


def main(argv: Sequence[str] | None = None, transport: httpx.BaseTransport | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch approved UNECON HTML and PDF sources")
    parser.add_argument("command", choices=["fetch", "list", "audit", "freshness"])
    parser.add_argument("--source-id", help="Fetch one active source")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the destination directory (default depends on source type)",
    )
    parser.add_argument("--originals-root", type=Path, default=DEFAULT_ORIGINALS_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--as-of", type=parse_date, default=datetime.now(UTC).date())
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"FAILED manifest: {exc}")
        print("processed=0 failed=1 skipped=0")
        return 1

    if args.command == "list":
        for source in manifest.sources:
            print(
                f"{source.id} logical={source.logical_document_id} version={source.version} "
                f"status={source.status} year={source.admission_year} type={source.source_type} "
                f"table_aware={source.processing.table_aware}"
            )
        return 0
    if args.command == "audit":
        for warning in audit(manifest, args.as_of):
            print(f"WARNING {warning['source_id']}: {warning['warning']}")
        return 0
    if args.command == "freshness":
        with httpx.Client(timeout=30, transport=transport) as client:
            report = freshness(manifest, args.input_root, args.originals_root, client)
        path = write_freshness_report(args.output_dir or DEFAULT_OUTPUT_ROOT / "freshness", report)
        print(path)
        for result in report["sources"]:
            print(f"{result['source_id']}: {result['status']}")
        return int(
            any(r["status"] not in {"unchanged", "presentation_changed"} for r in report["sources"])
        )
    sources = manifest.sources
    active_ids = {source.id for source in active_sources(manifest)}
    if args.source_id:
        sources = [source for source in sources if source.id == args.source_id]
        if not sources or sources[0].id not in active_ids:
            print(f"FAILED {args.source_id}: unknown or inactive source id")
            print("processed=0 failed=1 skipped=0")
            return 1

    processed = failed = skipped = 0
    with httpx.Client(timeout=15.0, transport=transport) as client:
        for source in sources:
            if source.id not in active_ids:
                skipped += 1
                print(f"SKIPPED {source.id}")
                continue
            try:
                if source.source_type == "html":
                    page = fetch_html(source, client)
                    extracted = extract_html(page.html, source.title)
                    snapshot = page.html.encode("utf-8")
                    document = build_document(source, page.final_url, extracted, snapshot)
                    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / "html"
                else:
                    pdf = fetch_pdf(source, client)
                    snapshot = pdf.content
                    extracted_pdf = extract_pdf(pdf.content)
                    document = build_pdf_document(source, pdf.final_url, pdf.content, extracted_pdf)
                    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / "pdf"
                write_pair(source, output_dir, args.originals_root, document, snapshot)
            except (IngestionError, OSError, ValueError) as exc:
                failed += 1
                print(f"FAILED {source.id}: {exc}")
            else:
                processed += 1
                print(f"OK {source.id}")

    print(f"processed={processed} failed={failed} skipped={skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
