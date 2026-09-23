"""Manual entry point for curated HTML and PDF ingestion."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import httpx

from app.ingestion.errors import IngestionError
from app.ingestion.fetcher import fetch_html, fetch_pdf
from app.ingestion.html import extract_html
from app.ingestion.manifest import ManifestError, load_manifest
from app.ingestion.pdf import extract_pdf
from app.ingestion.writer import build_document, build_pdf_document, write_document

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "source_manifest.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"


def main(argv: Sequence[str] | None = None, transport: httpx.BaseTransport | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch approved UNECON HTML and PDF sources")
    parser.add_argument("command", choices=["fetch"])
    parser.add_argument("--source-id", help="Fetch one active source")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the destination directory (default depends on source type)",
    )
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"FAILED manifest: {exc}")
        print("processed=0 failed=1 skipped=0")
        return 1

    sources = manifest.sources
    if args.source_id:
        sources = [source for source in sources if source.id == args.source_id]
        if not sources or not sources[0].active:
            print(f"FAILED {args.source_id}: unknown or inactive source id")
            print("processed=0 failed=1 skipped=0")
            return 1

    processed = failed = skipped = 0
    with httpx.Client(timeout=15.0, transport=transport) as client:
        for source in sources:
            if not source.active:
                skipped += 1
                print(f"SKIPPED {source.id}")
                continue
            try:
                if source.source_type == "html":
                    page = fetch_html(source, client)
                    extracted = extract_html(page.html, source.title)
                    document = build_document(source, page.final_url, extracted)
                    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / "html"
                else:
                    pdf = fetch_pdf(source, client)
                    extracted_pdf = extract_pdf(pdf.content)
                    document = build_pdf_document(source, pdf.final_url, pdf.content, extracted_pdf)
                    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / "pdf"
                write_document(output_dir, source.id, document)
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
