"""Command-line entry point for deterministic local document chunking."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from app.chunking.core import ChunkingError, build_source_chunks, load_source_manifest
from app.ingestion.manifest import ManifestError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "source_manifest.json"
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "chunks"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build local provenance-aware document chunks")
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--source-id", help="Build chunks for one active source")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    try:
        manifest = load_source_manifest(args.manifest)
    except ManifestError as exc:
        print(f"FAILED manifest: {exc}")
        print("processed=0 failed=1 skipped=0 chunks=0")
        return 1

    sources = manifest.sources
    if args.source_id:
        sources = [source for source in sources if source.id == args.source_id]
        if not sources or not sources[0].active:
            print(f"FAILED {args.source_id}: unknown or inactive source id")
            print("processed=0 failed=1 skipped=0 chunks=0")
            return 1

    processed = failed = skipped = chunks = 0
    for source in sources:
        if not source.active:
            skipped += 1
            print(f"SKIPPED {source.id}")
            continue
        try:
            source_chunks = build_source_chunks(source, args.input_root, args.output_dir)
        except (ChunkingError, ManifestError, OSError, ValueError) as exc:
            failed += 1
            print(f"FAILED {source.id}: {exc}")
        else:
            processed += 1
            chunks += source_chunks
            print(f"OK {source.id} chunks={source_chunks}")

    print(f"processed={processed} failed={failed} skipped={skipped} chunks={chunks}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
