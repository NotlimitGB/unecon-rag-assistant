"""Manual commands for building and searching the local dense index."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from app.config import settings
from app.retrieval.corpus import RetrievalError
from app.retrieval.index import build_index, search

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "source_manifest.json"
DEFAULT_CHUNKS_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "index"


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Build and search a local dense FAISS index")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    shared.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    shared.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    shared.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default=settings.embedding_device
    )
    shared.add_argument("--batch-size", type=int, default=settings.embedding_batch_size)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build-index", parents=[shared])
    search_parser = commands.add_parser("search", parents=[shared])
    search_parser.add_argument("question")
    search_parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.batch_size > 128:
        parser.error("--batch-size must be from 1 to 128")
    if args.command == "search" and (args.top_k < 1 or args.top_k > 50):
        parser.error("--top-k must be from 1 to 50")
    try:
        if args.command == "build-index":
            metadata = build_index(
                args.manifest,
                args.chunks_dir,
                args.index_dir,
                settings.embedding_model,
                args.device,
                args.batch_size,
            )
            print(
                f"OK vectors={metadata['index']['vector_count']} "
                f"dimension={metadata['embedding']['dimension']} "
                f"sources={len(metadata['corpus']['sources'])}"
            )
        else:
            results = search(
                args.question,
                args.top_k,
                args.manifest,
                args.chunks_dir,
                args.index_dir,
                settings.embedding_model,
                args.device,
                args.batch_size,
            )
            for rank, result in enumerate(results, start=1):
                page = result["page_start"] if result["page_start"] is not None else "-"
                print(
                    f"{rank}. {result['chunk_id']} score={result['score']:.6f} "
                    f"page={page} source={result['source_id']} url={result['final_url']}"
                )
                print(result["text"][:240].replace("\n", " "))
        return 0
    except (RetrievalError, OSError, ValueError, RuntimeError) as exc:
        print(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
