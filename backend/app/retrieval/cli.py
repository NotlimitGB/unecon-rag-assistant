"""Manual commands for building and searching the local dense index."""

import argparse
import io
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path

from app.config import settings
from app.corpus.storage import current_paths
from app.ingestion.snapshots import DEFAULT_ORIGINALS_ROOT
from app.retrieval.corpus import RetrievalError
from app.retrieval.index import RetrievalSession, build_index, search
from app.retrieval.reranker import RerankedRetrievalSession
from app.retrieval.service import RetrievalService
from app.retrieval.table_index import build_table_index

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "source_manifest.json"
DEFAULT_CHUNKS_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "index"
DEFAULT_PDF_ROOT = PROJECT_ROOT / "data" / "processed" / "pdf"
DEFAULT_TABLE_INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "table_index"


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Build and search a local dense FAISS index")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--manifest", type=Path)
    shared.add_argument("--chunks-dir", type=Path)
    shared.add_argument("--index-dir", type=Path)
    shared.add_argument("--originals-root", type=Path)
    shared.add_argument("--pdf-root", type=Path)
    shared.add_argument("--table-index-dir", type=Path)
    shared.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default=settings.embedding_device
    )
    shared.add_argument("--batch-size", type=int, default=settings.embedding_batch_size)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build-index", parents=[shared])
    commands.add_parser("build-table-index", parents=[shared])
    search_parser = commands.add_parser("search", parents=[shared])
    search_parser.add_argument("question")
    search_parser.add_argument("--top-k", type=int, default=5)
    rerank_parser = commands.add_parser("rerank-search", parents=[shared])
    rerank_parser.add_argument("question")
    rerank_parser.add_argument("--top-k", type=int, default=5)
    rerank_parser.add_argument(
        "--reranker-device", choices=["auto", "cpu", "cuda"], default=settings.reranker_device
    )
    rerank_parser.add_argument(
        "--reranker-batch-size", type=int, default=settings.reranker_batch_size
    )
    rerank_parser.add_argument(
        "--reranker-max-length", type=int, default=settings.reranker_max_length
    )
    rerank_parser.add_argument("--candidate-k", type=int, default=settings.reranker_candidate_k)
    retrieve_parser = commands.add_parser("retrieve", parents=[shared])
    retrieve_parser.add_argument("question")
    retrieve_parser.add_argument("--mode", choices=["dense", "reranked"])
    retrieve_parser.add_argument("--top-k", type=int)
    retrieve_parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.batch_size > 128:
        parser.error("--batch-size must be from 1 to 128")
    if args.command == "search" and (args.top_k < 1 or args.top_k > 50):
        parser.error("--top-k must be from 1 to 50")
    if args.command == "rerank-search" and (
        not 5 <= args.candidate_k <= 100
        or not 1 <= args.top_k <= min(5, args.candidate_k)
        or not 1 <= args.reranker_batch_size <= 64
        or not 32 <= args.reranker_max_length <= 4096
    ):
        parser.error("invalid reranker options")
    try:
        path_defaults = {
            "manifest": DEFAULT_MANIFEST,
            "chunks_dir": DEFAULT_CHUNKS_DIR,
            "index_dir": DEFAULT_INDEX_DIR,
            "originals_root": DEFAULT_ORIGINALS_ROOT,
            "pdf_root": DEFAULT_PDF_ROOT,
            "table_index_dir": DEFAULT_TABLE_INDEX_DIR,
        }
        explicit_paths = any(getattr(args, key) is not None for key in path_defaults)
        if args.command not in ("retrieve", "build-index", "build-table-index") and all(
            getattr(args, key) is None for key in path_defaults
        ):
            corpus = current_paths()
            path_defaults = {
                "manifest": corpus.manifest,
                "chunks_dir": corpus.chunks,
                "index_dir": corpus.index,
                "originals_root": corpus.originals,
                "pdf_root": corpus.pdf,
                "table_index_dir": corpus.tables,
            }
        for key, default in path_defaults.items():
            if getattr(args, key) is None and (args.command != "retrieve" or explicit_paths):
                setattr(args, key, default)
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
        elif args.command == "build-table-index":
            metadata = build_table_index(
                args.manifest,
                args.chunks_dir,
                args.index_dir,
                args.pdf_root,
                args.table_index_dir,
                settings.embedding_model,
                args.device,
                args.batch_size,
                originals_root=args.originals_root,
            )
            print(
                f"OK table_vectors={metadata['index']['vector_count']} "
                f"dimension={metadata['embedding']['dimension']} "
                f"sources={len(metadata['sources'])}"
            )
        elif args.command == "rerank-search":
            dense = RetrievalSession(
                args.manifest,
                args.chunks_dir,
                args.index_dir,
                settings.embedding_model,
                args.device,
                args.batch_size,
            )
            session = RerankedRetrievalSession(
                dense,
                settings.reranker_model,
                args.reranker_device,
                args.reranker_batch_size,
                args.reranker_max_length,
                args.candidate_k,
            )
            for rank, result in enumerate(session.search(args.question, args.top_k), 1):
                page = result["page_start"] if result["page_start"] is not None else "-"
                print(
                    f"{rank}. rerank_score={result['rerank_score']:.6f} "
                    f"dense_rank={result['dense_rank']} dense_score={result['dense_score']:.6f}"
                )
                print(f"   source={result['source_id']} page={page} chunk_id={result['chunk_id']}")
                print(f"   text={result['text'][:240].replace(chr(10), ' ')}")
        elif args.command == "retrieve":
            config = settings.model_copy(
                update={
                    "embedding_device": args.device,
                    "embedding_batch_size": args.batch_size,
                }
            )
            service = RetrievalService(
                config=config,
                mode=args.mode,
                manifest_path=args.manifest,
                chunks_dir=args.chunks_dir,
                index_dir=args.index_dir,
                pdf_root=args.pdf_root,
                table_index_dir=args.table_index_dir,
            )
            if args.json:
                with redirect_stdout(io.StringIO()):
                    response = service.retrieve(args.question, args.top_k)
                print(response.model_dump_json())
            else:
                response = service.retrieve(args.question, args.top_k)
                print(f"mode={response.mode} top_k={response.top_k} query={response.query}")
                for result in response.results:
                    page = result.page if result.page is not None else "-"
                    print(
                        f"{result.rank}. source={result.source_id} "
                        f"page={page} score={result.score:.6f}"
                    )
                    print(f"   chunk_id={result.chunk_id}")
                    print(f"   url={result.source_url}")
                    print(f"   dense_score={result.dense_score:.6f}")
                    if result.rerank_score is not None:
                        print(f"   rerank_score={result.rerank_score:.6f}")
                    print(f"   text={result.text[:240].replace(chr(10), ' ')}")
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
        stream = sys.stderr if args.command == "retrieve" and args.json else sys.stdout
        print(f"FAILED: {exc}", file=stream)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
