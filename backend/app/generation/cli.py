"""Local grounded answer command; no API or frontend integration."""

import argparse
import io
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout

from app.generation.models import GenerationError
from app.generation.service import AnswerService
from app.retrieval.service import RetrievalService


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Answer from official UNECON chunks with Ollama")
    commands = parser.add_subparsers(dest="command", required=True)
    answer_parser = commands.add_parser("answer")
    answer_parser.add_argument("question")
    answer_parser.add_argument("--retrieval-mode", choices=["dense", "reranked"])
    answer_parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    service = AnswerService(retrieval_service=RetrievalService(mode=args.retrieval_mode))
    try:
        if args.json:
            with redirect_stdout(io.StringIO()):
                response = service.answer(args.question)
            print(response.model_dump_json())
        else:
            response = service.answer(args.question)
            print(f"status={response.status}")
            print(f"retrieval_mode={response.retrieval_mode}")
            print("Ответ:")
            print(response.answer)
            if response.citations:
                print("Источники:")
                for rank, citation in enumerate(response.citations, 1):
                    page = f", стр. {citation.page}" if citation.page is not None else ""
                    print(f"{rank}. {citation.source_title}{page} [{citation.context_id}]")
                    print(f"   {citation.source_url}")
        return 0
    except (GenerationError, OSError, RuntimeError, ValueError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
