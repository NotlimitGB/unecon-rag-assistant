"""Prepare and validate a blank human worksheet from frozen Task018 evidence."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from app.experiments.table_aware.human_review import (
    OUTPUT_DIR,
    _load,
    expected_packet,
    render_worksheet,
    summarize_review,
    validate_review,
)

REVIEW_PATH = OUTPUT_DIR / "pdf_page_diversity_human_review.json"
WORKSHEET_PATH = OUTPUT_DIR / "pdf_page_diversity_human_review.md"
SUMMARY_PATH = OUTPUT_DIR / "pdf_page_diversity_human_review_summary.json"


def _json_bytes(document: dict) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _create_new(path: Path, data: bytes) -> None:
    """Create a review artifact without replacing an existing human-edited file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def _write_summary(path: Path, data: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".human-review-summary-", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare() -> dict[str, object]:
    expected = expected_packet()
    blank_bytes = _json_bytes(expected)
    worksheet_bytes = (render_worksheet(expected) + "\n").encode("utf-8")
    if REVIEW_PATH.exists():
        if REVIEW_PATH.read_bytes() != blank_bytes:
            raise ValueError(
                "existing human review differs from blank template; refusing overwrite"
            )
        state = "already_exists"
    else:
        _create_new(REVIEW_PATH, blank_bytes)
        state = "created"
    if not WORKSHEET_PATH.exists():
        _create_new(WORKSHEET_PATH, worksheet_bytes)
    return {"state": state, "cases": len(expected["cases"]), "scoreable": 15, "refusals": 1}


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Frozen Task018 human review workflow")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    for name in ("validate", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--partial", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare()
            print(
                f"review={result['state']} cases={result['cases']} "
                f"scoreable={result['scoreable']} refusals={result['refusals']}"
            )
            print(f"json={REVIEW_PATH}")
            print(f"worksheet={WORKSHEET_PATH}")
            return 0
        expected = expected_packet()
        review = _load(REVIEW_PATH)
        if args.command == "validate":
            progress = validate_review(review, expected, partial=args.partial)
            print(
                f"valid=true fully_reviewed={progress['fully_reviewed']} "
                f"partially_reviewed={progress['partially_reviewed']} "
                f"untouched={progress['untouched']}"
            )
        else:
            summary = summarize_review(review, expected, partial=args.partial)
            if not args.partial:
                _write_summary(SUMMARY_PATH, _json_bytes(summary))
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"human review stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
