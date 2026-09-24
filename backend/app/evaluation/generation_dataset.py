"""Strict, offline validation of the frozen generation evaluation set."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.evaluation.dataset import REQUIRED_CATEGORIES, DatasetError

DATASET_ID = "unecon-generation-2026-v1"
ITEM_KEYS = {
    "question_id",
    "question",
    "expected_status",
    "retrieval_question_id",
    "reference_facts",
    "manual_review_note",
}


def validate_generation_dataset(raw: Any, retrieval: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "dataset_id",
        "admission_year",
        "questions",
    }:
        raise DatasetError("invalid generation dataset schema")
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["dataset_id"] != DATASET_ID
        or type(raw["admission_year"]) is not int
        or raw["admission_year"] != 2026
    ):
        raise DatasetError("invalid generation dataset identity")
    items = raw["questions"]
    if not isinstance(items, list) or len(items) != 60:
        raise DatasetError("generation dataset must contain 60 questions")
    gold = {item["question_id"]: item for item in retrieval["questions"]}
    seen_questions: set[str] = set()
    seen_links: set[str] = set()
    statuses: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    difficulties: dict[str, Counter[str]] = {
        category: Counter() for category in REQUIRED_CATEGORIES
    }
    for position, item in enumerate(items, 1):
        if not isinstance(item, dict) or set(item) != ITEM_KEYS:
            raise DatasetError(f"invalid generation question fields at {position}")
        if item["question_id"] != f"gen-{position:03d}":
            raise DatasetError("generation question IDs must be sequential")
        question = item["question"]
        if not isinstance(question, str) or not question.strip():
            raise DatasetError(f"empty generation question at {position}")
        normalized = " ".join(question.split()).casefold()
        if normalized in seen_questions:
            raise DatasetError(f"duplicate generation question at {position}")
        seen_questions.add(normalized)
        status = item["expected_status"]
        if not isinstance(status, str) or status not in {"answered", "insufficient_evidence"}:
            raise DatasetError(f"invalid expected status at {position}")
        statuses[status] += 1
        note = item["manual_review_note"]
        if not isinstance(note, str) or not note.strip():
            raise DatasetError(f"empty manual review note at {position}")
        facts = item["reference_facts"]
        if not isinstance(facts, list) or any(
            not isinstance(fact, str) or not fact.strip() for fact in facts
        ):
            raise DatasetError(f"invalid reference facts at {position}")
        link = item["retrieval_question_id"]
        if status == "answered":
            if not isinstance(link, str) or link not in gold or link in seen_links:
                raise DatasetError(f"invalid retrieval link at {position}")
            seen_links.add(link)
            if question != gold[link]["question"] or not facts:
                raise DatasetError(f"unsupported question text or facts at {position}")
            category = gold[link]["category"]
            categories[category] += 1
            difficulties[category][gold[link]["difficulty"]] += 1
        elif link is not None or facts:
            raise DatasetError(f"unsupported question has link or facts at {position}")
    if statuses != {"answered": 42, "insufficient_evidence": 18}:
        raise DatasetError("generation status distribution must be 42/18")
    if categories != {category: 6 for category in REQUIRED_CATEGORIES}:
        raise DatasetError("generation category distribution must be six each")
    if any(counts != {"easy": 2, "medium": 2, "hard": 2} for counts in difficulties.values()):
        raise DatasetError("generation difficulty distribution must be 2/2/2 per category")
    return raw


def load_generation_dataset(path: Path, retrieval: dict[str, Any]) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot read generation dataset: {exc}") from exc
    return validate_generation_dataset(raw, retrieval)
