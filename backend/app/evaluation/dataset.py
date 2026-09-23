"""Strict validation of the grounded retrieval evaluation dataset."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from app.ingestion.models import Manifest

DATASET_ID = "unecon-retrieval-2026-v1"
REQUIRED_CATEGORIES = {
    "admission_rules",
    "admission_deadlines",
    "admission_capacity",
    "entrance_exams",
    "tuition",
    "faq",
    "admissions_overview",
}
QUESTION_KEYS = {
    "question_id",
    "question",
    "category",
    "difficulty",
    "primary_source_id",
    "acceptable_source_ids",
    "expected_pages",
    "evidence_note",
}


class DatasetError(ValueError):
    """An evaluation dataset or page label is invalid."""


def validate_dataset(raw: Any, manifest: Manifest) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "dataset_id",
        "admission_year",
        "questions",
    }:
        raise DatasetError("invalid dataset top-level schema")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise DatasetError("invalid dataset schema_version")
    if (
        raw["dataset_id"] != DATASET_ID
        or type(raw["admission_year"]) is not int
        or raw["admission_year"] != 2026
    ):
        raise DatasetError("invalid dataset identity or admission year")
    questions = raw["questions"]
    if not isinstance(questions, list) or len(questions) != 80:
        raise DatasetError("dataset must contain exactly 80 questions")
    active = {
        source.id: source
        for source in manifest.sources
        if source.active and source.admission_year == raw["admission_year"]
    }
    categories = {source.category for source in active.values()}
    seen_texts: set[str] = set()
    difficulty_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    for index, item in enumerate(questions, 1):
        if not isinstance(item, dict) or set(item) != QUESTION_KEYS:
            raise DatasetError(f"invalid question schema at position {index}")
        if item["question_id"] != f"ret-{index:03d}":
            raise DatasetError("question IDs must be sequential ret-001 through ret-080")
        question = item["question"]
        if (
            not isinstance(question, str)
            or not question.strip()
            or not re.search(r"[А-Яа-яЁё]", question)
        ):
            raise DatasetError(f"invalid Russian question at {index}")
        normalized = " ".join(question.split()).casefold()
        if normalized in seen_texts:
            raise DatasetError(f"duplicate normalized question at {index}")
        seen_texts.add(normalized)
        category = item["category"]
        if not isinstance(category, str) or category not in categories:
            raise DatasetError(f"unknown category at {index}")
        category_counts[category] += 1
        difficulty = item["difficulty"]
        if not isinstance(difficulty, str) or difficulty not in {"easy", "medium", "hard"}:
            raise DatasetError(f"invalid difficulty at {index}")
        difficulty_counts[difficulty] += 1
        primary = item["primary_source_id"]
        if not isinstance(primary, str) or primary not in active:
            raise DatasetError(f"unknown or inactive primary source at {index}")
        accepted = item["acceptable_source_ids"]
        if (
            not isinstance(accepted, list)
            or not accepted
            or any(
                not isinstance(source_id, str) or source_id not in active for source_id in accepted
            )
            or len(accepted) != len(set(accepted))
            or primary not in accepted
        ):
            raise DatasetError(f"invalid acceptable sources at {index}")
        pages = item["expected_pages"]
        if pages is not None:
            if active[primary].source_type != "pdf":
                raise DatasetError(f"HTML primary source cannot have page labels at {index}")
            if (
                not isinstance(pages, list)
                or not pages
                or any(type(page) is not int or page < 1 for page in pages)
                or len(pages) != len(set(pages))
            ):
                raise DatasetError(f"invalid expected pages at {index}")
        note = item["evidence_note"]
        if not isinstance(note, str) or not note.strip():
            raise DatasetError(f"empty evidence note at {index}")
    if difficulty_counts != {"easy": 25, "medium": 35, "hard": 20}:
        raise DatasetError("difficulty distribution must be 25/35/20")
    if any(category_counts[category] < 5 for category in REQUIRED_CATEGORIES):
        raise DatasetError("required category has fewer than five questions")
    if any(count > 20 for count in category_counts.values()):
        raise DatasetError("category has more than twenty questions")
    return raw


def load_dataset(path: Path, manifest: Manifest) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot read dataset {path}: {exc}") from exc
    return validate_dataset(raw, manifest)


def validate_page_labels(dataset: dict[str, Any], records: list[dict[str, Any]]) -> None:
    available = {(record["source_id"], record["page_start"]) for record in records}
    for question in dataset["questions"]:
        pages = question["expected_pages"]
        if pages is not None:
            for page in pages:
                if (question["primary_source_id"], page) not in available:
                    raise DatasetError(
                        f"{question['question_id']} references missing indexed page {page}"
                    )
