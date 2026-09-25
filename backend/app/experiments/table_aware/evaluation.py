"""Frozen-question selection and separate representation/retrieval diagnostics."""

from collections import Counter
from typing import Any

from app.experiments.table_aware.extract import ExperimentError, normalize

TARGET_IDS = (
    "admission-capacity-pdf",
    "entrance-exams-list-pdf",
    "tuition-order-128-pdf",
)

EXPECTED_IDS = (
    "gen-007",
    "gen-008",
    "gen-009",
    "gen-010",
    "gen-011",
    "gen-012",
    "gen-019",
    "gen-020",
    "gen-021",
    "gen-022",
    "gen-023",
    "gen-025",
    "gen-027",
    "gen-028",
    "gen-029",
    "gen-030",
)
PROBE_IDS = ("gen-008", "gen-011", "gen-021", "gen-027")


def select_questions(generation: dict[str, Any], retrieval: dict[str, Any]) -> list[dict[str, Any]]:
    gold = {item["question_id"]: item for item in retrieval["questions"]}
    selected = []
    for item in generation["questions"]:
        if item["expected_status"] != "answered":
            continue
        reference = gold[item["retrieval_question_id"]]
        if reference["primary_source_id"] in TARGET_IDS:
            selected.append(
                {
                    "question_id": item["question_id"],
                    "question": item["question"],
                    "source_id": reference["primary_source_id"],
                    "expected_pages": reference["expected_pages"],
                }
            )
    if tuple(item["question_id"] for item in selected) != EXPECTED_IDS:
        raise ExperimentError("frozen table-question selection changed")
    if any(not item["expected_pages"] for item in selected):
        raise ExperimentError("table question has no verified expected page")
    return selected


def page_rank(hits: list[dict[str, Any]], source_id: str, pages: list[int]) -> int | None:
    return next(
        (
            rank
            for rank, hit in enumerate(hits, 1)
            if hit["source_id"] == source_id and hit["page_start"] in pages
        ),
        None,
    )


def page_recall(ranks: list[int | None]) -> dict[str, float]:
    if not ranks:
        raise ExperimentError("cannot evaluate an empty question set")
    return {
        f"page_recall_at_{k}": sum(rank is not None and rank <= k for rank in ranks) / len(ranks)
        for k in (1, 3, 5)
    }


def movement(
    production: dict[str, int | None], experimental: dict[str, int | None]
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"improved": [], "worsened": [], "unchanged": []}
    if production.keys() != experimental.keys():
        raise ExperimentError("retrieval question sets differ")
    for question_id, old in production.items():
        new = experimental[question_id]
        bucket = (
            "improved"
            if (new or 6) < (old or 6)
            else "worsened"
            if (new or 6) > (old or 6)
            else "unchanged"
        )
        result[bucket].append(question_id)
    return result


def _probe_row(
    artifact: dict[str, Any], question_id: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Match stable program identity first; expectations live only in evaluation."""
    identity = {
        "gen-008": ("38.03.02", "Менеджмент", 2),
        "gen-011": ("38.03.06", "Торговое дело", 3),
        "gen-021": ("38.03.01", "Экономика", 2),
        "gen-027": ("38.03.01", "Экономика", 2),
    }[question_id]
    for table in artifact["tables"]:
        if table["page"] != identity[2]:
            continue
        for row in table["rows"]:
            cells = [normalize(cell) for cell in row["raw_cells"]]
            joined = " ".join(cells)
            if identity[0] not in joined or identity[1] not in joined:
                continue
            if question_id == "gen-021" and "Математика" not in joined:
                continue
            if question_id == "gen-027" and "179 500" not in cells:
                continue
            if question_id in {"gen-008", "gen-011"} and "Всего мест" not in joined:
                continue
            return table, row
    return None


def evaluate_probes(
    artifacts: list[dict[str, Any]],
    dense: dict[str, list[dict[str, Any]]],
    reranked: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_source = {artifact["source"]["id"]: artifact for artifact in artifacts}
    source = {
        "gen-008": "admission-capacity-pdf",
        "gen-011": "admission-capacity-pdf",
        "gen-021": "entrance-exams-list-pdf",
        "gen-027": "tuition-order-128-pdf",
    }
    results = []
    for question_id in PROBE_IDS:
        found = _probe_row(by_source[source[question_id]], question_id)
        row = found[1] if found else None
        pairs = row["values"] if row else []
        required = {
            "gen-008": ("36", "686", "105"),
            "gen-011": ("3", "20", "8"),
            "gen-021": ("40", "60"),
            "gen-027": ("179 500",),
        }[question_id]
        matched = [pair for value in required for pair in pairs if pair["value"] == value]
        labels = [pair["label"] for pair in matched]
        column_mapping = (
            row is not None
            and len(matched) >= len(required)
            and len(set(pair["column_id"] for pair in matched)) >= len(required)
            and all(not label.startswith("column_") for label in labels)
        )
        period = bool(row and "семестр" in row["text"].casefold())
        if question_id == "gen-008":
            column_mapping = (
                column_mapping
                and any("общий конкурс" in label for label in labels)
                and any("отдельный" in label for label in labels)
            )
        elif question_id == "gen-011":
            column_mapping = (
                column_mapping
                and any("общий конкурс" in label for label in labels)
                and any("отдельный" in label for label in labels)
            )
        elif question_id == "gen-021":
            column_mapping = (
                column_mapping
                and any("платные" in label for label in labels)
                and any("бюджет" in label for label in labels)
            )
        row_id = row["row_id"] if row else None
        dense_rank = next(
            (i for i, hit in enumerate(dense[question_id], 1) if hit["chunk_id"] == row_id), None
        )
        reranked_rank = next(
            (i for i, hit in enumerate(reranked[question_id], 1) if hit["chunk_id"] == row_id), None
        )
        passed = bool((period if question_id == "gen-027" else column_mapping) and reranked_rank)
        results.append(
            {
                "question_id": question_id,
                "row_id": row_id,
                "raw_cells": row["raw_cells"] if row else None,
                "column_value_pairs": matched,
                "unambiguous_column_mapping": bool(column_mapping),
                "period_context_preserved": period,
                "dense_row_rank_top20": dense_rank,
                "reranked_row_rank_top5": reranked_rank,
                "passed": passed,
            }
        )
    return results


def summarize(
    questions: list[dict[str, Any]],
    production: dict[str, list[dict[str, Any]]],
    experimental: dict[str, list[dict[str, Any]]],
    probes: list[dict[str, Any]],
    source_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for q in questions:
        question_id = q["question_id"]
        old, new = production[question_id], experimental[question_id]
        rows.append(
            {
                "question_id": question_id,
                "question": q["question"],
                "primary_source_id": q["source_id"],
                "expected_pages": q["expected_pages"],
                "production_primary_source_rank": next(
                    (i for i, h in enumerate(old, 1) if h["source_id"] == q["source_id"]), None
                ),
                "production_expected_page_rank": page_rank(
                    old, q["source_id"], q["expected_pages"]
                ),
                "experimental_expected_page_rank": page_rank(
                    new, q["source_id"], q["expected_pages"]
                ),
                "production_top5": [
                    {
                        "chunk_id": h["chunk_id"],
                        "source_id": h["source_id"],
                        "page": h["page_start"],
                    }
                    for h in old
                ],
                "experimental_top5": [
                    {"row_id": h["chunk_id"], "source_id": h["source_id"], "page": h["page_start"]}
                    for h in new
                ],
            }
        )
    old_ranks = {row["question_id"]: row["production_expected_page_rank"] for row in rows}
    new_ranks = {row["question_id"]: row["experimental_expected_page_rank"] for row in rows}
    old_metrics = page_recall(list(old_ranks.values()))
    new_metrics = page_recall(list(new_ranks.values()))
    if not all(check["sha_match"] for check in source_checks):
        raise ExperimentError("source version mismatch")
    if len(probes) != 4 or not all(probe["passed"] for probe in probes):
        verdict = "inconclusive"
    elif new_metrics["page_recall_at_5"] < old_metrics["page_recall_at_5"]:
        verdict = "inconclusive"
    else:
        verdict = "promising"
    return {
        "questions": rows,
        "metrics": {"production": old_metrics, "experimental": new_metrics},
        "movement": movement(old_ranks, new_ranks),
        "verdict": verdict,
        "source_counts": dict(Counter(row["primary_source_id"] for row in rows)),
    }
