"""Pure rank-based retrieval metrics."""

from collections import defaultdict
from typing import Any


def rank_metrics(ranks: list[int | None]) -> dict[str, float | None]:
    if not ranks:
        return {"recall_at_1": None, "recall_at_3": None, "recall_at_5": None, "mrr_at_5": None}
    size = len(ranks)
    return {
        "recall_at_1": sum(rank is not None and rank <= 1 for rank in ranks) / size,
        "recall_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / size,
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / size,
        "mrr_at_5": sum(1 / rank for rank in ranks if rank is not None and rank <= 5) / size,
    }


def grouped_primary_metrics(
    rows: list[dict[str, Any]], group_field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[int | None]] = defaultdict(list)
    for row in rows:
        groups[row[group_field]].append(row["primary_rank"])
    return {
        group: {
            "question_count": len(ranks),
            **{f"primary_{name}": value for name, value in rank_metrics(ranks).items()},
        }
        for group, ranks in sorted(groups.items())
    }
