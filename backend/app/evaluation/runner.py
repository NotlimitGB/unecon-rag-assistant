"""Run one top-five retrieval for each question and produce portable reports."""

import json
from pathlib import Path
from typing import Any, Protocol

from app.corpus.paths import protect_saved_release
from app.evaluation.metrics import grouped_primary_metrics, rank_metrics


class SearchSession(Protocol):
    metadata: dict[str, Any]

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]: ...


def _first_rank(results: list[dict[str, Any]], predicate) -> int | None:
    return next((rank for rank, result in enumerate(results, 1) if predicate(result)), None)


def evaluate_retrieval(dataset: dict[str, Any], session: SearchSession) -> dict[str, Any]:
    rows = []
    for question in dataset["questions"]:
        results = session.search(question["question"], top_k=5)
        primary = question["primary_source_id"]
        accepted = question["acceptable_source_ids"]
        pages = question["expected_pages"]
        rows.append(
            {
                **{
                    key: question[key]
                    for key in (
                        "question_id",
                        "question",
                        "category",
                        "difficulty",
                        "primary_source_id",
                        "acceptable_source_ids",
                        "expected_pages",
                    )
                },
                "primary_rank": _first_rank(
                    results, lambda hit, primary=primary: hit["source_id"] == primary
                ),
                "accepted_rank": _first_rank(
                    results, lambda hit, accepted=accepted: hit["source_id"] in accepted
                ),
                "page_rank": (
                    _first_rank(
                        results,
                        lambda hit, primary=primary, pages=pages: (
                            hit["source_id"] == primary and hit["page_start"] in pages
                        ),
                    )
                    if pages is not None
                    else None
                ),
                "top_results": [
                    {
                        "rank": rank,
                        "score": hit["score"],
                        "source_id": hit["source_id"],
                        "chunk_id": hit["chunk_id"],
                        "page": hit["page_start"],
                        "text_excerpt": " ".join(hit["text"].split())[:240],
                    }
                    for rank, hit in enumerate(results[:5], 1)
                ],
            }
        )
    primary_metrics = rank_metrics([row["primary_rank"] for row in rows])
    accepted_metrics = rank_metrics([row["accepted_rank"] for row in rows])
    page_rows = [row for row in rows if row["expected_pages"] is not None]
    page_metrics = rank_metrics([row["page_rank"] for row in page_rows])
    metrics = {
        **{f"primary_{key}": value for key, value in primary_metrics.items()},
        **{f"accepted_{key}": value for key, value in accepted_metrics.items()},
        **{f"page_{key}": value for key, value in page_metrics.items()},
        "page_labeled_questions": len(page_rows),
    }
    return {
        "schema_version": 1,
        "dataset": {"dataset_id": dataset["dataset_id"], "question_count": len(rows)},
        "retrieval": {
            "model": session.metadata["embedding"]["model"],
            "index_type": session.metadata["index"]["type"],
            "top_k": 5,
            "vector_count": session.metadata["index"]["vector_count"],
        },
        "metrics": metrics,
        "category_metrics": grouped_primary_metrics(rows, "category"),
        "difficulty_metrics": grouped_primary_metrics(rows, "difficulty"),
        "questions": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Оценка плотного поиска",
        "",
        f"Датасет: `{report['dataset']['dataset_id']}`; "
        f"вопросов: {report['dataset']['question_count']}.",
        f"Модель: `{report['retrieval']['model']}`; "
        f"индекс: `{report['retrieval']['index_type']}`; "
        f"векторов: {report['retrieval']['vector_count']}.",
        f"Вопросов с меткой страницы PDF: {report['metrics']['page_labeled_questions']}.",
        "",
        "## Общие метрики",
        "",
        "| Метрика | Значение |",
        "|---|---:|",
    ]
    labels = {
        "primary": "Primary",
        "accepted": "Accepted",
        "page": "Page",
    }
    for prefix, label in labels.items():
        for suffix, metric_label in [
            ("recall_at_1", "Recall@1"),
            ("recall_at_3", "Recall@3"),
            ("recall_at_5", "Recall@5"),
            ("mrr_at_5", "MRR@5"),
        ]:
            value = report["metrics"][f"{prefix}_{suffix}"]
            lines.append(
                f"| {label} {metric_label} | {'N/A' if value is None else f'{value:.4f}'} |"
            )
    for heading, key in [
        ("По категориям", "category_metrics"),
        ("По сложности", "difficulty_metrics"),
    ]:
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                "| Группа | Вопросов | Recall@1 | Recall@3 | Recall@5 | MRR@5 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for group, values in report[key].items():
            lines.append(
                f"| {group} | {values['question_count']} | "
                f"{values['primary_recall_at_1']:.4f} | {values['primary_recall_at_3']:.4f} | "
                f"{values['primary_recall_at_5']:.4f} | {values['primary_mrr_at_5']:.4f} |"
            )
    lines.extend(["", "## Промахи Primary top-5", ""])
    misses = [row for row in report["questions"] if row["primary_rank"] is None]
    if not misses:
        lines.append("Нет.")
    for row in misses:
        hits = ", ".join(f"{hit['source_id']} ({hit['score']:.4f})" for hit in row["top_results"])
        lines.append(
            f"- {row['question_id']}: {row['question']} — "
            f"ожидается `{row['primary_source_id']}`; top-5: {hits}."
        )
    lines.extend(["", "## Primary на позициях 4–5", ""])
    low = [row for row in report["questions"] if row["primary_rank"] in (4, 5)]
    if not low:
        lines.append("Нет.")
    for row in low:
        lines.append(f"- {row['question_id']}: позиция {row['primary_rank']}; {row['question']}")
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    protect_saved_release(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "retrieval_report.json"
    md_path = output_dir / "retrieval_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
