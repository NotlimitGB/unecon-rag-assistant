"""Compare frozen dense top-five results with reranked dense candidates."""

import json
from pathlib import Path
from typing import Any

from app.evaluation.runner import _first_rank, evaluate_retrieval
from app.retrieval.corpus import RetrievalError
from app.retrieval.reranker import RerankedRetrievalSession

EXPERIMENT_ID = "dense-vs-bge-reranker-v2-m3-v1"
BASELINE = {
    "primary_recall_at_1": 0.7,
    "primary_recall_at_3": 0.775,
    "primary_recall_at_5": 0.8625,
    "primary_mrr_at_5": 0.7525,
}
BASELINE_MISSES = {
    "ret-006",
    "ret-037",
    "ret-038",
    "ret-039",
    "ret-042",
    "ret-046",
    "ret-047",
    "ret-051",
    "ret-061",
    "ret-066",
    "ret-071",
}


class _CachedSession:
    def __init__(self, metadata: dict[str, Any], results: dict[str, list[dict[str, Any]]]):
        self.metadata = metadata
        self.results = results

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.results[query][:top_k]


def _status(before: int | None, after: int | None) -> str:
    left = before if before is not None else 6
    right = after if after is not None else 6
    return "improved" if right < left else "worsened" if right > left else "unchanged"


def _delta(before: dict, after: dict) -> dict:
    return {
        key: (None if value is None else after[key] - value)
        for key, value in before.items()
        if key not in {"page_labeled_questions", "question_count"}
    }


def _group_comparison(before: dict, after: dict) -> dict:
    return {
        key: {
            "question_count": values["question_count"],
            "dense": values,
            "reranked": after[key],
            "deltas": _delta(values, after[key]),
        }
        for key, values in before.items()
    }


def compare(
    dataset: dict[str, Any],
    dense_session,
    reranker_factory=None,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "auto",
    batch_size: int = 8,
    max_length: int = 512,
    candidate_k: int = 20,
    require_baseline: bool = True,
) -> tuple[dict[str, Any], RerankedRetrievalSession]:
    if candidate_k != 20:
        raise RetrievalError("comparison requires dense candidate_k=20")
    pools = {
        question["question"]: dense_session.search(question["question"], candidate_k)
        for question in dataset["questions"]
    }
    dense = evaluate_retrieval(
        dataset,
        _CachedSession(dense_session.metadata, pools),
    )
    misses = {row["question_id"] for row in dense["questions"] if row["primary_rank"] is None}
    if require_baseline and (
        any(abs(dense["metrics"][key] - value) > 1e-9 for key, value in BASELINE.items())
        or misses != BASELINE_MISSES
    ):
        raise RetrievalError("dense results do not reproduce the frozen Task006 baseline")
    reranked_session = RerankedRetrievalSession(
        dense_session, model_name, device, batch_size, max_length, candidate_k, reranker_factory
    )
    reranked_results = {
        question["question"]: reranked_session.rerank(
            question["question"], pools[question["question"]]
        )
        for question in dataset["questions"]
    }
    reranked = evaluate_retrieval(dataset, _CachedSession(dense_session.metadata, reranked_results))
    candidate_ranks = {"primary": [], "accepted": [], "page": []}
    rows = []
    for question, dense_row, reranked_row in zip(
        dataset["questions"], dense["questions"], reranked["questions"], strict=True
    ):
        pool = pools[question["question"]]
        primary = question["primary_source_id"]
        accepted = question["acceptable_source_ids"]
        pages = question["expected_pages"]
        candidate_ranks["primary"].append(
            _first_rank(pool, lambda hit, primary=primary: hit["source_id"] == primary)
        )
        candidate_ranks["accepted"].append(
            _first_rank(pool, lambda hit, accepted=accepted: hit["source_id"] in accepted)
        )
        if pages is not None:
            candidate_ranks["page"].append(
                _first_rank(
                    pool,
                    lambda hit, primary=primary, pages=pages: (
                        hit["source_id"] == primary and hit["page_start"] in pages
                    ),
                )
            )
        rows.append(
            {
                "question_id": question["question_id"],
                "question": question["question"],
                "category": question["category"],
                "difficulty": question["difficulty"],
                "primary_source_id": primary,
                "expected_pages": pages,
                "dense_primary_rank": dense_row["primary_rank"],
                "reranked_primary_rank": reranked_row["primary_rank"],
                "candidate_primary_rank": candidate_ranks["primary"][-1],
                "dense_accepted_rank": dense_row["accepted_rank"],
                "reranked_accepted_rank": reranked_row["accepted_rank"],
                "dense_page_rank": dense_row["page_rank"],
                "reranked_page_rank": reranked_row["page_rank"],
                "dense_top5": dense_row["top_results"],
                "reranked_top5": reranked_row["top_results"],
                "status": _status(dense_row["primary_rank"], reranked_row["primary_rank"]),
            }
        )
    candidate_recall = {
        f"{key}_candidate_recall_at_20": (
            sum(rank is not None for rank in ranks) / len(ranks) if ranks else None
        )
        for key, ranks in candidate_ranks.items()
    }
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "dataset": dense["dataset"],
        "dense": {
            "model": dense["retrieval"]["model"],
            "vector_count": dense["retrieval"]["vector_count"],
            "metrics": dense["metrics"],
        },
        "reranked": {
            "model": model_name,
            "candidate_k": candidate_k,
            "final_k": 5,
            "metrics": reranked["metrics"],
            "pairs_scored": reranked_session.pairs_scored,
        },
        "candidate_recall": candidate_recall,
        "deltas": _delta(dense["metrics"], reranked["metrics"]),
        "category_comparison": _group_comparison(
            dense["category_metrics"], reranked["category_metrics"]
        ),
        "difficulty_comparison": _group_comparison(
            dense["difficulty_metrics"], reranked["difficulty_metrics"]
        ),
        "questions": rows,
    }
    return report, reranked_session


def render_comparison(report: dict[str, Any]) -> str:
    lines = [
        "# Эксперимент: плотный поиск и BGE reranker",
        "",
        f"Датасет: {report['dataset']['dataset_id']}; "
        f"вопросов: {report['dataset']['question_count']}.",
        f"Пары: {report['reranked']['pairs_scored']}; "
        f"кандидатов: {report['reranked']['candidate_k']}.",
        "",
        "| Метрика | Dense | Reranked | Δ |",
        "|---|---:|---:|---:|",
    ]
    for key, old in report["dense"]["metrics"].items():
        if key == "page_labeled_questions":
            continue
        new = report["reranked"]["metrics"][key]
        delta = report["deltas"][key]
        lines.append(
            f"| {key} | {old:.4f} | {new:.4f} | {delta:+.4f} |"
            if old is not None
            else f"| {key} | N/A | N/A | N/A |"
        )
    lines.extend(["", "## Предел кандидатов top-20", ""])
    for key, value in report["candidate_recall"].items():
        lines.append(f"- {key}: {value:.4f}" if value is not None else f"- {key}: N/A")
    for heading, field in (
        ("Категории", "category_comparison"),
        ("Сложность", "difficulty_comparison"),
    ):
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                "| Группа | Вопросов | Dense R@1/3/5, MRR | Reranked R@1/3/5, MRR | Δ R@5 |",
                "|---|---:|---|---|---:|",
            ]
        )
        for key, item in report[field].items():

            def values(metrics):
                return "/".join(
                    f"{metrics[f'primary_{name}']:.4f}"
                    for name in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr_at_5")
                )

            lines.append(
                f"| {key} | {item['question_count']} | {values(item['dense'])} | "
                f"{values(item['reranked'])} | "
                f"{item['deltas']['primary_recall_at_5']:+.4f} |"
            )
    sections = (
        (
            "Восстановленные промахи",
            lambda r: r["dense_primary_rank"] is None and r["reranked_primary_rank"] is not None,
        ),
        (
            "Новые промахи",
            lambda r: r["dense_primary_rank"] is not None and r["reranked_primary_rank"] is None,
        ),
        ("Улучшенные ранги", lambda r: r["status"] == "improved"),
        ("Ухудшенные ранги", lambda r: r["status"] == "worsened"),
        ("Источники вне top-20", lambda r: r["candidate_primary_rank"] is None),
    )
    for heading, predicate in sections:
        lines.extend(["", f"## {heading}", ""])
        matching = [r for r in report["questions"] if predicate(r)]
        lines.extend(
            f"- {r['question_id']}: dense={r['dense_primary_rank']}, "
            f"reranked={r['reranked_primary_rank']}, candidate={r['candidate_primary_rank']}"
            for r in matching
        )
        if not matching:
            lines.append("Нет.")
    lines.extend(["", "## Прежние 11 промахов", ""])
    for row in report["questions"]:
        if row["question_id"] in BASELINE_MISSES:
            lines.append(
                f"- {row['question_id']}: candidate={row['candidate_primary_rank']}, "
                f"reranked={row['reranked_primary_rank']}"
            )
    entrance = report["category_comparison"].get("entrance_exams")
    if entrance:
        rows = [r for r in report["questions"] if r["category"] == "entrance_exams"]
        lines.extend(["", "## Вступительные испытания", ""])
        for mode in ("dense", "reranked"):
            lines.append(
                f"- {mode}: "
                + ", ".join(
                    f"{key}={value:.4f}"
                    for key, value in entrance[mode].items()
                    if key.startswith("primary_")
                )
            )
        lines.append(
            f"- improved={sum(r['status'] == 'improved' for r in rows)}, "
            f"worsened={sum(r['status'] == 'worsened' for r in rows)}"
        )
    return "\n".join(lines) + "\n"


def write_comparison(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "reranker_comparison.json"
    md_path = output_dir / "reranker_comparison.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_comparison(report), encoding="utf-8")
    return json_path, md_path
