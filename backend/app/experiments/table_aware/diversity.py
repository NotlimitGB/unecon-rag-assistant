"""Frozen post-reranker PDF-page diversity experiment; production retrieval is untouched."""

from collections import Counter, deque
from typing import Any

from app.experiments.table_aware.augmented import baseline_matches
from app.experiments.table_aware.dual_channel import new_misses
from app.experiments.table_aware.extract import ExperimentError
from app.retrieval.service import RetrievalResponse, RetrievedChunk

PDF_PAGE_LIMIT = 2
FINAL_K = 5


def select_with_pdf_page_diversity(ranked: list[dict[str, Any]]) -> dict[str, Any]:
    """Accept in original reranker order, with one fixed cap per PDF source/page."""
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    accepted_ranks: list[int] = []
    page_counts: Counter[tuple[str, int]] = Counter()
    for rank, hit in enumerate(ranked, 1):
        source_type = hit["source_type"]
        page = hit["page_start"]
        if source_type == "pdf":
            if type(page) is not int or page < 1:
                raise ExperimentError("PDF diversity candidate requires a positive page")
            key = (hit["source_id"], page)
            if page_counts[key] >= PDF_PAGE_LIMIT:
                skipped.append(
                    {
                        "chunk_id": hit["chunk_id"],
                        "original_reranker_rank": rank,
                        "source_id": hit["source_id"],
                        "page": page,
                        "representation": hit["representation"],
                    }
                )
                continue
            page_counts[key] += 1
        elif source_type != "html" or page is not None:
            raise ExperimentError("invalid source type or HTML page in diversity candidate")
        selected.append(hit)
        accepted_ranks.append(rank)
        if len(selected) == FINAL_K:
            break
    replacements = deque(
        {
            "chunk_id": hit["chunk_id"],
            "original_reranker_rank": rank,
            "source_id": hit["source_id"],
            "page": hit["page_start"],
            "representation": hit["representation"],
        }
        for hit, rank in zip(selected, accepted_ranks, strict=True)
        if rank > FINAL_K
    )
    return {
        "selected": selected,
        "selected_original_ranks": accepted_ranks,
        "skipped": skipped,
        "replacement_pairs": [
            {"skipped": item, "replacement": replacements.popleft()}
            for item in skipped
            if item["original_reranker_rank"] <= FINAL_K and replacements
        ],
    }


def saturation(hits: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(hit["source_id"] for hit in hits)
    pdf_pages = Counter(
        (hit["source_id"], hit["page_start"]) for hit in hits if hit["source_type"] == "pdf"
    )
    table_pages = Counter(
        (hit["source_id"], hit["page_start"])
        for hit in hits
        if hit["source_type"] == "pdf" and hit["representation"] == "table_row"
    )
    return {
        "three_same_source": any(n >= 3 for n in source_counts.values()),
        "three_same_pdf_page": any(n >= 3 for n in pdf_pages.values()),
        "three_table_same_pdf_page": any(n >= 3 for n in table_pages.values()),
        "max_same_pdf_page": max(pdf_pages.values(), default=0),
    }


def retrieval_gate(
    production: dict[str, Any],
    raw_dual_reproduced: bool,
    reranker_unchanged: bool,
    final_sizes_valid: bool,
    mandatory_rows_selected: bool,
    diversity: dict[str, Any],
    table16: dict[str, Any],
    production_ranks: dict[str, dict[str, int | None]],
    diversity_ranks: dict[str, dict[str, int | None]],
    non_table_ids: set[str],
    labeled_ids: set[str],
) -> dict[str, bool]:
    """The 17 preregistered Phase A gates; recoveries never offset new losses."""
    return {
        "01_production_baseline": baseline_matches(production),
        "02_raw_task016": raw_dual_reproduced,
        "03_reranker_unchanged": reranker_unchanged,
        "04_final_size": final_sizes_valid,
        "05_mandatory_rows": mandatory_rows_selected,
        "06_overall_primary_r5": diversity["primary_hits_at_5"] >= 70,
        "07_overall_accepted_r5": diversity["accepted_hits_at_5"] >= 71,
        "08_overall_page_r5": diversity["page_hits_at_5"] >= 47,
        "09_table_primary_r5": table16["primary_hits_at_5"] == 16,
        "10_table_accepted_r5": table16["accepted_hits_at_5"] == 16,
        "11_table_page_r5": table16["page_hits_at_5"] >= 13,
        "12_non_table_primary_lossless": not new_misses(
            production_ranks, diversity_ranks, non_table_ids, "primary"
        ),
        "13_non_table_accepted_lossless": not new_misses(
            production_ranks, diversity_ranks, non_table_ids, "accepted"
        ),
        "14_labeled_page_lossless": not new_misses(
            production_ranks, diversity_ranks, labeled_ids, "page"
        ),
        "15_ret020_page": diversity_ranks["ret-020"]["page"] is not None,
        "16_ret044_page": diversity_ranks["ret-044"]["page"] is not None,
        "17_ret052_primary": diversity_ranks["ret-052"]["primary"] is not None,
    }


class CachedDiversityAdapter:
    """Expose only Phase A's validated top-5 to the unchanged AnswerService."""

    def __init__(self, hits_by_question: dict[str, list[dict[str, Any]]]):
        self.hits_by_question = hits_by_question
        self.history: dict[str, list[dict[str, Any]]] = {}

    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResponse:
        count = FINAL_K if top_k is None else top_k
        if type(count) is not int or not 1 <= count <= FINAL_K:
            raise ExperimentError("invalid diversity top_k")
        if question not in self.hits_by_question:
            raise ExperimentError("question is absent from frozen Phase A results")
        hits = self.hits_by_question[question]
        if len(hits) != FINAL_K:
            raise ExperimentError("frozen diversity result is incomplete")
        chosen = hits[:count]
        self.history[question] = chosen
        return RetrievalResponse(
            query=question,
            mode="reranked",
            top_k=count,
            results=[
                RetrievedChunk(
                    rank=rank,
                    chunk_id=hit["chunk_id"],
                    text=hit["text"],
                    source_id=hit["source_id"],
                    source_title=hit["source_title"],
                    source_url=hit["final_url"],
                    source_type=hit["source_type"],
                    category=hit["category"],
                    admission_year=hit["admission_year"],
                    page=hit["page_start"],
                    score=hit["score"],
                    dense_score=hit["dense_score"],
                    rerank_score=hit["rerank_score"],
                )
                for rank, hit in enumerate(chosen, 1)
            ],
        )
