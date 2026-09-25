"""Read-only diagnostics for the frozen dual-channel reranker competition."""

import re
import statistics
from collections import Counter
from typing import Any

from app.experiments.table_aware.extract import ExperimentError
from app.retrieval.reranker import rerank_candidates

KINDS = ("production_chunk", "table_row")
WORD = re.compile(r"\w+", re.UNICODE)
NUMBER = re.compile(r"(?<!\w)\d+(?:[\s\u00a0\u202f]\d{3})*(?:[.,]\d+)?%?(?!\w)", re.UNICODE)
PATTERNS = frozenset(
    {
        "source_saturation",
        "same_page_row_saturation",
        "structured_text_score_advantage",
        "expected_evidence_low_reranker_score",
        "candidate_absence",
        "reranker_truncation_risk",
        "lexical_concentration",
        "near_cutoff_instability",
        "no_clear_single_factor",
    }
)


def lexical_diagnostics(query: str, text: str) -> dict[str, float | int]:
    normalized = " ".join(text.split())
    words = WORD.findall(normalized.casefold())
    word_count = len(normalized.split()) if normalized else 0
    query_words = set(WORD.findall(query.casefold()))
    numbers = NUMBER.findall(text)
    return {
        "char_count": len(text),
        "word_count": word_count,
        "query_token_coverage": len(query_words & set(words)) / len(query_words)
        if query_words
        else 0.0,
        "numeric_token_count": len(numbers),
        "numeric_density": len(numbers) / word_count if word_count else 0.0,
    }


def token_diagnostics(tokenizer: Any, query: str, text: str, max_length: int) -> dict[str, Any]:
    """Ask the model tokenizer for the untruncated pair, never infer tokens from text."""
    encoded = tokenizer(query, text, add_special_tokens=True, truncation=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    length = len(ids)
    return {
        "pair_tokens_before_truncation": length,
        "reranker_max_length": max_length,
        "would_truncate": length > max_length,
        "tokens_over_limit": max(0, length - max_length),
    }


def label_diagnostics(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    values = [pair for pair in row["values"] if pair["value"]]
    return {
        "label_value_pairs": len(values),
        "label_char_ratio": sum(len(pair["label"]) for pair in values) / len(row["text"])
        if row["text"]
        else 0.0,
    }


def trace_candidates(
    question: dict[str, Any],
    merged: list[dict[str, Any]],
    reranker: Any,
    channel_ranks: dict[int, int],
    rows: dict[str, dict[str, Any]],
    tokenizer: Any | None = None,
    max_length: int = 512,
) -> list[dict[str, Any]]:
    if len(merged) != 25 or len({hit["vector_id"] for hit in merged}) != 25:
        raise ExperimentError("expected 25 unique merged candidates")
    ranked = rerank_candidates(question["question"], merged, reranker, 25)
    merged_positions = {hit["vector_id"]: i for i, hit in enumerate(merged, 1)}
    expected_pages = question["expected_pages"]
    result = []
    for rank, hit in enumerate(ranked, 1):
        source_id = hit["source_id"]
        representation = hit["representation"]
        if representation not in KINDS:
            raise ExperimentError("unknown candidate representation")
        entry = {
            "question_id": question["question_id"],
            "representation": representation,
            "vector_id": hit["vector_id"],
            "chunk_id": hit["chunk_id"],
            "source_id": source_id,
            "page": hit["page_start"],
            "text": hit["text"],
            **lexical_diagnostics(question["question"], hit["text"]),
            "dense_channel_rank": channel_ranks[hit["vector_id"]],
            "dense_score": hit["dense_score"],
            "merged_dense_position": merged_positions[hit["vector_id"]],
            "reranker_score": hit["rerank_score"],
            "reranker_rank": rank,
            "in_final_top5": rank <= 5,
            "is_primary_source": source_id == question["primary_source_id"],
            "is_acceptable_source": source_id in question["acceptable_source_ids"],
            "is_expected_page": expected_pages is not None
            and source_id == question["primary_source_id"]
            and hit["page_start"] in expected_pages,
            "label_diagnostics": label_diagnostics(rows.get(hit["chunk_id"]))
            if representation == "table_row"
            else None,
        }
        entry["token_diagnostics"] = (
            token_diagnostics(tokenizer, question["question"], hit["text"], max_length)
            if tokenizer is not None
            else None
        )
        result.append(entry)
    return result


def cutoff_diagnostics(trace: list[dict[str, Any]], page_labeled: bool) -> dict[str, Any]:
    if len(trace) != 25:
        raise ExperimentError("cutoff requires 25 ranked candidates")
    cutoff = trace[4]["reranker_score"]

    def best(field: str) -> dict[str, Any] | None:
        candidate = next((item for item in trace if item[field]), None)
        return (
            {
                "chunk_id": candidate["chunk_id"],
                "rank": candidate["reranker_rank"],
                "score": candidate["reranker_score"],
                "gap_to_rank5": cutoff - candidate["reranker_score"],
            }
            if candidate
            else None
        )

    return {
        "rank1_score": trace[0]["reranker_score"],
        "rank5_score": cutoff,
        "rank6_score": trace[5]["reranker_score"],
        "rank5_rank6_margin": cutoff - trace[5]["reranker_score"],
        "best_primary": best("is_primary_source"),
        "best_expected_page": best("is_expected_page") if page_labeled else None,
    }


def saturation(trace: list[dict[str, Any]]) -> dict[str, Any]:
    top = trace[:5]
    sources = Counter(item["source_id"] for item in top)
    pages = Counter((item["source_id"], item["page"]) for item in top)
    table = [item for item in top if item["representation"] == "table_row"]
    table_sources = Counter(item["source_id"] for item in table)
    table_pages = Counter((item["source_id"], item["page"]) for item in table)
    return {
        "distinct_sources": len(sources),
        "max_same_source": max(sources.values()),
        "max_same_source_page": max(pages.values()),
        "max_table_same_source": max(table_sources.values(), default=0),
        "max_table_same_source_page": max(table_pages.values(), default=0),
        "three_same_source": any(n >= 3 for n in sources.values()),
        "three_table_same_source": any(n >= 3 for n in table_sources.values()),
        "three_table_same_page": any(n >= 3 for n in table_pages.values()),
        "table_ranks_1_5": sum(item["representation"] == "table_row" for item in trace[:5]),
        "table_ranks_6_10": sum(item["representation"] == "table_row" for item in trace[5:10]),
        "table_ranks_11_25": sum(item["representation"] == "table_row" for item in trace[10:]),
    }


def distribution(values: list[float | int]) -> dict[str, float | int]:
    if not values:
        raise ExperimentError("empty diagnostic distribution")
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        return float(
            ordered[lower]
            + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)
        )

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def group_statistics(traces: list[list[dict[str, Any]]]) -> dict[str, Any]:
    candidates = [item for trace in traces for item in trace]
    result = {}
    for kind in KINDS:
        group = [item for item in candidates if item["representation"] == kind]
        token_counts = [
            item["token_diagnostics"]["pair_tokens_before_truncation"]
            for item in group
            if item["token_diagnostics"]
        ]
        result[kind] = {
            "scores": distribution([item["reranker_score"] for item in group]),
            "ranks": distribution([item["reranker_rank"] for item in group]),
            "top5_count": sum(item["in_final_top5"] for item in group),
            "top5_rate": sum(item["in_final_top5"] for item in group) / len(group),
            "mean_char_count": statistics.mean(item["char_count"] for item in group),
            "mean_word_count": statistics.mean(item["word_count"] for item in group),
            "mean_pair_tokens": statistics.mean(token_counts) if token_counts else None,
            "truncated_pairs": sum(
                bool(item["token_diagnostics"] and item["token_diagnostics"]["would_truncate"])
                for item in group
            ),
        }
    result["table_occupancy"] = {
        field: sum(saturation(trace)[field] for trace in traces)
        for field in ("table_ranks_1_5", "table_ranks_6_10", "table_ranks_11_25")
    }
    return result


def validate_patterns(primary: str, contributing: list[str]) -> None:
    if primary not in PATTERNS or any(item not in PATTERNS for item in contributing):
        raise ExperimentError("unknown competition pattern")
    if len(contributing) != len(set(contributing)) or primary in contributing:
        raise ExperimentError("duplicate competition pattern")
