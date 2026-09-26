"""Strict, independently reviewed E2E evidence; historical gold stays immutable."""

from app.evaluation.dataset import DatasetError
from app.ingestion.models import active_sources

CASE_IDS = ("gen-001", "gen-007", "gen-019", "gen-026", "gen-027", "gen-049")


def validate_spec(raw: object, manifest, generation: dict) -> list[dict]:
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "cases"}:
        raise DatasetError("invalid smoke specification fields")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise DatasetError("invalid smoke schema version")
    cases = raw["cases"]
    if not isinstance(cases, list) or len(cases) != 6:
        raise DatasetError("smoke requires exactly six cases")
    questions = {q["question_id"]: q for q in generation["questions"]}
    sources = {s.id: s for s in active_sources(manifest)}
    for case, expected_id in zip(cases, CASE_IDS, strict=True):
        if not isinstance(case, dict):
            raise DatasetError("invalid smoke case")
        supported = expected_id != "gen-049"
        keys = {"generation_question_id", "purpose", "expected_status"}
        if supported:
            keys.add("acceptable_evidence")
        if set(case) != keys or case["generation_question_id"] != expected_id:
            raise DatasetError("invalid smoke case fields or order")
        if not isinstance(case["purpose"], str) or not case["purpose"].strip():
            raise DatasetError("empty smoke purpose")
        if case["expected_status"] != questions[expected_id]["expected_status"]:
            raise DatasetError("smoke status disagrees with frozen question")
        if not supported:
            continue
        evidence = case["acceptable_evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise DatasetError("supported smoke case requires evidence")
        seen = set()
        for item in evidence:
            if not isinstance(item, dict) or not isinstance(item.get("source_id"), str):
                raise DatasetError("invalid evidence")
            source_id = item["source_id"]
            if source_id not in sources or source_id in seen:
                raise DatasetError("inactive, unknown or duplicate evidence source")
            seen.add(source_id)
            pdf = sources[source_id].source_type == "pdf"
            if set(item) != ({"source_id", "pages"} if pdf else {"source_id"}):
                raise DatasetError("invalid evidence fields for source type")
            if pdf:
                pages = item["pages"]
                if (
                    not isinstance(pages, list)
                    or not pages
                    or any(type(p) is not int or p < 1 for p in pages)
                    or len(set(pages)) != len(pages)
                ):
                    raise DatasetError("invalid evidence pages")
    return cases


def validate_evidence_pages(cases: list[dict], records: list[dict]) -> None:
    pages = {(r["source_id"], r["page_start"]) for r in records}
    for case in cases:
        for item in case.get("acceptable_evidence", []):
            if any((item["source_id"], page) not in pages for page in item.get("pages", [None])):
                raise DatasetError("smoke evidence source/page absent from validated corpus")
