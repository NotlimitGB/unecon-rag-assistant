"""Strict on-disk release, inventory, validation, and pointer contracts."""

import math
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.corpus.paths import safe_id

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("schema_version", check_fields=False, mode="before")
    @classmethod
    def integer_schema(cls, value):
        if type(value) is not int:
            raise ValueError("schema version must be integer")
        return value

    @field_validator("created_at", "updated_at", check_fields=False)
    @classmethod
    def utc_time(cls, value: str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if not value.endswith("Z") or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("timestamp must be UTC with Z suffix")
        return value


class State(StrictModel):
    schema_version: Literal[1]
    generation: int = Field(gt=0)
    current_release: str
    published_releases: list[str]
    updated_at: str
    operation: Literal["bootstrap", "publish", "rollback"]

    @model_validator(mode="after")
    def identifiers(self):
        safe_id(self.current_release)
        for value in self.published_releases:
            safe_id(value)
        if len(set(self.published_releases)) != len(self.published_releases):
            raise ValueError("duplicate published release")
        if self.current_release not in self.published_releases:
            raise ValueError("current release must have been published")
        return self


class InventoryEntry(StrictModel):
    path: str
    size: int = Field(ge=0)
    sha256: Digest

    @field_validator("path")
    @classmethod
    def relative_path(cls, value):
        if "\\" in value or ":" in value or value.startswith("/"):
            raise ValueError("inventory requires relative POSIX paths")
        if any(p in ("", ".", "..") for p in value.split("/")):
            raise ValueError("invalid inventory path")
        return value


class IncomingRecord(StrictModel):
    package_id: str
    metadata_sha256: Digest
    document_sha256: Digest
    source_id: str

    @field_validator("package_id", "source_id")
    @classmethod
    def identifier(cls, value):
        return safe_id(value)


class ModelSettings(StrictModel):
    embedding_model: str = Field(min_length=1)
    reranker_model: str = Field(min_length=1)
    reranker_max_length: int = Field(ge=32, le=4096)
    reranker_candidate_k: Literal[20]


class Release(StrictModel):
    schema_version: Literal[1]
    release_id: str
    base_release: str | None
    base_generation: int | None = Field(ge=1)
    created_at: str
    registry_sha256: Digest
    incoming: list[IncomingRecord]
    changed_families: list[str]
    active_source_ids: list[str]
    document_count: int = Field(gt=0)
    chunk_count: int = Field(gt=0)
    table_row_count: int = Field(gt=0)
    models: ModelSettings
    inventory: list[InventoryEntry]
    release_digest: Digest
    sealed: bool
    validation_sha256: Digest | None

    @model_validator(mode="after")
    def consistent(self):
        safe_id(self.release_id)
        if self.base_release is not None:
            safe_id(self.base_release)
            if self.base_release == self.release_id:
                raise ValueError("release cannot be its own base")
        if (self.base_release is None) != (self.base_generation is None):
            raise ValueError("base ID and generation must appear together")
        for values in (self.active_source_ids, self.changed_families):
            if len(values) != len(set(values)):
                raise ValueError("duplicate release identifiers")
            for value in values:
                safe_id(value)
        paths = [entry.path for entry in self.inventory]
        if paths != sorted(set(paths)):
            raise ValueError("inventory must be sorted and unique")
        if not self.active_source_ids or len(self.active_source_ids) > self.document_count:
            raise ValueError("invalid active source count")
        for key in ("package_id", "source_id"):
            values = [getattr(package, key) for package in self.incoming]
            if len(values) != len(set(values)):
                raise ValueError("duplicate incoming package or source")
        if self.sealed != (self.validation_sha256 is not None):
            raise ValueError("sealed release requires validation SHA")
        return self


class Validation(StrictModel):
    schema_version: Literal[1]
    release_id: str
    release_digest: Digest
    base_release: str | None
    created_at: str
    kind: Literal["bootstrap", "candidate"]
    publishable: bool
    checks: dict[str, bool]
    errors: list[str]
    # Detailed retrieval diagnostics have the existing evaluation report contracts.
    comparison: dict

    @model_validator(mode="after")
    def passed_checks(self):
        allowed = (
            {"local_corpus", "copy_verified"}
            if self.kind == "bootstrap"
            else {
                "corpus_and_transitions",
                "historical_dataset",
                "all_80_queries_with_five_results",
                "incoming_unchanged",
            }
        )
        if set(self.checks) - allowed:
            raise ValueError("unknown validation check")
        if self.publishable and (
            self.errors or set(self.checks) != allowed or not all(self.checks.values())
        ):
            raise ValueError("publishable validation requires all checks")
        if self.kind == "bootstrap":
            _exact(self.comparison, {"note"})
            if not isinstance(self.comparison["note"], str):
                raise ValueError("invalid bootstrap note")
        elif self.comparison:
            allowed_comparison = {
                "base",
                "candidate",
                "changed_question_ids",
                "affected_categories",
                "count_deltas",
                "changed_families",
            }
            if self.publishable:
                _exact(self.comparison, allowed_comparison)
            elif set(self.comparison) - allowed_comparison:
                raise ValueError("unknown comparison field")
            if not {"base", "candidate", "changed_question_ids", "affected_categories"} <= set(
                self.comparison
            ):
                raise ValueError("incomplete comparison evidence")
            for name in ("base", "candidate"):
                _validate_retrieval_report(self.comparison[name])
            if "count_deltas" in self.comparison:
                _exact(self.comparison["count_deltas"], {"chunks", "table_rows"})
                if any(type(n) is not int for n in self.comparison["count_deltas"].values()):
                    raise ValueError("count deltas must be integers")
        elif self.publishable:
            raise ValueError("candidate requires comparison evidence")
        return self


def _exact(value, keys):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("invalid comparison schema")


def _validate_retrieval_report(report):
    """Check the nested, existing evaluation contract without accepting extension fields."""
    _exact(
        report,
        {
            "schema_version",
            "dataset",
            "retrieval",
            "metrics",
            "category_metrics",
            "difficulty_metrics",
            "questions",
        },
    )
    if type(report["schema_version"]) is not int or report["schema_version"] != 1:
        raise ValueError("invalid evaluation schema")
    _exact(report["dataset"], {"dataset_id", "question_count"})
    if report["dataset"]["question_count"] != 80:
        raise ValueError("validation requires 80 questions")
    _exact(report["retrieval"], {"model", "index_type", "top_k", "vector_count"})
    if report["retrieval"]["top_k"] != 5 or report["retrieval"]["index_type"] != "IndexFlatIP":
        raise ValueError("invalid validation retrieval configuration")
    names = {
        f"{label}_{metric}"
        for label in ("primary", "accepted", "page")
        for metric in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr_at_5")
    }
    _exact(report["metrics"], names | {"page_labeled_questions"})
    for field in ("category_metrics", "difficulty_metrics"):
        if not isinstance(report[field], dict):
            raise ValueError("invalid metric grouping")
        for group in report[field].values():
            _exact(
                group,
                {
                    "question_count",
                    "primary_recall_at_1",
                    "primary_recall_at_3",
                    "primary_recall_at_5",
                    "primary_mrr_at_5",
                },
            )
    questions = report["questions"]
    if not isinstance(questions, list) or len(questions) != 80:
        raise ValueError("validation requires all 80 question records")
    for index, question in enumerate(questions, 1):
        _exact(
            question,
            {
                "question_id",
                "question",
                "category",
                "difficulty",
                "primary_source_id",
                "acceptable_source_ids",
                "expected_pages",
                "primary_rank",
                "accepted_rank",
                "page_rank",
                "top_results",
            },
        )
        if question["question_id"] != f"ret-{index:03d}" or len(question["top_results"]) != 5:
            raise ValueError("invalid validation question order or result count")
        for rank, hit in enumerate(question["top_results"], 1):
            _exact(hit, {"rank", "score", "source_id", "chunk_id", "page", "text_excerpt"})
            if (
                type(hit["rank"]) is not int
                or hit["rank"] != rank
                or type(hit["score"]) not in (float, int)
                or not math.isfinite(hit["score"])
            ):
                raise ValueError("invalid validation result rank or score")
