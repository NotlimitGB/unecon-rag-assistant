"""Strict manifest models and the allowed URL boundary."""

import re
from datetime import date
from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)


def validate_official_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise ValueError("invalid URL") from exc
    host = parts.hostname
    if (
        parts.scheme != "https"
        or not host
        or (host != "unecon.ru" and not host.endswith(".unecon.ru"))
        or parts.username is not None
        or parts.password is not None
        or port not in (None, 443)
        or parts.fragment
    ):
        raise ValueError("URL must be HTTPS on unecon.ru or its subdomain")
    return url


class Processing(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    table_aware: StrictBool


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: StrictStr
    logical_document_id: StrictStr
    version: StrictInt = Field(ge=1)
    title: StrictStr
    url: StrictStr
    source_type: Literal["html", "pdf"]
    category: StrictStr
    admission_year: StrictInt = Field(ge=1900, le=2100)
    status: Literal["draft", "active", "superseded"]
    supersedes: StrictStr | None
    published_at: StrictStr | None
    effective_from: StrictStr | None
    effective_to: StrictStr | None
    processing: Processing

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @field_validator("published_at", "effective_from", "effective_to")
    @classmethod
    def valid_date(cls, value: str | None) -> str | None:
        if value is not None:
            parse_date(value)
        return value

    @model_validator(mode="after")
    def valid_processing_and_interval(self) -> "Source":
        if self.processing.table_aware and self.source_type != "pdf":
            raise ValueError("table_aware requires PDF")
        if self.effective_from and self.effective_to and self.effective_from > self.effective_to:
            raise ValueError("effective_from must not exceed effective_to")
        return self

    @field_validator("id", "logical_document_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
            raise ValueError("id must be lowercase kebab-case")
        return value

    @field_validator("title", "category")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must not be empty")
        return value

    @field_validator("url")
    @classmethod
    def official_url(cls, value: str) -> str:
        return validate_official_url(value)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[2]
    sources: list[Source] = Field(min_length=1)

    @field_validator("schema_version", mode="before")
    @classmethod
    def strict_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def unique_source_ids(self) -> "Manifest":
        ids = [source.id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("source ids must be unique")
        versions = [(s.logical_document_id, s.admission_year, s.version) for s in self.sources]
        if len(versions) != len(set(versions)):
            raise ValueError("logical document versions must be unique")
        active = [(s.logical_document_id, s.admission_year) for s in active_sources(self)]
        if len(active) != len(set(active)):
            raise ValueError("only one active version is allowed per logical document and year")
        by_id = {s.id: s for s in self.sources}
        referenced = set()
        for source in self.sources:
            if source.supersedes is None:
                continue
            prior = by_id.get(source.supersedes)
            if (
                prior is None
                or prior.id == source.id
                or (prior.logical_document_id, prior.admission_year)
                != (source.logical_document_id, source.admission_year)
                or prior.version >= source.version
            ):
                raise ValueError(
                    "supersedes must reference an earlier version of the same document"
                )
            # Strictly decreasing versions also make cycles impossible.
            referenced.add(prior.id)
        if any(s.status == "superseded" and s.id not in referenced for s in self.sources):
            raise ValueError("superseded source has no newer descendant")
        return self


def parse_date(value: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("date must be YYYY-MM-DD")
    return date.fromisoformat(value)


def active_sources(manifest: Manifest) -> list[Source]:
    """The single status-based selection rule; dates never activate sources."""
    return [source for source in manifest.sources if source.is_active]


def source_identity(source: Source) -> dict:
    """Immutable version provenance, independent of registry workflow state."""
    return source.model_dump(
        include={
            "id",
            "logical_document_id",
            "version",
            "title",
            "url",
            "source_type",
            "category",
            "admission_year",
        }
    )
