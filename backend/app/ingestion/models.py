"""Strict manifest models and the allowed URL boundary."""

import re
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


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: StrictStr
    title: StrictStr
    url: StrictStr
    source_type: Literal["html"]
    category: StrictStr
    admission_year: StrictInt = Field(ge=1900, le=2100)
    active: StrictBool

    @field_validator("id")
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

    schema_version: Literal[1]
    sources: list[Source] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_source_ids(self) -> "Manifest":
        ids = [source.id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("source ids must be unique")
        return self
