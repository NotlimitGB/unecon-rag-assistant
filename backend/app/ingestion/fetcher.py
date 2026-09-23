"""Fetch a manifest source without leaving the official domain."""

from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

from app.ingestion.models import Source, validate_official_url

MAX_REDIRECTS = 5
MAX_PDF_BYTES = 25 * 1024 * 1024
BASE_HEADERS = {
    "User-Agent": "unecon-rag-assistant-thesis/0.1",
}


class IngestionError(ValueError):
    """A source could not be safely fetched or normalized."""


@dataclass(frozen=True)
class FetchedPage:
    html: str
    final_url: str


@dataclass(frozen=True)
class FetchedPdf:
    content: bytes
    final_url: str


@dataclass(frozen=True)
class FetchedContent:
    content: bytes
    final_url: str
    content_type: str


def fetch_content(
    source: Source,
    client: httpx.Client,
    *,
    accepted_content_types: set[str],
    accept: str,
    max_bytes: int | None = None,
) -> FetchedContent:
    url = validate_official_url(source.url)
    headers = {**BASE_HEADERS, "Accept": accept}
    for redirect_count in range(MAX_REDIRECTS + 1):
        try:
            with client.stream("GET", url, headers=headers, follow_redirects=False) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise IngestionError("redirect has no Location header")
                    if redirect_count == MAX_REDIRECTS:
                        raise IngestionError("too many redirects")
                    next_url = urljoin(url, location)
                    try:
                        validate_official_url(next_url)
                    except ValueError as exc:
                        raise IngestionError(f"unsafe redirect to {next_url}") from exc
                    url = next_url
                    continue

                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise IngestionError(f"HTTP {response.status_code}") from exc

                content_type_header = response.headers.get("content-type", "")
                content_type = content_type_header.split(";", 1)[0].strip().lower()
                if content_type not in accepted_content_types:
                    raise IngestionError(
                        f"unsupported content type: {content_type or 'missing'}"
                    )

                if max_bytes is not None:
                    content_length = response.headers.get("content-length")
                    if content_length is not None:
                        try:
                            declared_length = int(content_length)
                        except ValueError as exc:
                            raise IngestionError("invalid Content-Length header") from exc
                        if declared_length < 0:
                            raise IngestionError("invalid Content-Length header")
                        if declared_length > max_bytes:
                            raise IngestionError(
                                f"PDF exceeds maximum supported size of {max_bytes} bytes"
                            )

                chunks: list[bytes] = []
                total_size = 0
                for chunk in response.iter_bytes(chunk_size=64 * 1024):
                    total_size += len(chunk)
                    if max_bytes is not None and total_size > max_bytes:
                        raise IngestionError(
                            f"PDF exceeds maximum supported size of {max_bytes} bytes"
                        )
                    chunks.append(chunk)
                return FetchedContent(
                    content=b"".join(chunks),
                    final_url=url,
                    content_type=content_type_header,
                )
        except httpx.HTTPError as exc:
            raise IngestionError(f"request failed: {exc}") from exc

    raise IngestionError("too many redirects")


def fetch_html(source: Source, client: httpx.Client) -> FetchedPage:
    fetched = fetch_content(
        source,
        client,
        accepted_content_types={"text/html", "application/xhtml+xml"},
        accept="text/html, application/xhtml+xml",
    )
    response = httpx.Response(
        status_code=200,
        content=fetched.content,
        headers={"content-type": fetched.content_type},
    )
    return FetchedPage(html=response.text, final_url=fetched.final_url)


def fetch_pdf(source: Source, client: httpx.Client) -> FetchedPdf:
    fetched = fetch_content(
        source,
        client,
        accepted_content_types={"application/pdf"},
        accept="application/pdf",
        max_bytes=MAX_PDF_BYTES,
    )
    if not fetched.content.startswith(b"%PDF-"):
        raise IngestionError("PDF response has an invalid %PDF- signature")
    return FetchedPdf(content=fetched.content, final_url=fetched.final_url)
