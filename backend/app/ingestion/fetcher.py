"""Fetch a manifest source without leaving the official domain."""

from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

from app.ingestion.models import Source, validate_official_url

MAX_REDIRECTS = 5
HEADERS = {
    "User-Agent": "unecon-rag-assistant-thesis/0.1",
    "Accept": "text/html, application/xhtml+xml",
}


class IngestionError(ValueError):
    """A source could not be safely fetched or normalized."""


@dataclass(frozen=True)
class FetchedPage:
    html: str
    final_url: str


def fetch_html(source: Source, client: httpx.Client) -> FetchedPage:
    url = validate_official_url(source.url)
    for redirect_count in range(MAX_REDIRECTS + 1):
        try:
            response = client.get(url, headers=HEADERS, follow_redirects=False)
        except httpx.HTTPError as exc:
            raise IngestionError(f"request failed: {exc}") from exc

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

        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in {"text/html", "application/xhtml+xml"}:
            raise IngestionError(f"unsupported content type: {content_type or 'missing'}")
        return FetchedPage(html=response.text, final_url=url)

    raise IngestionError("too many redirects")
