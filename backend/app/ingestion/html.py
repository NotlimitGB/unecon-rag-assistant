"""Extract readable content from a university HTML page."""

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

BLOCK_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr")
NOISE_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside")


@dataclass(frozen=True)
class ExtractedDocument:
    title: str
    text: str


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def extract_html(html: str, fallback_title: str) -> ExtractedDocument:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup.find_all(NOISE_TAGS):
        element.decompose()
    # UNECON places a repeated sidebar menu inside <main>, outside semantic <nav>.
    for element in soup.select(".slide-menu"):
        element.decompose()

    content = soup.find("main") or soup.find("article") or soup.body or soup
    assert isinstance(content, Tag)
    heading = content.find("h1")
    title_tag = soup.find("title")
    title = normalize_space(heading.get_text(" ", strip=True)) if heading else ""
    if not title and title_tag:
        title = normalize_space(title_tag.get_text(" ", strip=True))
    title = title or fallback_title

    blocks: list[str] = []
    for element in content.find_all(BLOCK_TAGS):
        if element.find_parent(BLOCK_TAGS) is not None:
            continue
        if element.name == "tr":
            cells = element.find_all(("th", "td"), recursive=False)
            value = " | ".join(
                cell_text
                for cell in cells
                if (cell_text := normalize_space(cell.get_text(" ", strip=True)))
            )
        else:
            value = normalize_space(element.get_text(" ", strip=True))
        if value:
            blocks.append(value)

    if not blocks:
        fallback = normalize_space(content.get_text(" ", strip=True))
        if fallback:
            blocks.append(fallback)
    return ExtractedDocument(title=title, text="\n\n".join(blocks))
