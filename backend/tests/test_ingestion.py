import hashlib
import json
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import httpx
import pymupdf
import pytest

from app.ingestion.cli import main
from app.ingestion.fetcher import IngestionError, fetch_html, fetch_pdf
from app.ingestion.html import ExtractedDocument, extract_html
from app.ingestion.manifest import ManifestError, load_manifest
from app.ingestion.models import Source, source_identity
from app.ingestion.pdf import PAGE_SEPARATOR, extract_pdf
from app.ingestion.writer import build_document, build_pdf_document, write_document

SOURCE = {
    "id": "admissions-bachelor",
    "logical_document_id": "admissions-bachelor",
    "title": "Бакалавриат и специалитет",
    "url": "https://unecon.ru/priem/bachelor/",
    "source_type": "html",
    "category": "admissions_overview",
    "admission_year": 2026,
    "status": "active",
    "version": 1,
    "supersedes": None,
    "published_at": None,
    "effective_from": None,
    "effective_to": None,
    "processing": {"table_aware": False},
}
HTML = """<html><head><title>Заголовок сайта</title><script>hidden()</script></head>
<body><header>Шапка сайта</header><nav>Меню сайта</nav>
<main><h1>Приём в вуз</h1><div class="slide-menu">Боковое меню</div>
<p>Важная   информация\n для абитуриентов.</p>
<ul><li>Первый пункт</li></ul><table><tr><th>Предмет</th><td>Русский язык</td></tr></table>
</main><footer>Подвал сайта</footer></body></html>"""
PDF_SOURCE = {
    **SOURCE,
    "id": "admission-deadlines-pdf",
    "logical_document_id": "admission-deadlines-pdf",
    "title": "Сроки проведения приема в 2026 году",
    "url": "https://unecon.ru/example.pdf",
    "source_type": "pdf",
    "category": "admission_deadlines",
}


@lru_cache(maxsize=1)
def synthetic_pdf() -> bytes:
    document = pymupdf.open()
    font = pymupdf.Font("cjk")
    for text in (
        "Первая страница документа 2026",
        "Вторая страница с содержимым 2026",
        "",
    ):
        page = document.new_page()
        page.insert_font(fontname="testfont", fontbuffer=font.buffer)
        if text:
            page.insert_text((72, 72), text, fontname="testfont")
    content = document.tobytes()
    document.close()
    return content


@lru_cache(maxsize=1)
def empty_pdf() -> bytes:
    document = pymupdf.open()
    document.new_page()
    content = document.tobytes()
    document.close()
    return content


def zero_page_pdf() -> bytes:
    header = b"%PDF-1.4\n"
    catalog = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    pages = b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
    catalog_offset = len(header)
    pages_offset = catalog_offset + len(catalog)
    content = header + catalog + pages
    xref_offset = len(content)
    xref_entries = (
        b"0 3\n0000000000 65535 f \n"
        + f"{catalog_offset:010d} 00000 n \n{pages_offset:010d} 00000 n \n".encode()
    )
    return (
        content
        + b"xref\n"
        + xref_entries
        + b"trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n"
        + str(xref_offset).encode()
        + b"\n%%EOF\n"
    )


def manifest_file(path: Path, sources: list[dict] | None = None) -> Path:
    path.write_text(
        json.dumps({"schema_version": 2, "sources": sources or [SOURCE]}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_valid_manifest_loads(tmp_path: Path) -> None:
    manifest = load_manifest(manifest_file(tmp_path / "manifest.json"))
    assert manifest.schema_version == 2
    assert manifest.sources[0].id == SOURCE["id"]


def test_pdf_source_is_valid_in_manifest() -> None:
    source = Source.model_validate(PDF_SOURCE)
    assert source.source_type == "pdf"


def test_curated_manifest_contains_exact_approved_pdf_sources() -> None:
    project_root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(project_root / "data" / "source_manifest.json")
    pdf_sources = [source for source in manifest.sources if source.source_type == "pdf"]
    assert {source.id for source in pdf_sources} == {
        "admission-rules-pdf",
        "admission-capacity-pdf",
        "admission-deadlines-pdf",
        "entrance-exams-list-pdf",
        "entrance-exam-regulations-pdf",
        "tuition-order-128-pdf",
    }
    assert all(source.is_active and source.admission_year == 2026 for source in pdf_sources)
    assert all(source.url.endswith(".pdf") for source in pdf_sources)
    assert "tuition-order-177-pdf" not in {source.id for source in manifest.sources}


@pytest.mark.parametrize(
    ("change", "sources"),
    [
        ({}, [SOURCE, SOURCE]),
        ({"url": "http://unecon.ru/priem/bachelor/"}, None),
        ({"url": "https://example.org/page/"}, None),
        ({"source_type": "image"}, None),
        ({"unexpected": "value"}, None),
    ],
)
def test_invalid_manifest_is_rejected(
    tmp_path: Path, change: dict, sources: list[dict] | None
) -> None:
    entries = sources if sources is not None else [{**SOURCE, **change}]
    path = manifest_file(tmp_path / "manifest.json", entries)
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_html_extraction_removes_noise_and_preserves_blocks() -> None:
    extracted = extract_html(HTML, "Fallback")
    assert extracted.title == "Приём в вуз"
    assert "Важная информация для абитуриентов." in extracted.text
    assert "Первый пункт" in extracted.text
    assert "Предмет | Русский язык" in extracted.text
    assert "hidden" not in extracted.text
    assert "Меню" not in extracted.text
    assert "Боковое" not in extracted.text
    assert "Подвал" not in extracted.text
    assert "  " not in extracted.text


def test_fetch_success_and_safe_redirect() -> None:
    requested: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if len(requested) == 1:
            return httpx.Response(302, headers={"location": "/new-page/"})
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=HTML)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        page = fetch_html(Source.model_validate(SOURCE), client)
    assert requested == [SOURCE["url"], "https://unecon.ru/new-page/"]
    assert page.final_url == "https://unecon.ru/new-page/"
    assert "Приём" in page.html


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(500), "HTTP 500"),
        (httpx.Response(200, headers={"content-type": "application/pdf"}), "unsupported"),
    ],
)
def test_fetch_rejects_http_error_or_non_html(response: httpx.Response, message: str) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: response)) as client:
        with pytest.raises(IngestionError, match=message):
            fetch_html(Source.model_validate(SOURCE), client)


def test_external_redirect_is_rejected_before_external_request() -> None:
    requested: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://evil.example/secret"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(IngestionError, match="unsafe redirect"):
            fetch_html(Source.model_validate(SOURCE), client)
    assert requested == [SOURCE["url"]]


def test_external_pdf_redirect_is_rejected_before_external_request() -> None:
    requested: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://evil.example/file.pdf"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(IngestionError, match="unsafe redirect"):
            fetch_pdf(Source.model_validate(PDF_SOURCE), client)
    assert requested == [PDF_SOURCE["url"]]


def test_pdf_fetch_success_with_internal_redirect() -> None:
    requested: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if len(requested) == 1:
            return httpx.Response(301, headers={"location": "/official.pdf"})
        return httpx.Response(
            200,
            headers={"content-type": "application/pdf"},
            content=synthetic_pdf(),
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = fetch_pdf(Source.model_validate(PDF_SOURCE), client)
    assert requested == [PDF_SOURCE["url"], "https://unecon.ru/official.pdf"]
    assert result.final_url == "https://unecon.ru/official.pdf"
    assert result.content.startswith(b"%PDF-")


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(503), "HTTP 503"),
        (
            httpx.Response(200, headers={"content-type": "text/html"}, text="<html/>"),
            "unsupported content type",
        ),
        (
            httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"not a PDF"),
            "signature",
        ),
    ],
)
def test_pdf_fetch_rejects_http_type_and_signature(response: httpx.Response, message: str) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: response)) as client:
        with pytest.raises(IngestionError, match=message):
            fetch_pdf(Source.model_validate(PDF_SOURCE), client)


def test_pdf_fetch_rejects_declared_and_actual_oversize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.ingestion.fetcher.MAX_PDF_BYTES", 32)
    large_content = b"%PDF-" + b"x" * 40

    declared_response = httpx.Response(
        200,
        headers={"content-type": "application/pdf", "content-length": str(len(large_content))},
        content=large_content,
    )
    with httpx.Client(transport=httpx.MockTransport(lambda _: declared_response)) as client:
        with pytest.raises(IngestionError, match="maximum supported size"):
            fetch_pdf(Source.model_validate(PDF_SOURCE), client)

    actual_response = httpx.Response(
        200,
        headers={"content-type": "application/pdf"},
        stream=httpx.ByteStream(large_content),
    )
    assert "content-length" not in actual_response.headers
    with httpx.Client(transport=httpx.MockTransport(lambda _: actual_response)) as client:
        with pytest.raises(IngestionError, match="maximum supported size"):
            fetch_pdf(Source.model_validate(PDF_SOURCE), client)


def test_pdf_extracts_pages_deterministically_and_preserves_empty_page() -> None:
    first = extract_pdf(synthetic_pdf())
    second = extract_pdf(synthetic_pdf())
    assert len(first.pages) == 3
    assert [page.page_number for page in first.pages] == [1, 2, 3]
    assert first.pages[0].text == "Первая страница документа 2026"
    assert first.pages[1].text == "Вторая страница с содержимым 2026"
    assert first.pages[2].text == ""
    assert first.text == PAGE_SEPARATOR.join(page.text for page in first.pages if page.text)
    assert first.text == second.text


def test_pdf_document_hashes_and_metadata_are_deterministic() -> None:
    source = Source.model_validate(PDF_SOURCE)
    content = synthetic_pdf()
    extracted = extract_pdf(content)
    first = build_pdf_document(source, "https://unecon.ru/final.pdf", content, extracted)
    second = build_pdf_document(source, "https://unecon.ru/final.pdf", content, extracted)
    saved = first["document"]
    assert first == second
    assert first["source"]["url"] == PDF_SOURCE["url"]
    assert first["source"]["final_url"] == "https://unecon.ru/final.pdf"
    assert "active" not in first["source"]
    assert saved["title"] == PDF_SOURCE["title"]
    assert saved["page_count"] == 3
    assert saved["pages"] == [
        {"page_number": page.page_number, "text": page.text} for page in extracted.pages
    ]
    assert saved["file_sha256"] == hashlib.sha256(content).hexdigest()
    assert saved["content_sha256"] == hashlib.sha256(extracted.text.encode("utf-8")).hexdigest()


def test_pdf_rejects_malformed_encrypted_zero_page_and_no_text() -> None:
    with pytest.raises(IngestionError, match="could not open PDF"):
        extract_pdf(b"%PDF-not-a-real-document")

    encrypted = pymupdf.open()
    encrypted.new_page()
    encrypted_bytes = encrypted.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="user-secret",
    )
    encrypted.close()
    with pytest.raises(IngestionError, match="password-protected"):
        extract_pdf(encrypted_bytes)

    with pytest.raises(IngestionError, match="no pages"):
        extract_pdf(zero_page_pdf())

    with pytest.raises(IngestionError, match="OCR is required"):
        extract_pdf(empty_pdf())


def test_pdf_cli_dispatches_and_does_not_replace_output_on_extraction_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = manifest_file(tmp_path / "manifest.json", [PDF_SOURCE])
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing = output_dir / f"{PDF_SOURCE['id']}.json"

    responses = [synthetic_pdf(), empty_pdf()]

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept"] == "application/pdf"
        return httpx.Response(
            200,
            headers={"content-type": "application/pdf"},
            content=responses.pop(0),
        )

    transport = httpx.MockTransport(respond)
    first_result = main(
        [
            "fetch",
            "--source-id",
            PDF_SOURCE["id"],
            "--manifest",
            str(path),
            "--output-dir",
            str(output_dir),
            "--originals-root",
            str(tmp_path / "originals"),
        ],
        transport=transport,
    )
    assert first_result == 0
    first_json = json.loads(existing.read_text(encoding="utf-8"))
    assert first_json["document"]["page_count"] == 3
    assert "OK admission-deadlines-pdf" in capsys.readouterr().out

    second_result = main(
        [
            "fetch",
            "--source-id",
            PDF_SOURCE["id"],
            "--manifest",
            str(path),
            "--output-dir",
            str(output_dir),
            "--originals-root",
            str(tmp_path / "originals"),
        ],
        transport=transport,
    )
    assert second_result == 1
    assert json.loads(existing.read_text(encoding="utf-8")) == first_json
    assert "OCR is required" in capsys.readouterr().out


def test_document_contract_utf8_hash_and_atomic_replace(tmp_path: Path) -> None:
    source = Source.model_validate(SOURCE)
    extracted = ExtractedDocument(title="Приём в вуз", text="Русский текст\n\nВторой абзац")
    document = build_document(source, "https://unecon.ru/new-page/", extracted, HTML.encode())
    output = write_document(tmp_path, source.id, document)
    saved = json.loads(output.read_text(encoding="utf-8"))
    expected_source = source_identity(source)
    assert saved["source"] == {
        **expected_source,
        "final_url": "https://unecon.ru/new-page/",
        "snapshot_sha256": hashlib.sha256(HTML.encode()).hexdigest(),
    }
    assert "active" not in saved["source"]
    assert saved["document"]["text"] == extracted.text
    assert (
        saved["document"]["content_sha256"]
        == hashlib.sha256(extracted.text.encode("utf-8")).hexdigest()
    )
    assert "Русский текст" in output.read_text(encoding="utf-8")
    assert list(tmp_path.glob("*.tmp")) == []

    with pytest.raises(IngestionError, match="no meaningful text"):
        build_document(source, source.url, ExtractedDocument(title="", text=" "), b" ")
    assert json.loads(output.read_text(encoding="utf-8")) == saved

    with patch("app.ingestion.writer.os.replace", side_effect=OSError("disk failure")):
        with pytest.raises(OSError, match="disk failure"):
            write_document(tmp_path, source.id, {**document, "schema_version": 2})
    assert json.loads(output.read_text(encoding="utf-8")) == saved
    assert list(tmp_path.glob("*.tmp")) == []


def test_cli_unknown_id_has_no_network(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = manifest_file(tmp_path / "manifest.json")
    requests = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        raise AssertionError("network request was not expected")

    result = main(
        ["fetch", "--source-id", "unknown", "--manifest", str(path)],
        transport=httpx.MockTransport(respond),
    )
    assert result != 0
    assert requests == 0
    assert "FAILED unknown" in capsys.readouterr().out


def test_cli_one_source_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = manifest_file(tmp_path / "manifest.json")
    output_dir = tmp_path / "output"
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, headers={"content-type": "text/html"}, text=HTML)
    )
    result = main(
        [
            "fetch",
            "--source-id",
            SOURCE["id"],
            "--manifest",
            str(path),
            "--output-dir",
            str(output_dir),
            "--originals-root",
            str(tmp_path / "originals"),
        ],
        transport=transport,
    )
    assert result == 0
    assert (output_dir / "admissions-bachelor.json").exists()
    assert "processed=1 failed=0 skipped=0" in capsys.readouterr().out


def test_cli_validates_every_source_before_network(tmp_path: Path) -> None:
    invalid = {**SOURCE, "id": "invalid", "url": "https://example.org/"}
    path = manifest_file(tmp_path / "manifest.json", [SOURCE, invalid])

    def unexpected_request(_: httpx.Request) -> httpx.Response:
        raise AssertionError("manifest must be validated before any request")

    assert (
        main(
            ["fetch", "--source-id", SOURCE["id"], "--manifest", str(path)],
            transport=httpx.MockTransport(unexpected_request),
        )
        == 1
    )


def test_cli_skips_inactive_sources(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = manifest_file(tmp_path / "manifest.json", [{**SOURCE, "status": "draft"}])

    def unexpected_request(_: httpx.Request) -> httpx.Response:
        raise AssertionError("inactive source must not be fetched")

    assert (
        main(["fetch", "--manifest", str(path)], transport=httpx.MockTransport(unexpected_request))
        == 0
    )
    assert "processed=0 failed=0 skipped=1" in capsys.readouterr().out
