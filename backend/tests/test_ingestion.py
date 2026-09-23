import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from app.ingestion.cli import main
from app.ingestion.fetcher import IngestionError, fetch_html
from app.ingestion.html import ExtractedDocument, extract_html
from app.ingestion.manifest import ManifestError, load_manifest
from app.ingestion.models import Source
from app.ingestion.writer import build_document, write_document

SOURCE = {
    "id": "admissions-bachelor",
    "title": "Бакалавриат и специалитет",
    "url": "https://unecon.ru/priem/bachelor/",
    "source_type": "html",
    "category": "admissions_overview",
    "admission_year": 2026,
    "active": True,
}
HTML = """<html><head><title>Заголовок сайта</title><script>hidden()</script></head>
<body><header>Шапка сайта</header><nav>Меню сайта</nav>
<main><h1>Приём в вуз</h1><div class="slide-menu">Боковое меню</div>
<p>Важная   информация\n для абитуриентов.</p>
<ul><li>Первый пункт</li></ul><table><tr><th>Предмет</th><td>Русский язык</td></tr></table>
</main><footer>Подвал сайта</footer></body></html>"""


def manifest_file(path: Path, sources: list[dict] | None = None) -> Path:
    path.write_text(
        json.dumps({"schema_version": 1, "sources": sources or [SOURCE]}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_valid_manifest_loads(tmp_path: Path) -> None:
    manifest = load_manifest(manifest_file(tmp_path / "manifest.json"))
    assert manifest.schema_version == 1
    assert manifest.sources[0].id == SOURCE["id"]


@pytest.mark.parametrize(
    ("change", "sources"),
    [
        ({}, [SOURCE, SOURCE]),
        ({"url": "http://unecon.ru/priem/bachelor/"}, None),
        ({"url": "https://example.org/page/"}, None),
        ({"source_type": "pdf"}, None),
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


def test_document_contract_utf8_hash_and_atomic_replace(tmp_path: Path) -> None:
    source = Source.model_validate(SOURCE)
    extracted = ExtractedDocument(title="Приём в вуз", text="Русский текст\n\nВторой абзац")
    document = build_document(source, "https://unecon.ru/new-page/", extracted)
    output = write_document(tmp_path, source.id, document)
    saved = json.loads(output.read_text(encoding="utf-8"))
    expected_source = {key: value for key, value in SOURCE.items() if key != "active"}
    assert saved["source"] == {**expected_source, "final_url": "https://unecon.ru/new-page/"}
    assert "active" not in saved["source"]
    assert saved["document"]["text"] == extracted.text
    assert saved["document"]["content_sha256"] == hashlib.sha256(
        extracted.text.encode("utf-8")
    ).hexdigest()
    assert "Русский текст" in output.read_text(encoding="utf-8")
    assert list(tmp_path.glob("*.tmp")) == []

    with pytest.raises(IngestionError, match="no meaningful text"):
        build_document(source, source.url, ExtractedDocument(title="", text=" "))
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
        ["fetch", "--source-id", SOURCE["id"], "--manifest", str(path),
         "--output-dir", str(output_dir)],
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

    assert main(
        ["fetch", "--source-id", SOURCE["id"], "--manifest", str(path)],
        transport=httpx.MockTransport(unexpected_request),
    ) == 1


def test_cli_skips_inactive_sources(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = manifest_file(tmp_path / "manifest.json", [{**SOURCE, "active": False}])

    def unexpected_request(_: httpx.Request) -> httpx.Response:
        raise AssertionError("inactive source must not be fetched")

    assert main(["fetch", "--manifest", str(path)],
                transport=httpx.MockTransport(unexpected_request)) == 0
    assert "processed=0 failed=0 skipped=1" in capsys.readouterr().out
