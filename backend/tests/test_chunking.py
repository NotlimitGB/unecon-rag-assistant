import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from app.chunking.cli import main
from app.chunking.core import (
    MAX_CHARS,
    TARGET_OVERLAP_CHARS,
    ChunkingError,
    build_chunk_artifact,
    build_source_chunks,
    validate_chunk_artifact,
    validate_normalized_document,
)
from app.ingestion.models import Source, source_identity

HTML_SOURCE = Source.model_validate(
    {
        "id": "admissions-faq",
        "logical_document_id": "admissions-faq",
        "title": "Вопросы и ответы",
        "url": "https://unecon.ru/priem/vopros-otvet/",
        "source_type": "html",
        "category": "faq",
        "admission_year": 2026,
        "status": "active",
        "version": 1,
        "supersedes": None,
        "published_at": None,
        "effective_from": None,
        "effective_to": None,
        "processing": {"table_aware": False},
    }
)
PDF_SOURCE = Source.model_validate(
    {
        "id": "admission-capacity-pdf",
        "logical_document_id": "admission-capacity-pdf",
        "title": "Количество мест для приема",
        "url": "https://unecon.ru/places.pdf",
        "source_type": "pdf",
        "category": "admission_capacity",
        "admission_year": 2026,
        "status": "active",
        "version": 1,
        "supersedes": None,
        "published_at": None,
        "effective_from": None,
        "effective_to": None,
        "processing": {"table_aware": False},
    }
)
INACTIVE_SOURCE = Source.model_validate(
    {
        **HTML_SOURCE.model_dump(),
        "id": "inactive-page",
        "logical_document_id": "inactive-page",
        "title": "Неактивный источник",
        "url": "https://unecon.ru/inactive/",
        "status": "draft",
    }
)
MISSING_SOURCE = Source.model_validate(
    {
        **HTML_SOURCE.model_dump(),
        "id": "missing-document",
        "logical_document_id": "missing-document",
        "title": "Отсутствующий документ",
        "url": "https://unecon.ru/missing/",
    }
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def html_document(text: str, source: Source = HTML_SOURCE) -> dict:
    return {
        "schema_version": 2,
        "source": {
            **source_identity(source),
            "final_url": source.url,
            "snapshot_sha256": hashlib.sha256(b"synthetic-pdf").hexdigest(),
        },
        "document": {
            "title": "Часто задаваемые вопросы",
            "text": text,
            "content_sha256": _hash(text),
        },
    }


def pdf_document(page_texts: list[str], source: Source = PDF_SOURCE) -> dict:
    text = "\n\n\f\n\n".join(page for page in page_texts if page)
    return {
        "schema_version": 2,
        "source": {
            **source_identity(source),
            "final_url": source.url,
            "snapshot_sha256": hashlib.sha256(b"synthetic-pdf").hexdigest(),
        },
        "document": {
            "title": source.title,
            "page_count": len(page_texts),
            "pages": [
                {"page_number": index, "text": page}
                for index, page in enumerate(page_texts, start=1)
            ],
            "text": text,
            "content_sha256": _hash(text),
            "file_sha256": hashlib.sha256(b"synthetic-pdf").hexdigest(),
        },
    }


def write_manifest(path: Path, sources: list[Source]) -> Path:
    path.write_text(
        json.dumps(
            {"schema_version": 2, "sources": [source.model_dump() for source in sources]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_short_html_is_one_exact_chunk_with_null_page_provenance() -> None:
    text = "Первый абзац.\n\nВторой абзац без изменений."
    result = build_chunk_artifact(HTML_SOURCE, html_document(text))

    assert len(result["chunks"]) == 1
    chunk = result["chunks"][0]
    assert chunk["text"] == text
    assert chunk["char_count"] == len(text)
    assert chunk["page_start"] is None
    assert chunk["page_end"] is None
    assert chunk["overlap_prefix_chars"] == 0
    assert "chunk_id" in chunk
    assert "id" not in chunk
    assert result["source"]["content_sha256"] == _hash(text)
    assert result["source"]["file_sha256"] is None
    assert "document" not in result
    assert set(result) == {"schema_version", "chunking", "source", "chunks"}
    assert set(result["chunking"]) == {
        "algorithm",
        "max_chunk_chars",
        "overlap_chars",
        "pdf_cross_page_chunks",
    }
    assert result["chunking"] == {
        "algorithm": "paragraph-aware-v1",
        "max_chunk_chars": 1200,
        "overlap_chars": 150,
        "pdf_cross_page_chunks": False,
    }
    assert set(result["source"]) == {
        "id",
        "logical_document_id",
        "version",
        "snapshot_sha256",
        "title",
        "url",
        "final_url",
        "source_type",
        "category",
        "admission_year",
        "content_sha256",
        "file_sha256",
    }


def test_html_paragraph_order_overlap_and_source_coverage() -> None:
    paragraphs = [f"Абзац {index}. " + ("Сведения для поступающих. " * 32) for index in range(6)]
    text = "\n\n".join(paragraphs)
    result = build_chunk_artifact(HTML_SOURCE, html_document(text))
    chunks = result["chunks"]

    assert len(chunks) > 2
    assert all(chunk["char_count"] <= MAX_CHARS for chunk in chunks)
    assert any(0 < chunk["overlap_prefix_chars"] <= TARGET_OVERLAP_CHARS for chunk in chunks[1:])
    rebuilt = "".join(chunk["text"][chunk["overlap_prefix_chars"] :] for chunk in chunks)
    assert rebuilt == text
    assert all(chunk["page_start"] is None for chunk in chunks)
    assert [chunk["ordinal"] for chunk in chunks] == list(range(1, len(chunks) + 1))


@pytest.mark.parametrize(
    ("text", "expected_boundary"),
    [
        ("Предложение завершено. " * 90, "sentence"),
        ("слово " * 500, "word"),
        ("x" * 3000, "character"),
    ],
)
def test_long_blocks_split_at_sentence_word_then_character_boundaries(
    text: str, expected_boundary: str
) -> None:
    chunks = build_chunk_artifact(HTML_SOURCE, html_document(text))["chunks"]
    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= MAX_CHARS for chunk in chunks)
    assert "".join(chunk["text"][chunk["overlap_prefix_chars"] :] for chunk in chunks) == text
    if expected_boundary == "sentence":
        assert chunks[0]["text"].rstrip().endswith(".")
    if expected_boundary == "word":
        assert chunks[0]["text"].endswith(" ")
    if expected_boundary == "character":
        assert len(chunks[0]["text"]) == MAX_CHARS


def test_pdf_chunks_are_isolated_by_page_and_keep_empty_pages() -> None:
    first = "Страница первая: набор программ и места 2026. " * 32
    third = "Страница третья: экономика и управление 140 мест."
    normalized = pdf_document([first, "", third])
    result = build_chunk_artifact(PDF_SOURCE, normalized)
    chunks = result["chunks"]

    assert [page["page_number"] for page in normalized["document"]["pages"]] == [1, 2, 3]
    assert all(chunk["page_start"] == chunk["page_end"] for chunk in chunks)
    assert {chunk["page_start"] for chunk in chunks} == {1, 3}
    assert result["source"]["file_sha256"] == normalized["document"]["file_sha256"]
    assert result["source"]["content_sha256"] == normalized["document"]["content_sha256"]
    assert "document" not in result
    for page_number, page_text in ((1, first), (2, ""), (3, third)):
        page_chunks = [chunk for chunk in chunks if chunk["page_start"] == page_number]
        assert (
            "".join(chunk["text"][chunk["overlap_prefix_chars"] :] for chunk in page_chunks)
            == page_text
        )


def test_normalized_document_integrity_metadata_and_pdf_page_join_are_checked() -> None:
    normalized = html_document("Целостный исходный текст")
    normalized["document"]["content_sha256"] = "0" * 64
    with pytest.raises(ChunkingError, match="content_sha256"):
        validate_normalized_document(HTML_SOURCE, normalized)

    normalized = html_document("Целостный исходный текст")
    normalized["source"]["category"] = "untrusted"
    with pytest.raises(ChunkingError, match="metadata mismatch"):
        validate_normalized_document(HTML_SOURCE, normalized)

    normalized_pdf = pdf_document(["Текст первой страницы", "Текст второй страницы"])
    normalized_pdf["document"]["text"] += "ошибочная вставка"
    normalized_pdf["document"]["content_sha256"] = _hash(normalized_pdf["document"]["text"])
    with pytest.raises(ChunkingError, match="does not match its non-empty pages"):
        validate_normalized_document(PDF_SOURCE, normalized_pdf)


def test_external_final_url_is_rejected() -> None:
    normalized = html_document("Проверка URL")
    normalized["source"]["final_url"] = "https://example.org/redirected"
    with pytest.raises(ChunkingError, match="approved boundary"):
        build_chunk_artifact(HTML_SOURCE, normalized)


def test_ids_and_hashes_are_deterministic_and_follow_source_content() -> None:
    first = build_chunk_artifact(HTML_SOURCE, html_document("Текст документа 2026"))
    repeat = build_chunk_artifact(HTML_SOURCE, html_document("Текст документа 2026"))
    changed = build_chunk_artifact(HTML_SOURCE, html_document("Текст документа 2027"))

    assert first == repeat
    assert first["chunks"][0]["chunk_id"] == "admissions-faq:0001:34acf49a11c6"
    assert first["chunks"][0]["chunk_id"] == repeat["chunks"][0]["chunk_id"]
    assert first["chunks"][0]["chunk_id"] != changed["chunks"][0]["chunk_id"]
    chunk = first["chunks"][0]
    assert chunk["content_sha256"] == _hash(chunk["text"])
    assert chunk["chunk_id"].startswith(f"{HTML_SOURCE.id}:0001:")
    assert "id" not in chunk
    assert "overlap_prefix_chars" in chunk
    assert "overlap_chars" not in chunk


def test_chunk_artifact_validator_rejects_obsolete_or_extra_public_fields() -> None:
    text = "Короткий документ для проверки контракта."
    normalized = html_document(text)
    artifact = build_chunk_artifact(HTML_SOURCE, normalized)
    chunks = [(None, text)]

    with_document = {**artifact, "document": {"content_sha256": _hash(text)}}
    with pytest.raises(ChunkingError, match="invalid chunk artifact schema"):
        validate_chunk_artifact(HTML_SOURCE, with_document, chunks)

    with_old_chunking_key = json.loads(json.dumps(artifact))
    with_old_chunking_key["chunking"]["max_chars"] = 1200
    with pytest.raises(ChunkingError, match="invalid chunking metadata schema"):
        validate_chunk_artifact(HTML_SOURCE, with_old_chunking_key, chunks)
    with_old_target_key = json.loads(json.dumps(artifact))
    with_old_target_key["chunking"]["target_overlap_chars"] = 150
    with pytest.raises(ChunkingError, match="invalid chunking metadata schema"):
        validate_chunk_artifact(HTML_SOURCE, with_old_target_key, chunks)

    with_old_chunk_key = json.loads(json.dumps(artifact))
    chunk = with_old_chunk_key["chunks"][0]
    chunk["id"] = chunk.pop("chunk_id")
    with pytest.raises(ChunkingError, match=r"invalid chunks\[0\] schema"):
        validate_chunk_artifact(HTML_SOURCE, with_old_chunk_key, chunks)


def test_failed_validation_and_atomic_write_leave_existing_artifact_unchanged(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    input_dir = input_root / "html"
    input_dir.mkdir(parents=True)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    input_file = input_dir / f"{HTML_SOURCE.id}.json"
    input_file.write_text(json.dumps(html_document("Новый корректный текст")), encoding="utf-8")
    destination = output_dir / f"{HTML_SOURCE.id}.json"
    old_content = '{"existing": true}\n'
    destination.write_text(old_content, encoding="utf-8")

    with patch("app.ingestion.writer.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            build_source_chunks(HTML_SOURCE, input_root, output_dir)
    assert destination.read_text(encoding="utf-8") == old_content
    assert list(output_dir.glob("*.tmp")) == []

    input_file.write_text(
        json.dumps(html_document("Сломанный текст"), ensure_ascii=False), encoding="utf-8"
    )
    broken = json.loads(input_file.read_text(encoding="utf-8"))
    broken["document"]["content_sha256"] = "0" * 64
    input_file.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ChunkingError, match="content_sha256"):
        build_source_chunks(HTML_SOURCE, input_root, output_dir)
    assert destination.read_text(encoding="utf-8") == old_content


def test_cli_builds_selected_source_and_reports_missing_inactive_or_unknown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = write_manifest(tmp_path / "manifest.json", [HTML_SOURCE])
    input_root = tmp_path / "input"
    input_dir = input_root / "html"
    input_dir.mkdir(parents=True)
    (input_dir / f"{HTML_SOURCE.id}.json").write_text(
        json.dumps(html_document("CLI читает только локальный JSON"), ensure_ascii=False),
        encoding="utf-8",
    )
    output_dir = tmp_path / "chunks"
    args = [
        "build",
        "--source-id",
        HTML_SOURCE.id,
        "--manifest",
        str(manifest),
        "--input-root",
        str(input_root),
        "--output-dir",
        str(output_dir),
    ]

    assert main(args) == 0
    assert (output_dir / f"{HTML_SOURCE.id}.json").exists()
    success_output = capsys.readouterr().out
    assert "OK admissions-faq chunks=1" in success_output
    assert "processed=1 failed=0 skipped=0 chunks=1" in success_output

    missing_args = [*args]
    missing_args[2] = "missing-source"
    assert main(missing_args) == 1
    unknown_output = capsys.readouterr().out
    assert "unknown or inactive" in unknown_output
    assert "processed=0 failed=1 skipped=0 chunks=0" in unknown_output

    inactive_manifest = write_manifest(tmp_path / "inactive.json", [INACTIVE_SOURCE])
    inactive_args = [*args]
    inactive_args[inactive_args.index(str(manifest))] = str(inactive_manifest)
    inactive_args[inactive_args.index("--source-id") + 1] = INACTIVE_SOURCE.id
    assert main(inactive_args) == 1
    inactive_output = capsys.readouterr().out
    assert "unknown or inactive" in inactive_output
    assert "processed=0 failed=1 skipped=0 chunks=0" in inactive_output

    assert main(["build", "--manifest", str(tmp_path / "missing-manifest.json")]) == 1
    manifest_error_output = capsys.readouterr().out
    assert "processed=0 failed=1 skipped=0 chunks=0" in manifest_error_output

    failed_args = [*args]
    failed_args[failed_args.index("--source-id") + 1] = MISSING_SOURCE.id
    failed_manifest = write_manifest(tmp_path / "failed-source.json", [MISSING_SOURCE])
    failed_args[failed_args.index(str(manifest))] = str(failed_manifest)
    failed_args[failed_args.index("--input-root") + 1] = str(tmp_path / "no-local-input")
    assert main(failed_args) == 1
    assert "processed=0 failed=1 skipped=0 chunks=0" in capsys.readouterr().out


def test_cli_continues_after_missing_local_artifact_and_skips_inactive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = write_manifest(
        tmp_path / "manifest.json", [HTML_SOURCE, MISSING_SOURCE, INACTIVE_SOURCE]
    )
    input_root = tmp_path / "input"
    input_dir = input_root / "html"
    input_dir.mkdir(parents=True)
    (input_dir / f"{HTML_SOURCE.id}.json").write_text(
        json.dumps(html_document("Один успешный локальный чанк"), ensure_ascii=False),
        encoding="utf-8",
    )
    result = main(
        [
            "build",
            "--manifest",
            str(manifest),
            "--input-root",
            str(input_root),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    output = capsys.readouterr().out

    assert result == 1
    assert "OK admissions-faq chunks=1" in output
    assert "FAILED missing-document" in output
    assert "SKIPPED inactive-page" in output
    assert "processed=1 failed=1 skipped=1 chunks=1" in output


def test_chunking_import_boundary_has_no_network_dependencies() -> None:
    package_dir = Path(__file__).resolve().parents[1] / "app" / "chunking"
    forbidden_modules = {"httpx", "app.ingestion.fetcher", "app.ingestion.cli"}
    for path in package_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = {node.module}
            else:
                continue
            assert imported.isdisjoint(forbidden_modules), f"network import in {path}"

    check = (
        "import sys; import app.chunking.cli; "
        "assert not {'httpx', 'app.ingestion.fetcher', 'app.ingestion.cli'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", check], check=True, cwd=package_dir.parents[1])
