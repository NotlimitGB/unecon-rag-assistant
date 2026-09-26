"""Offline checks for the isolated PDF table experiment."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import PROJECT_ROOT
from app.evaluation.dataset import load_dataset
from app.evaluation.generation_dataset import load_generation_dataset
from app.experiments.table_aware.evaluation import (
    EXPECTED_IDS,
    movement,
    page_rank,
    page_recall,
    select_questions,
)
from app.experiments.table_aware.extract import (
    ExperimentError,
    extract_table,
    normalize,
    validate_artifact,
)
from app.experiments.table_aware.retrieval import build_row_index, dense_search, save_row_index
from app.ingestion.manifest import load_manifest
from app.ingestion.models import Source


def _source() -> Source:
    return Source(
        id="admission-capacity-pdf",
        logical_document_id="admission-capacity-pdf",
        title="Тестовая таблица",
        url="https://unecon.ru/table.pdf",
        source_type="pdf",
        category="test",
        admission_year=2026,
        status="active",
        version=1,
        supersedes=None,
        published_at=None,
        effective_from=None,
        effective_to=None,
        processing={"table_aware": False},
    )


def _table() -> tuple[SimpleNamespace, SimpleNamespace]:
    group = "Стоимость обучения\nза семестр"
    raw = [
        ["Код", group, None],
        [None, "для граждан РФ", "для остальных"],
        ["38.03.01 Экономика", "179 500", "205 000"],
    ]
    header = SimpleNamespace(
        names=["Код", group, None],
        cells=[(0, 0, 100, 10), (100, 0, 300, 10), None],
        external=False,
    )
    rows = [
        SimpleNamespace(cells=[(0, 10, 100, 20), (100, 10, 200, 20), (200, 10, 300, 20)])
        for _ in raw
    ]
    table = SimpleNamespace(
        header=header,
        rows=rows,
        col_count=3,
        bbox=(0, 20, 300, 300),
        extract=lambda: raw,
    )
    page = SimpleNamespace(get_text=lambda *_args, **_kwargs: [(0, 0, 300, 15, "Название таблицы")])
    return page, table


def _artifact() -> dict:
    source = _source()
    page, table = _table()
    extracted = extract_table(source, "a" * 64, page, 2, table, 1)
    return {
        "schema_version": 1,
        "experiment_id": "table-aware-pdf-v1",
        "source": {
            "id": source.id,
            "title": source.title,
            "final_url": source.url,
            "admission_year": 2026,
            "file_sha256": "a" * 64,
        },
        "extraction": {"library": "PyMuPDF", "version": "test", "strategy": "lines_strict"},
        "tables": [extracted],
    }


def test_headers_context_and_deterministic_row() -> None:
    first, second = _artifact(), _artifact()
    assert first == second
    validate_artifact(first, _source(), "a" * 64, _source().url)
    table = first["tables"][0]
    assert len(table["rows"]) == 1  # Header continuation is not an indexed data row.
    assert table["context"] == "Название таблицы"
    assert table["columns"][1]["label"] != table["columns"][2]["label"]
    assert "за семестр" in table["rows"][0]["text"]
    assert table["rows"][0]["raw_cells"] == ["38.03.01 Экономика", "179 500", "205 000"]
    assert table["rows"][0]["row_id"].startswith("admission-capacity-pdf:p0002:t001:r0001:")


def test_blank_duplicate_headers_and_order() -> None:
    page, table = _table()
    table.header.names = ["Код", "Места", "Места"]
    table.header.cells = [(0, 0, 100, 10), (100, 0, 200, 10), (200, 0, 300, 10)]
    table.extract = lambda: [["Код", "Места", "Места"], ["X", "", "7"]]
    record = extract_table(_source(), "a" * 64, page, 1, table, 1)
    assert [col["label"] for col in record["columns"]] == ["Код", "column_2", "column_3"]
    assert [pair["value"] for pair in record["rows"][0]["values"]] == ["X", "", "7"]
    table.header.names = ["Код", "Места", ""]
    blank = extract_table(_source(), "a" * 64, page, 1, table, 1)
    assert blank["columns"][2]["label"] == "column_3"
    assert blank["columns"][2]["original_header"] == ""
    assert normalize("  два\n слова ") == "два слова"


def test_strict_artifact_integrity() -> None:
    original = _artifact()
    changes = [
        lambda x: x.update(unexpected=True),
        lambda x: x["source"].update(file_sha256="b" * 64),
        lambda x: x["tables"][0]["rows"][0].update(row_index=2),
        lambda x: x["tables"][0]["rows"][0].update(content_sha256="0" * 64),
        lambda x: x["tables"][0]["rows"][0]["values"][1].update(value="999"),
    ]
    for change in changes:
        damaged = copy.deepcopy(original)
        change(damaged)
        with pytest.raises(ExperimentError):
            validate_artifact(damaged, _source(), "a" * 64, _source().url)


def test_frozen_question_selection() -> None:
    manifest = load_manifest(PROJECT_ROOT / "data/source_manifest.json")
    retrieval = load_dataset(PROJECT_ROOT / "data/evaluation/retrieval_questions.json", manifest)
    generation = load_generation_dataset(
        PROJECT_ROOT / "data/evaluation/generation_questions.json", retrieval
    )
    assert (
        tuple(item["question_id"] for item in select_questions(generation, retrieval))
        == EXPECTED_IDS
    )
    modified = copy.deepcopy(generation)
    modified["questions"][6]["expected_status"] = "insufficient_evidence"
    with pytest.raises(ExperimentError):
        select_questions(modified, retrieval)


def test_page_metrics_and_movement() -> None:
    hits = [{"source_id": "x", "page_start": 2}, {"source_id": "x", "page_start": 1}]
    assert page_rank(hits, "x", [1]) == 2
    assert page_rank(hits, "y", [1]) is None
    assert page_recall([1, 3, None, 5]) == {
        "page_recall_at_1": 0.25,
        "page_recall_at_3": 0.5,
        "page_recall_at_5": 0.75,
    }
    assert movement({"a": None, "b": 1, "c": 3}, {"a": 4, "b": None, "c": 3}) == {
        "improved": ["a"],
        "worsened": ["b"],
        "unchanged": ["c"],
    }


class FakeEmbedder:
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.eye(len(texts), dtype=np.float32)

    def encode_query(self, _text: str) -> np.ndarray:
        return np.array([1.0], dtype=np.float32)


def test_row_index_and_metadata(tmp_path: Path) -> None:
    artifact = _artifact()
    index, records, raw = build_row_index([artifact], FakeEmbedder())
    assert index.ntotal == 1
    assert (
        dense_search(index, records, FakeEmbedder(), "вопрос")[0]["chunk_id"]
        == artifact["tables"][0]["rows"][0]["row_id"]
    )
    save_row_index(tmp_path, raw, records, "fake")
    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert meta["records"] == records
    assert (tmp_path / "index.faiss").read_bytes() == raw


def test_extractor_is_independent_of_probe_expectations() -> None:
    extractor = (PROJECT_ROOT / "backend/app/experiments/table_aware/extract.py").read_text(
        encoding="utf-8"
    )
    assert "gen-008" not in extractor
    assert "38.03.01" not in extractor
    assert "app.evaluation" not in extractor
