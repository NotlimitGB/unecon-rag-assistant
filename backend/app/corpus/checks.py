"""Offline complete corpus validation before models, sealing, and publication."""

from app.chunking.core import build_chunk_artifact
from app.corpus.paths import CorpusError, CorpusPaths, regular_file, safe_id
from app.corpus.storage import inventory, read_json
from app.ingestion.manifest import load_manifest
from app.ingestion.models import active_sources
from app.ingestion.pdf import extract_pdf
from app.ingestion.snapshots import read_pair
from app.ingestion.writer import build_pdf_document
from app.retrieval.table_extract import extract_pdf_tables
from app.retrieval.table_index import _production, _records, validated_table_index


def check_corpus(paths: CorpusPaths, model: str):
    # Check link boundaries before any source or index loader can follow a path.
    for directory in ("originals", "html", "pdf", "chunks", "index", "table_index"):
        if (paths.root / directory).exists():
            inventory(paths.root / directory)
    regular_file(paths.manifest, paths.manifest.parent)
    manifest = load_manifest(paths.manifest)
    active = active_sources(manifest)
    artifacts, snapshots = {}, {}
    for source in manifest.sources:
        safe_id(source.id)
        safe_id(source.logical_document_id)
        artifacts[source.id], snapshots[source.id] = read_pair(
            source, paths.root / source.source_type, paths.originals
        )
        if source.source_type == "pdf":
            rebuilt = build_pdf_document(
                source,
                artifacts[source.id]["source"]["final_url"],
                snapshots[source.id],
                extract_pdf(snapshots[source.id]),
            )
            if rebuilt != artifacts[source.id]:
                raise CorpusError("PDF snapshot does not reproduce normalized document")
    for kind in ("html", "pdf"):
        sources = [s for s in manifest.sources if s.source_type == kind]
        for directory, suffix in ((paths.root / kind, "json"), (paths.originals / kind, kind)):
            expected = {f"{s.id}.{suffix}" for s in sources}
            actual = {p.name for p in directory.iterdir()} if directory.exists() else set()
            if actual != expected:
                raise CorpusError("retained source files differ from registry")
    expected_chunks = {f"{source.id}.json" for source in active}
    if {p.name for p in paths.chunks.iterdir()} != expected_chunks:
        raise CorpusError("chunks must contain exactly the active sources")
    for source in active:
        if read_json(paths.chunks / f"{source.id}.json") != build_chunk_artifact(
            source, artifacts[source.id]
        ):
            raise CorpusError("chunks differ from normalized source")
    _, metadata = _production(paths.manifest, paths.chunks, paths.index, model)
    _, rows = validated_table_index(paths.manifest, paths.pdf, paths.tables, metadata, model)
    table_sources = [s for s in active if s.source_type == "pdf" and s.processing.table_aware]
    extracted = [
        extract_pdf_tables(s, artifacts[s.id]["source"]["final_url"], snapshots[s.id])
        for s in table_sources
    ]
    if rows != _records(extracted, table_sources, len(metadata["records"])):
        raise CorpusError("table index records differ from local PDF extraction")
    return manifest, metadata, rows
