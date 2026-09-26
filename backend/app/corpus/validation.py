"""Explicit full candidate validation; historical labels remain diagnostic only."""

from app.config import settings
from app.corpus.checks import check_corpus
from app.corpus.incoming import recheck, transition
from app.corpus.models import Release, Validation
from app.corpus.operations import model_settings, seal
from app.corpus.paths import INCOMING, PROCESSED, ROOT, CorpusError, CorpusPaths, safe_id, safe_path
from app.corpus.storage import (
    atomic_json,
    inventory,
    now,
    operator_lock,
    read_json,
    read_state,
    release_digest,
    sha,
    verify_release,
)
from app.evaluation.canonical import RETRIEVAL_DATASET_SHA
from app.evaluation.dataset import load_dataset, validate_page_labels
from app.evaluation.runner import evaluate_retrieval
from app.ingestion.manifest import load_manifest
from app.ingestion.models import active_sources
from app.retrieval.index import RetrievalSession, _default_factory
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.service import RetrievedChunk
from app.retrieval.table_session import TableAwareRerankedRetrievalSession


def compare(paths, dataset, config, embedder_factory=None, reranker_factory=None):
    """One model pair, two fully explicit sessions; no runtime pointer lookup."""
    embedder = (
        embedder_factory
        or _default_factory(
            config.embedding_model, config.embedding_device, config.embedding_batch_size
        )
    )()
    reranker = (
        reranker_factory
        or (
            lambda: CrossEncoderReranker(
                config.reranker_model,
                config.reranker_device,
                config.reranker_batch_size,
                config.reranker_max_length,
            )
        )
    )()
    reports = []
    for corpus in paths:
        dense = RetrievalSession(
            corpus.manifest,
            corpus.chunks,
            corpus.index,
            config.embedding_model,
            embedder_factory=lambda: embedder,
        )
        session = TableAwareRerankedRetrievalSession(
            dense,
            corpus.manifest,
            corpus.pdf,
            corpus.tables,
            config.embedding_model,
            config.reranker_model,
            config.reranker_device,
            config.reranker_batch_size,
            config.reranker_max_length,
            config.reranker_candidate_k,
            reranker_factory=lambda: reranker,
        )

        class CheckedSession:
            metadata = dense.metadata
            underlying = session

            def search(self, query, top_k=5):
                hits = self.underlying.search(query, top_k)
                if len(hits) != 5:
                    raise CorpusError("candidate validation requires five results per question")
                for rank, hit in enumerate(hits, 1):
                    RetrievedChunk(
                        rank=rank,
                        chunk_id=hit["chunk_id"],
                        text=hit["text"],
                        source_id=hit["source_id"],
                        source_title=hit["source_title"],
                        source_url=hit["final_url"],
                        source_type=hit["source_type"],
                        category=hit["category"],
                        admission_year=hit["admission_year"],
                        page=hit["page_start"],
                        score=hit["score"],
                        dense_score=hit["dense_score"],
                        rerank_score=hit["rerank_score"],
                    )
                return hits

        reports.append(evaluate_retrieval(dataset, CheckedSession()))
    changed = []
    for prior, current in zip(reports[0]["questions"], reports[1]["questions"], strict=True):
        fields = ("chunk_id", "source_id", "page")
        if [[r[k] for k in fields] for r in prior["top_results"]] != [
            [r[k] for k in fields] for r in current["top_results"]
        ]:
            changed.append(current["question_id"])
    return {
        "base": reports[0],
        "candidate": reports[1],
        "changed_question_ids": changed,
        "affected_categories": sorted(
            {q["category"] for q in dataset["questions"] if q["question_id"] in changed}
        ),
    }


def validate(
    release_id,
    *,
    root=PROCESSED,
    incoming_root=INCOMING,
    config=settings,
    dataset_path=ROOT / "data/evaluation/retrieval_questions.json",
    reference_manifest=ROOT / "data/source_manifest.json",
    embedder_factory=None,
    reranker_factory=None,
):
    safe_id(release_id)
    with operator_lock(root):
        path = safe_path(root / "staging" / release_id, root)
        release = Release.model_validate(read_json(path / "release.json"))
        if release.release_id != release_id or release.base_release is None:
            raise CorpusError("validation requires a staged candidate with a base")
        # Invalidate the old token before doing any work, even when the attempt fails.
        release.sealed = False
        release.validation_sha256 = None
        atomic_json(path / "release.json", release.model_dump())
        checks, errors, comparison = {}, [], {}
        try:
            state = read_state(root)
            if (
                release.base_release != state.current_release
                or release.base_generation != state.generation
            ):
                raise CorpusError("candidate base is stale")
            base_path = root / "releases" / release.base_release
            base = verify_release(base_path)
            if release.models != model_settings(config) or base.models != model_settings(config):
                raise CorpusError("release model settings differ from validation configuration")
            base_paths = CorpusPaths(base_path / "registry.json", base_path)
            base_manifest, base_meta, base_rows = check_corpus(base_paths, config.embedding_model)
            packages = recheck(release.incoming, incoming_root)
            expected = transition(base_manifest, packages)
            candidate_paths = CorpusPaths(path / "registry.json", path)
            candidate_manifest, metadata, rows = check_corpus(
                candidate_paths, config.embedding_model
            )
            if expected != candidate_manifest:
                raise CorpusError("candidate registry does not match declared incoming transitions")
            for package in packages:
                copied = path / "incoming" / package.record.package_id / "metadata.json"
                if sha(copied.read_bytes()) != package.record.metadata_sha256:
                    raise CorpusError("copied incoming metadata mismatch")
                artifact = read_json(
                    path / package.source.source_type / f"{package.source.id}.json"
                )
                if artifact != package.artifact:
                    raise CorpusError("candidate document differs from incoming package")
            # Historical pairs are immutable even when their registry status changes.
            for entry in base.inventory:
                if entry.path.startswith(("html/", "pdf/", "originals/", "incoming/")):
                    if sha((path / entry.path).read_bytes()) != entry.sha256:
                        raise CorpusError("historical artifact changed")
            checks["corpus_and_transitions"] = True
            if sha(dataset_path.read_bytes()) != RETRIEVAL_DATASET_SHA:
                raise CorpusError("frozen retrieval dataset changed")
            dataset = load_dataset(dataset_path, load_manifest(reference_manifest))
            # The first published release is the preserved historical reference evidence.
            reference_path = root / "releases" / state.published_releases[0]
            verify_release(reference_path)
            _, reference_metadata, _ = check_corpus(
                CorpusPaths(reference_path / "registry.json", reference_path),
                config.embedding_model,
            )
            validate_page_labels(dataset, reference_metadata["records"])
            checks["historical_dataset"] = True
            before = inventory(path)
            comparison = compare(
                (base_paths, candidate_paths), dataset, config, embedder_factory, reranker_factory
            )
            recheck(release.incoming, incoming_root)
            if inventory(path) != before:
                raise CorpusError("candidate changed during validation")
            verify_release(base_path)
            checks["all_80_queries_with_five_results"] = True
            checks["incoming_unchanged"] = True
            comparison["count_deltas"] = {
                "chunks": len(metadata["records"]) - len(base_meta["records"]),
                "table_rows": len(rows) - len(base_rows),
            }
            comparison["changed_families"] = release.changed_families
            release.registry_sha256 = sha(candidate_paths.manifest.read_bytes())
            release.active_source_ids = [s.id for s in active_sources(candidate_manifest)]
            release.document_count = len(candidate_manifest.sources)
            release.chunk_count, release.table_row_count = len(metadata["records"]), len(rows)
            release.inventory = inventory(path)
            release.release_digest = release_digest(release)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        validation = Validation(
            schema_version=1,
            release_id=release_id,
            release_digest=release.release_digest,
            base_release=release.base_release,
            created_at=now(),
            kind="candidate",
            publishable=not errors,
            checks=checks,
            errors=errors,
            comparison=comparison,
        )
        seal(path, release, validation)
        return validation, release.validation_sha256
