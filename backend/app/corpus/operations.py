"""Operator-only release preparation and publication, serialized under one lock."""

import os
import shutil
from pathlib import Path

from app.chunking.core import build_source_chunks
from app.config import settings
from app.corpus.checks import check_corpus
from app.corpus.incoming import read_package, recheck, transition
from app.corpus.models import ModelSettings, Release, State, Validation
from app.corpus.paths import (
    INCOMING,
    PROCESSED,
    CorpusError,
    CorpusPaths,
    legacy_paths,
    safe_id,
    safe_path,
)
from app.corpus.storage import (
    atomic_json,
    current_paths,
    inventory,
    now,
    operator_lock,
    read_state,
    release_digest,
    sha,
    verify_release,
)
from app.ingestion.models import active_sources
from app.ingestion.snapshots import write_pair
from app.retrieval.index import build_index
from app.retrieval.table_index import build_table_index


def model_settings(config=settings):
    return ModelSettings(
        embedding_model=config.embedding_model,
        reranker_model=config.reranker_model,
        reranker_max_length=config.reranker_max_length,
        reranker_candidate_k=config.reranker_candidate_k,
    )


def describe(path, release_id, manifest, dense, rows, *, base=None, packages=(), config=settings):
    release = Release(
        schema_version=1,
        release_id=release_id,
        base_release=base.current_release if base else None,
        base_generation=base.generation if base else None,
        created_at=now(),
        registry_sha256=sha((path / "registry.json").read_bytes()),
        incoming=[p.record for p in packages],
        changed_families=list(dict.fromkeys(p.source.logical_document_id for p in packages)),
        active_source_ids=[s.id for s in active_sources(manifest)],
        document_count=len(manifest.sources),
        chunk_count=len(dense["records"]),
        table_row_count=len(rows),
        models=model_settings(config),
        inventory=inventory(path),
        release_digest="0" * 64,
        sealed=False,
        validation_sha256=None,
    )
    release.release_digest = release_digest(release)
    atomic_json(path / "release.json", release.model_dump())
    return release


def seal(path, release, validation):
    # Report first, then metadata. Any interrupted pair fails closed on read.
    atomic_json(path / "validation.json", validation.model_dump())
    release.sealed = validation.publishable
    release.validation_sha256 = (
        sha((path / "validation.json").read_bytes()) if validation.publishable else None
    )
    atomic_json(path / "release.json", release.model_dump())
    return release


def _publish_pointer(root, target, previous, operation):
    history = list(previous.published_releases) if previous else []
    if target not in history:
        history.append(target)
    state = State(
        schema_version=1,
        generation=previous.generation + 1 if previous else 1,
        current_release=target,
        published_releases=history,
        updated_at=now(),
        operation=operation,
    )
    atomic_json(root / "release_state.json", state.model_dump())
    selected = current_paths(root)
    if selected.root.name != target or read_state(root) != state:
        raise CorpusError("published pointer readback mismatch")
    return state


def bootstrap(release_id, *, root=PROCESSED, legacy=None, config=settings):
    safe_id(release_id)
    legacy = legacy or legacy_paths()
    with operator_lock(root):
        if (root / "release_state.json").exists():
            raise CorpusError("bootstrap refused: publication state already exists")
        manifest, dense, rows = check_corpus(legacy, config.embedding_model)
        destination = safe_path(root / "releases" / release_id, root)
        destination.mkdir(parents=True, exist_ok=False)
        # Copy only validated corpus artifacts, not unrelated processed reports or caches.
        shutil.copyfile(legacy.manifest, destination / "registry.json")
        for name in ("originals", "html", "pdf", "chunks", "index", "table_index"):
            if (legacy.root / name).exists():
                # inventory rejects reparse points before copytree can follow them.
                inventory(legacy.root / name)
                shutil.copytree(legacy.root / name, destination / name)
        check_corpus(
            CorpusPaths(destination / "registry.json", destination), config.embedding_model
        )
        release = describe(destination, release_id, manifest, dense, rows, config=config)
        validation = Validation(
            schema_version=1,
            release_id=release_id,
            release_digest=release.release_digest,
            base_release=None,
            created_at=now(),
            kind="bootstrap",
            publishable=True,
            checks={"local_corpus": True, "copy_verified": True},
            errors=[],
            comparison={"note": "Structural bootstrap; no embedding or network."},
        )
        seal(destination, release, validation)
        verify_release(destination)
        _publish_pointer(root, release_id, None, "bootstrap")
        return release


def stage(
    release_id,
    package_paths,
    *,
    root=PROCESSED,
    incoming_root=INCOMING,
    config=settings,
    embedder_factory=None,
):
    safe_id(release_id)
    with operator_lock(root):
        state = read_state(root)
        base_paths = current_paths(root)
        base_manifest, _, _ = check_corpus(base_paths, config.embedding_model)
        packages = [read_package(Path(p), incoming_root) for p in package_paths]
        if not packages:
            raise CorpusError("at least one incoming package is required")
        candidate = transition(base_manifest, packages)
        if any((base_paths.root / "incoming" / p.record.package_id).exists() for p in packages):
            raise CorpusError("incoming package ID was already used by this lineage")
        path = safe_path(root / "staging" / release_id, root)
        if path.exists() or (root / "releases" / release_id).exists():
            raise CorpusError("release ID already exists")
        path.mkdir(parents=True)
        # Rebuild active outputs into empty directories; retain all historical source pairs.
        for name in ("originals", "html", "pdf", "incoming"):
            if (base_paths.root / name).exists():
                shutil.copytree(base_paths.root / name, path / name)
        atomic_json(path / "registry.json", candidate.model_dump())
        for package in packages:
            metadata_path = path / "incoming" / package.record.package_id / "metadata.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            if metadata_path.exists():
                raise CorpusError("incoming package ID was already used by this lineage")
            metadata_path.write_bytes(package.metadata)
            write_pair(
                package.source,
                path / package.source.source_type,
                path / "originals",
                package.artifact,
                package.content,
            )
        for source in active_sources(candidate):
            build_source_chunks(source, path, path / "chunks")
        # One embedder is shared by the two complete builds.
        from app.retrieval.index import _default_factory

        embedder = (
            embedder_factory
            or _default_factory(
                config.embedding_model, config.embedding_device, config.embedding_batch_size
            )
        )()
        dense = build_index(
            path / "registry.json",
            path / "chunks",
            path / "index",
            config.embedding_model,
            config.embedding_device,
            config.embedding_batch_size,
            embedder_factory=lambda: embedder,
        )
        table = build_table_index(
            path / "registry.json",
            path / "chunks",
            path / "index",
            path / "pdf",
            path / "table_index",
            config.embedding_model,
            config.embedding_device,
            config.embedding_batch_size,
            embedder_factory=lambda: embedder,
            originals_root=path / "originals",
        )
        recheck([p.record for p in packages], incoming_root)
        check_corpus(CorpusPaths(path / "registry.json", path), config.embedding_model)
        return describe(
            path,
            release_id,
            candidate,
            dense,
            table["records"],
            base=state,
            packages=packages,
            config=config,
        )


def publish(release_id, expected_current, validation_sha, confirm, *, root=PROCESSED):
    safe_id(release_id)
    with operator_lock(root):
        state = read_state(root)
        current_paths(root)
        if confirm != release_id or state.current_release != expected_current:
            raise CorpusError("confirmation or expected current release mismatch")
        staged = safe_path(root / "staging" / release_id, root)
        saved = safe_path(root / "releases" / release_id, root)
        if staged.exists() and saved.exists():
            raise CorpusError("ambiguous staged and saved release")
        path = staged if staged.exists() else saved
        release = verify_release(path)
        if (
            release.validation_sha256 != validation_sha
            or release.base_release != state.current_release
            or release.base_generation != state.generation
            or release_id in state.published_releases
        ):
            raise CorpusError("validation token or candidate base is stale")
        if path == staged:
            saved.parent.mkdir(parents=True, exist_ok=True)
            os.rename(staged, saved)
        verify_release(saved)
        return _publish_pointer(root, release_id, state, "publish")


def rollback(release_id, expected_current, confirm, *, root=PROCESSED):
    safe_id(release_id)
    with operator_lock(root):
        state = read_state(root)
        if (
            confirm != release_id
            or expected_current != state.current_release
            or release_id not in state.published_releases
            or release_id == state.current_release
        ):
            raise CorpusError("rollback confirmation, target history or current release mismatch")
        verify_release(safe_path(root / "releases" / release_id, root))
        return _publish_pointer(root, release_id, state, "rollback")
