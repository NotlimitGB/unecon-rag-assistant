"""Hashing, strict reads, exclusive operator lock, and atomic pointer writes."""

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from app.corpus.models import InventoryEntry, Release, State, Validation
from app.corpus.paths import PROCESSED, CorpusError, CorpusPaths, regular_file, safe_id, safe_path


def now():
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha(data: bytes):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()


def read_json(path: Path):
    regular_file(path, path.parent)
    return parse_json(path.read_bytes())


def parse_json(content: bytes):

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CorpusError("duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise CorpusError(f"non-finite JSON number: {value}")

    return json.loads(content.decode("utf-8"), object_pairs_hook=unique, parse_constant=nonfinite)


def atomic_json(path: Path, value):
    safe_path(path, path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp") as handle:
            temporary = Path(handle.name)
            handle.write(canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def operator_lock(root: Path):
    safe_path(root, root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".corpus.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise CorpusError("corpus operation lock exists; automatic recovery is forbidden") from exc
    try:
        os.close(fd)
        yield
    finally:
        lock.unlink()


def inventory(root: Path):
    entries = []

    def walk(folder):
        for path in sorted(folder.iterdir()):
            safe_path(path, root)
            relative = path.relative_to(root).as_posix()
            if relative in {"release.json", "validation.json"}:
                continue
            if path.is_dir():
                walk(path)
            else:
                regular_file(path, root)
                if path.suffix == ".tmp" or path.name.endswith(".lock"):
                    raise CorpusError("unfinished temporary or lock file in release")
                entries.append(
                    InventoryEntry(
                        path=relative, size=path.stat().st_size, sha256=sha(path.read_bytes())
                    )
                )

    walk(root)
    return sorted(entries, key=lambda entry: entry.path)


def release_digest(release: Release):
    return sha(
        canonical(release.model_dump(exclude={"release_digest", "sealed", "validation_sha256"}))
    )


def verify_release(path: Path, *, require_sealed=True) -> Release:
    try:
        return _verify_release(path, require_sealed=require_sealed)
    except (KeyError, TypeError, AttributeError) as exc:
        raise CorpusError("malformed release validation data") from exc


def _verify_release(path: Path, *, require_sealed=True) -> Release:
    safe_path(path, path.parent)
    release = Release.model_validate(read_json(path / "release.json"))
    if release.release_id != path.name or release.release_digest != release_digest(release):
        raise CorpusError("release identity or digest mismatch")
    if release.inventory != inventory(path):
        raise CorpusError("release inventory mismatch")
    if sha((path / "registry.json").read_bytes()) != release.registry_sha256:
        raise CorpusError("registry hash mismatch")
    if require_sealed or release.sealed:
        if not release.sealed:
            raise CorpusError("release is not sealed")
        validation = Validation.model_validate(read_json(path / "validation.json"))
        if (
            not validation.publishable
            or validation.release_id != release.release_id
            or validation.release_digest != release.release_digest
            or validation.base_release != release.base_release
            or sha((path / "validation.json").read_bytes()) != release.validation_sha256
        ):
            raise CorpusError("validation is not bound to this sealed release")
    return release


def read_state(root: Path = PROCESSED):
    return State.model_validate(read_json(root / "release_state.json"))


def current_paths(root: Path = PROCESSED) -> CorpusPaths:
    state = read_state(root)
    path = safe_path(root / "releases" / safe_id(state.current_release), root)
    verify_release(path)
    return CorpusPaths(path / "registry.json", path)
