"""Filesystem boundary shared by operators and runtime, without model imports."""

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data/processed"
INCOMING = ROOT / "data/incoming"


class CorpusError(ValueError):
    pass


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
        raise CorpusError("ID must be safe lowercase kebab-case")
    if value in {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(10)),
        *(f"lpt{i}" for i in range(10)),
    }:
        raise CorpusError("reserved Windows ID")
    return value


def safe_path(path: Path, root: Path) -> Path:
    """Reject reparse points before resolving, including links in ancestor directories."""
    path, root = Path(os.path.abspath(path)), Path(os.path.abspath(root))
    if not path.is_relative_to(root):
        raise CorpusError("path escapes configured root")
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise CorpusError("symlink/junction/reparse points are forbidden")
    return path


def regular_file(path: Path, root: Path) -> Path:
    path = safe_path(path, root)
    if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise CorpusError("expected a regular file")
    return path


def protect_saved_release(path: Path) -> None:
    """Legacy writers may not mutate a release, including an explicitly supplied path."""
    path = Path(os.path.abspath(path))
    path = safe_path(path, Path(path.anchor))
    for parent in (path, *path.parents):
        if parent.name == "releases" or (parent / "release.json").is_file():
            raise CorpusError("release artifacts are immutable; use app.corpus staging")


@dataclass(frozen=True)
class CorpusPaths:
    manifest: Path
    root: Path

    @property
    def chunks(self):
        return self.root / "chunks"

    @property
    def index(self):
        return self.root / "index"

    @property
    def pdf(self):
        return self.root / "pdf"

    @property
    def tables(self):
        return self.root / "table_index"

    @property
    def originals(self):
        return self.root / "originals"


def legacy_paths() -> CorpusPaths:
    return CorpusPaths(ROOT / "data/source_manifest.json", PROCESSED)
