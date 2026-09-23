"""Load the complete manifest before any network access."""

import json
from pathlib import Path

from pydantic import ValidationError

from app.ingestion.models import Manifest


class ManifestError(ValueError):
    """The curated source manifest is missing or invalid."""


def load_manifest(path: Path) -> Manifest:
    try:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        return Manifest.model_validate(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ManifestError(f"invalid source manifest {path}: {exc}") from exc
