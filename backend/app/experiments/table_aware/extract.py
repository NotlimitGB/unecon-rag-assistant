"""Historical import path for the production-neutral table extractor."""

from app.retrieval.table_extract import (  # noqa: F401
    EXPERIMENT_ID,
    MAX_CONTEXT_CHARS,
    STRATEGY,
    ExperimentError,
    extract_pdf_tables,
    extract_table,
    normalize,
    row_id,
    validate_artifact,
)
