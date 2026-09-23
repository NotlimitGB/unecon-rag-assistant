"""Errors shared by the ingestion pipeline without importing network code."""


class IngestionError(ValueError):
    """A source could not be safely fetched or normalized."""
