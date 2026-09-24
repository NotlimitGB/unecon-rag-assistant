from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, StrictInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "UNEcon RAG Assistant"
    environment: Literal["development", "test", "production"] = Field(
        default="development",
        validation_alias="APP_ENV",
    )
    frontend_origin: str = "http://localhost:5173"
    embedding_model: str = Field(default="BAAI/bge-m3", validation_alias="EMBEDDING_MODEL")
    embedding_device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto", validation_alias="EMBEDDING_DEVICE"
    )
    embedding_batch_size: int = Field(
        default=16, ge=1, le=128, validation_alias="EMBEDDING_BATCH_SIZE"
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3", validation_alias="RERANKER_MODEL"
    )
    reranker_device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto", validation_alias="RERANKER_DEVICE"
    )
    reranker_batch_size: int = Field(default=8, ge=1, le=64, validation_alias="RERANKER_BATCH_SIZE")
    reranker_max_length: int = Field(
        default=512, ge=32, le=4096, validation_alias="RERANKER_MAX_LENGTH"
    )
    reranker_candidate_k: int = Field(
        default=20, ge=5, le=100, validation_alias="RERANKER_CANDIDATE_K"
    )
    retrieval_mode: Literal["dense", "reranked"] = Field(
        default="reranked", validation_alias="RETRIEVAL_MODE"
    )
    retrieval_top_k: StrictInt = Field(default=5, ge=1, le=20, validation_alias="RETRIEVAL_TOP_K")
    generation_provider: Literal["ollama"] = Field(
        default="ollama", validation_alias="GENERATION_PROVIDER"
    )
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434", validation_alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="qwen3.5:9b", validation_alias="OLLAMA_MODEL")
    ollama_timeout_seconds: int = Field(
        default=180, ge=1, le=600, validation_alias="OLLAMA_TIMEOUT_SECONDS"
    )
    generation_temperature: float = Field(
        default=0.1, ge=0, le=2, validation_alias="GENERATION_TEMPERATURE"
    )
    generation_max_tokens: int = Field(
        default=512, ge=1, le=4096, validation_alias="GENERATION_MAX_TOKENS"
    )

    @field_validator("ollama_base_url")
    @classmethod
    def valid_ollama_base_url(cls, value: str) -> str:
        try:
            parts = urlsplit(value)
            port = parts.port
        except ValueError as exc:
            raise ValueError("invalid Ollama base URL") from exc
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or (parts.path not in {"", "/"})
            or parts.query
            or parts.fragment
            or port == 0
        ):
            raise ValueError("Ollama base URL must be an HTTP(S) origin")
        return value.rstrip("/")

    @field_validator("ollama_model")
    @classmethod
    def nonempty_ollama_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Ollama model must not be blank")
        return value.strip()

    @field_validator("retrieval_top_k", mode="before")
    @classmethod
    def parse_retrieval_top_k(cls, value: object) -> int:
        if type(value) is int:
            return value
        if isinstance(value, str) and value.isascii() and value.isdecimal():
            return int(value)
        raise ValueError("retrieval top K must be an integer")

    @field_validator("embedding_model")
    @classmethod
    def nonempty_embedding_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("embedding model must not be blank")
        return value.strip()

    @field_validator("reranker_model")
    @classmethod
    def nonempty_reranker_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reranker model must not be blank")
        return value.strip()


settings = Settings()
