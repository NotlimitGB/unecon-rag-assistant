from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
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

    @field_validator("embedding_model")
    @classmethod
    def nonempty_embedding_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("embedding model must not be blank")
        return value.strip()


settings = Settings()
