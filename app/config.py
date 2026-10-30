"""Application configuration loaded from environment variables."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def configure_logging(level: str | None = None) -> None:
    """Configure application logging safely and idempotently."""
    settings = get_settings()
    log_level = (level or settings.log_level or "INFO").upper()
    root_logger = logging.getLogger()

    if root_logger.handlers:
        root_logger.handlers.clear()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


class Settings(BaseSettings):
    """Centralized application settings for the chatbot project."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    llm_provider: Literal["openai", "ollama"] = Field(
        default="openai",
        validation_alias=AliasChoices("LLM_PROVIDER", "llm_provider"),
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("LLM_MODEL", "llm_model"),
    )
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "llm_api_key"),
    )
    llm_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_BASE_URL", "llm_base_url"),
    )

    @field_validator("llm_base_url", mode="before")
    @classmethod
    def normalize_llm_base_url(cls, value: str | None) -> str | None:
        """Convert blank values to None for optional base URL configuration."""
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "embedding_model"),
    )
    chroma_path: str = Field(
        default="./chroma_db",
        validation_alias=AliasChoices("CHROMA_PATH", "chroma_path"),
    )
    sqlite_url: str = Field(
        default="sqlite:///./parking.db",
        validation_alias=AliasChoices("SQLITE_URL", "sqlite_url"),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        validation_alias=AliasChoices("TOP_K", "top_k"),
    )
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL", "log_level"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, lazily initialized application settings object."""
    return Settings()


__all__ = ["Settings", "get_settings", "configure_logging"]
