"""Central, typed application settings.

All values can be overridden with environment variables (prefix ``EHR_``) or a
``.env`` file in the working directory. ``ANTHROPIC_API_KEY`` is read without a
prefix so the standard Anthropic variable works out of the box.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DateOrder = Literal["MDY", "DMY"]
SummaryProviderName = Literal["auto", "claude", "rule_based"]
EmbeddingProviderName = Literal["auto", "sentence_transformer", "hashing"]
VectorBackendName = Literal["auto", "faiss", "numpy"]


class Settings(BaseSettings):
    """Runtime configuration for every pipeline stage."""

    model_config = SettingsConfigDict(
        env_prefix="EHR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ---------------------------------------------------------------
    raw_data_dir: Path = Path("data/raw")
    var_dir: Path = Field(
        default=Path("var"), description="Root for all generated artifacts (DBs, index, reports)."
    )

    # --- Ingestion -----------------------------------------------------------
    date_order: DateOrder = Field(
        default="MDY", description="Interpretation of ambiguous numeric dates such as 03/04/2024."
    )
    mrn_digits: int = Field(default=8, ge=4, le=12, description="Canonical MRN width.")
    source_timezone: str = Field(
        default="UTC",
        description="IANA zone for naive source timestamps; FHIR instants need an offset.",
    )

    # --- FHIR ------------------------------------------------------------------
    mrn_system: str = Field(
        default="urn:oid:2.16.840.1.113883.3.0000.1",
        description="Identifier.system for MRNs (placeholder OID; set to your facility's).",
    )

    # --- Summarization -------------------------------------------------------
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "EHR_ANTHROPIC_API_KEY")
    )
    summary_provider: SummaryProviderName = "auto"
    claude_model: str = "claude-haiku-5-5"
    summary_max_words: int = Field(default=200, ge=50, le=500)
    summary_timeout_s: float = 30.0

    # --- Search --------------------------------------------------------------
    embedding_provider: EmbeddingProviderName = "auto"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hashing_dim: int = Field(default=512, ge=64, le=4096)
    search_top_k: int = Field(default=5, ge=1, le=50)
    vector_backend: VectorBackendName = Field(
        default="auto",
        description="auto = FAISS when installed, else NumPy (identical exact cosine search).",
    )

    # --- API -----------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8000", "http://127.0.0.1:8000"]
    )
    frontend_dir: Path = Path("frontend")

    # --- Derived paths -------------------------------------------------------
    @property
    def bundle_db_path(self) -> Path:
        return self.var_dir / "fhir_store.sqlite3"

    @property
    def summary_cache_path(self) -> Path:
        return self.var_dir / "summary_cache.sqlite3"

    @property
    def index_dir(self) -> Path:
        return self.var_dir / "index"

    @property
    def reports_dir(self) -> Path:
        return self.var_dir / "reports"

    def ensure_dirs(self) -> None:
        """Create every output directory the pipeline writes to."""
        for path in (self.var_dir, self.index_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (cleared in tests via ``get_settings.cache_clear()``)."""
    return Settings()
