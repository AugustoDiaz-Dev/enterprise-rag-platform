from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "dev"
    log_level: str = "INFO"

    # Security (#20)
    api_key: str = "rag-admin-secret"

    database_url: str

    embedding_provider: str = "hash"  # hash | openai

    openai_api_key: str | None = None

    # LLM settings
    llm_provider: str = "openai"          # openai | local  (#11)
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # #11 Local model provider (Ollama)
    local_model_url: str = "http://localhost:11434"
    local_model_name: str = "llama3"

    # Retrieval settings (#3 Score thresholding)
    # Cosine distance ranges 0–2; lower = more similar.
    # Chunks with distance > this value are filtered out as irrelevant.
    retrieval_score_threshold: float = 0.95


settings = Settings()
