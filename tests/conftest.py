"""Shared pytest fixtures and configuration."""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Ensure environment variables required by pydantic-settings are present
# before any app module is imported during test collection.
# ---------------------------------------------------------------------------

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://rag:rag@localhost:5432/rag")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("EMBEDDING_PROVIDER", "hash")
