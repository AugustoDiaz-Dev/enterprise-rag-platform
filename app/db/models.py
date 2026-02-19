from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(String(128), default="application/pdf")
    # SHA-256 hex digest of the raw PDF bytes — used for idempotent ingestion (#1)
    file_hash: Mapped[str | None] = mapped_column(String(64), unique=True, index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column()
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship(back_populates="chunks")


# ── #7 Token usage tracking ────────────────────────────────────────────────
class QueryLog(Base):
    """Persists per-query telemetry: tokens, cost estimate, latency."""

    __tablename__ = "query_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_text: Mapped[str] = mapped_column(Text)
    chunks_used: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    # GPT-4o-mini pricing as of 2024: $0.15/1M input + $0.60/1M output
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ── #8 Prompt versioning registry ─────────────────────────────────────────
class SystemPrompt(Base):
    """Versioned system prompts stored in the DB.

    One prompt per name can be active at a time; the active one is used
    for all queries unless a specific version is requested.
    """

    __tablename__ = "system_prompts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), index=True)          # e.g. "default"
    version: Mapped[int] = mapped_column(Integer)                       # 1, 2, 3 …
    content: Mapped[str] = mapped_column(Text)                          # full system prompt text
    author: Mapped[str | None] = mapped_column(String(256), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)     # only one active per name
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
