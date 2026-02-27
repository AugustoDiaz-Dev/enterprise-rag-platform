from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Ingestion ─────────────────────────────────────────────────────────────

class DocumentIngestResponse(BaseModel):
    document_id: uuid.UUID
    chunks_ingested: int
    already_existed: bool = False       # True when file was already ingested (#1)
    ocr_used: bool = False              # True when OCR fallback was triggered (#6)


class DocumentListItem(BaseModel):
    """Single entry in GET /documents response."""
    id: uuid.UUID
    filename: str
    content_type: str
    created_at: datetime
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]
    total: int


# ── Query ─────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=50)
    # #3 Per-request threshold override. None means use server default.
    score_threshold: float | None = Field(default=None, ge=0.0, le=2.0)
    # #9 Metadata filter — restrict search to a single document
    document_id: uuid.UUID | None = None
    # #10 Retrieval debug mode — returns extra scoring info in response
    debug: bool = False


class RetrievedChunkOut(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    distance: float
    text: str


class Citation(BaseModel):
    label: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    text: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    chunks_retrieved: int
    threshold_applied: float | None
    chunks: list[RetrievedChunkOut]
    # #7 Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    latency_ms: int = 0
    # #10 Debug info (only populated when debug=True)
    debug_info: dict[str, Any] | None = None
    citations: list[Citation] = Field(default_factory=list)


# ── #7 Query logs ─────────────────────────────────────────────────────────

class QueryLogOut(BaseModel):
    id: uuid.UUID
    query_text: str
    chunks_used: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None
    latency_ms: int
    created_at: datetime


# ── #8 Prompt versioning ──────────────────────────────────────────────────

class PromptCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1)
    author: str | None = None


class PromptOut(BaseModel):
    id: uuid.UUID
    name: str
    version: int
    content: str
    author: str | None
    is_active: bool
    created_at: datetime


class PromptListResponse(BaseModel):
    prompts: list[PromptOut]
    total: int


# ── #14 Service metrics ───────────────────────────────────────────────────

class ServiceMetricsOut(BaseModel):
    """Aggregated service-level metrics."""
    total_queries: int
    total_documents: int
    total_chunks: int
    avg_latency_ms: float | None
    total_tokens: int
    avg_tokens_per_query: float | None
    total_estimated_cost_usd: float | None
