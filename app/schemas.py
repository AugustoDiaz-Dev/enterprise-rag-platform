from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class DocumentIngestResponse(BaseModel):
    document_id: uuid.UUID
    chunks_ingested: int
    already_existed: bool = False  # True when the file was already ingested (#1)


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


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=50)
    # Per-request threshold override (#3). None means use server default.
    score_threshold: float | None = Field(default=None, ge=0.0, le=2.0)


class RetrievedChunkOut(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    distance: float
    text: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    chunks_retrieved: int
    threshold_applied: float | None
    chunks: list[RetrievedChunkOut]
