"""#17 — Schemas unit tests (Pydantic validation)."""
from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from app.schemas import (
    DocumentIngestResponse,
    PromptCreate,
    QueryRequest,
    ServiceMetricsOut,
)


# ---------------------------------------------------------------------------
# QueryRequest
# ---------------------------------------------------------------------------

def test_query_request_defaults() -> None:
    req = QueryRequest(query="hello")
    assert req.top_k == 5
    assert req.score_threshold is None
    assert req.document_id is None
    assert req.debug is False


def test_query_request_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(query="")


def test_query_request_rejects_top_k_zero() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(query="hello", top_k=0)


def test_query_request_rejects_top_k_over_limit() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(query="hello", top_k=51)


def test_query_request_accepts_document_id() -> None:
    doc_id = uuid.uuid4()
    req = QueryRequest(query="hello", document_id=doc_id)
    assert req.document_id == doc_id


# ---------------------------------------------------------------------------
# DocumentIngestResponse
# ---------------------------------------------------------------------------

def test_document_ingest_response_defaults() -> None:
    resp = DocumentIngestResponse(
        document_id=uuid.uuid4(),
        chunks_ingested=10,
    )
    assert resp.already_existed is False
    assert resp.ocr_used is False


def test_document_ingest_response_already_existed() -> None:
    resp = DocumentIngestResponse(
        document_id=uuid.uuid4(),
        chunks_ingested=0,
        already_existed=True,
    )
    assert resp.already_existed is True
    assert resp.chunks_ingested == 0


# ---------------------------------------------------------------------------
# PromptCreate
# ---------------------------------------------------------------------------

def test_prompt_create_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        PromptCreate(name="", content="some prompt")


def test_prompt_create_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        PromptCreate(name="default", content="")


def test_prompt_create_optional_author() -> None:
    p = PromptCreate(name="default", content="You are an assistant.")
    assert p.author is None


# ---------------------------------------------------------------------------
# ServiceMetricsOut
# ---------------------------------------------------------------------------

def test_service_metrics_out_nullable_fields() -> None:
    m = ServiceMetricsOut(
        total_queries=0,
        total_documents=0,
        total_chunks=0,
        avg_latency_ms=None,
        total_tokens=0,
        avg_tokens_per_query=None,
        total_estimated_cost_usd=None,
    )
    assert m.total_queries == 0
    assert m.avg_latency_ms is None
