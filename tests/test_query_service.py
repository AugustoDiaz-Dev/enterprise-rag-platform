"""#17 — QueryService unit tests (fully mocked dependencies)."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.rag.vector_store import RetrievedChunk
from app.schemas import QueryResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(index: int = 0, distance: float = 0.15) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=index,
        text=f"Relevant passage {index} about enterprise RAG systems.",
        distance=distance,
    )


def _make_llm_response(content: str = "The answer is 42.") -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.prompt_tokens = 100
    resp.completion_tokens = 50
    resp.total_tokens = 150
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_returns_query_response() -> None:
    """Happy-path: service calls retriever + LLM and returns a QueryResponse."""
    session = AsyncMock()
    embedder = AsyncMock()
    chunks = [_make_chunk(0), _make_chunk(1)]
    llm_resp = _make_llm_response("Mocked answer.")

    with (
        patch("app.services.query_service.Retriever") as MockRetriever,
        patch("app.services.query_service.VectorStore") as MockStore,
        patch("app.services.query_service._make_llm") as mock_make_llm,
    ):
        retriever_inst = MockRetriever.return_value
        retrieval_result = MagicMock()
        retrieval_result.chunks = chunks
        retrieval_result.query = "What is RAG?"
        retrieval_result.threshold_applied = 0.4
        retrieval_result.query_embedding = [0.1] * 1536
        retrieval_result.document_id = None
        retriever_inst.retrieve = AsyncMock(return_value=retrieval_result)

        store_inst = MockStore.return_value
        store_inst.get_active_prompt = AsyncMock(return_value=None)
        store_inst.log_query = AsyncMock(return_value=uuid.uuid4())

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=llm_resp)
        mock_make_llm.return_value = llm

        from app.services.query_service import QueryService
        svc = QueryService(session, embedder)
        response = await svc.query(query="What is RAG?", top_k=5)

    assert isinstance(response, QueryResponse)
    assert response.answer == "Mocked answer."
    assert response.chunks_retrieved == 2
    assert response.prompt_tokens == 100
    assert response.completion_tokens == 50
    assert response.total_tokens == 150
    assert len(response.citations) == 2
    assert response.citations[0].label == "[Passage 1]"
    assert response.citations[0].chunk_id == chunks[0].chunk_id


@pytest.mark.asyncio
async def test_query_no_chunks_returns_fallback_message() -> None:
    """When retriever returns no chunks, a fallback answer is given."""
    session = AsyncMock()
    embedder = AsyncMock()

    with (
        patch("app.services.query_service.Retriever") as MockRetriever,
        patch("app.services.query_service.VectorStore") as MockStore,
        patch("app.services.query_service._make_llm"),
    ):
        retriever_inst = MockRetriever.return_value
        retrieval_result = MagicMock()
        retrieval_result.chunks = []
        retrieval_result.query = "unknown?"
        retrieval_result.threshold_applied = 0.4
        retrieval_result.query_embedding = []
        retrieval_result.document_id = None
        retriever_inst.retrieve = AsyncMock(return_value=retrieval_result)

        store_inst = MockStore.return_value
        store_inst.get_active_prompt = AsyncMock(return_value=None)
        store_inst.log_query = AsyncMock(return_value=uuid.uuid4())

        from app.services.query_service import QueryService
        svc = QueryService(session, embedder)
        response = await svc.query(query="unknown?", top_k=5)

    assert response.chunks_retrieved == 0
    assert "No relevant information" in response.answer
    assert response.citations == []


@pytest.mark.asyncio
async def test_query_debug_mode_populates_debug_info() -> None:
    """debug=True should populate debug_info in the response."""
    session = AsyncMock()
    embedder = AsyncMock()
    chunks = [_make_chunk(0)]
    llm_resp = _make_llm_response("Debug answer.")

    with (
        patch("app.services.query_service.Retriever") as MockRetriever,
        patch("app.services.query_service.VectorStore") as MockStore,
        patch("app.services.query_service._make_llm") as mock_make_llm,
    ):
        retriever_inst = MockRetriever.return_value
        retrieval_result = MagicMock()
        retrieval_result.chunks = chunks
        retrieval_result.query = "debug?"
        retrieval_result.threshold_applied = 0.4
        retrieval_result.query_embedding = [0.1] * 1536
        retrieval_result.document_id = None
        retriever_inst.retrieve = AsyncMock(return_value=retrieval_result)

        store_inst = MockStore.return_value
        store_inst.get_active_prompt = AsyncMock(return_value=None)
        store_inst.log_query = AsyncMock(return_value=uuid.uuid4())

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=llm_resp)
        mock_make_llm.return_value = llm

        from app.services.query_service import QueryService
        svc = QueryService(session, embedder)
    response = await svc.query(query="debug?", top_k=5, debug=True)

    assert response.debug_info is not None
    assert "embedding_dim" in response.debug_info
    assert "per_chunk_scores" in response.debug_info
    assert response.debug_info["embedding_dim"] == 1536
    assert len(response.citations) == 1
