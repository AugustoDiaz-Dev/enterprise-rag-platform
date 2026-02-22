"""#17 — Retriever unit tests (mocked async session + embedder)."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.rag.retriever import Retriever, RetrievalResult
from app.rag.vector_store import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(index: int = 0, distance: float = 0.2) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=index,
        text=f"Sample passage {index}.",
        distance=distance,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_returns_result() -> None:
    """Happy path: embedder + store are called, result is a RetrievalResult."""
    session = AsyncMock()
    embedder = AsyncMock()
    embedder.embed_query = AsyncMock(return_value=[0.1] * 1536)

    chunks = [_make_chunk(0, 0.1), _make_chunk(1, 0.25)]

    with patch("app.rag.retriever.VectorStore") as MockStore:
        store_instance = MockStore.return_value
        store_instance.query = AsyncMock(return_value=chunks)

        retriever = Retriever(session, embedder)
        result = await retriever.retrieve(query="what is rag?", top_k=5)

    assert isinstance(result, RetrievalResult)
    assert result.query == "what is rag?"
    assert len(result.chunks) == 2
    assert result.threshold_applied is not None


@pytest.mark.asyncio
async def test_retrieve_empty_when_no_chunks() -> None:
    """When the store returns nothing, result.chunks is empty."""
    session = AsyncMock()
    embedder = AsyncMock()
    embedder.embed_query = AsyncMock(return_value=[0.0] * 1536)

    with patch("app.rag.retriever.VectorStore") as MockStore:
        store_instance = MockStore.return_value
        store_instance.query = AsyncMock(return_value=[])

        retriever = Retriever(session, embedder)
        result = await retriever.retrieve(query="nothing here", top_k=5)

    assert result.chunks == []


@pytest.mark.asyncio
async def test_retrieve_passes_document_filter() -> None:
    """document_id is forwarded to the vector store query."""
    session = AsyncMock()
    embedder = AsyncMock()
    embedder.embed_query = AsyncMock(return_value=[0.1] * 1536)
    doc_id = uuid.uuid4()

    with patch("app.rag.retriever.VectorStore") as MockStore:
        store_instance = MockStore.return_value
        store_instance.query = AsyncMock(return_value=[])

        retriever = Retriever(session, embedder)
        result = await retriever.retrieve(query="q", top_k=3, document_id=doc_id)

    store_instance.query.assert_called_once()
    call_kwargs = store_instance.query.call_args.kwargs
    assert call_kwargs["document_id"] == doc_id
    assert result.document_id == doc_id


@pytest.mark.asyncio
async def test_retrieve_stores_query_embedding() -> None:
    """The query embedding is stored in the result for debug mode (#10)."""
    session = AsyncMock()
    embedder = AsyncMock()
    embedding = [0.5] * 1536
    embedder.embed_query = AsyncMock(return_value=embedding)

    with patch("app.rag.retriever.VectorStore") as MockStore:
        store_instance = MockStore.return_value
        store_instance.query = AsyncMock(return_value=[])

        retriever = Retriever(session, embedder)
        result = await retriever.retrieve(query="debug test", top_k=5)

    assert result.query_embedding == embedding
