from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.rag.embedding import EmbeddingProvider
from app.rag.vector_store import RetrievedChunk, VectorStore


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    threshold_applied: float | None
    query_embedding: list[float] = field(default_factory=list)   # #10 debug
    document_id: uuid.UUID | None = None                          # #9 filter applied


class Retriever:
    def __init__(self, session: AsyncSession, embedder: EmbeddingProvider):
        self._store = VectorStore(session)
        self._embedder = embedder

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        document_id: uuid.UUID | None = None,   # #9 metadata filter
    ) -> RetrievalResult:
        threshold = (
            score_threshold if score_threshold is not None
            else settings.retrieval_score_threshold
        )
        q_emb = await self._embedder.embed_query(query)
        chunks = await self._store.query(
            query_embedding=q_emb,
            top_k=top_k,
            score_threshold=threshold,
            document_id=document_id,
        )
        return RetrievalResult(
            query=query,
            chunks=chunks,
            threshold_applied=threshold,
            query_embedding=q_emb,
            document_id=document_id,
        )
