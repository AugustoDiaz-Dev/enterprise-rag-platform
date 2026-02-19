from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.openai_provider import OpenAILLM
from app.rag.embedding import EmbeddingProvider
from app.rag.retriever import Retriever
from app.schemas import QueryResponse, RetrievedChunkOut

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a knowledgeable assistant. Your job is to answer the user's question \
using ONLY the context passages provided below. Follow these rules:

1. Base your answer exclusively on the provided context — do not add outside knowledge.
2. If the context does not contain enough information to answer, say so clearly.
3. Be concise, accurate, and well-structured.
4. Respond in the same language as the user's question.
"""


class QueryService:
    def __init__(self, session: AsyncSession, embedder: EmbeddingProvider):
        self._retriever = Retriever(session, embedder)
        self._llm = OpenAILLM()

    async def query(
        self,
        *,
        query: str,
        top_k: int,
        score_threshold: float | None = None,
    ) -> QueryResponse:
        # 1. Retrieve semantically relevant chunks (with optional threshold override)
        result = await self._retriever.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        # 2. Synthesize answer with GPT using the retrieved chunks as context
        if not result.chunks:
            answer = "No relevant information found in the knowledge base."
        else:
            context_blocks = "\n\n---\n\n".join(
                f"[Passage {i + 1}]\n{c.text}"
                for i, c in enumerate(result.chunks)
            )
            user_message = f"Context passages:\n\n{context_blocks}\n\nQuestion: {query}"

            llm_response = await self._llm.complete(
                system=_SYSTEM_PROMPT,
                user=user_message,
            )
            answer = llm_response.content
            logger.info(
                "query_answered",
                extra={
                    "chunks_used": len(result.chunks),
                    "threshold_applied": result.threshold_applied,
                    "total_tokens": llm_response.total_tokens,
                },
            )

        return QueryResponse(
            query=result.query,
            answer=answer,
            chunks_retrieved=len(result.chunks),
            threshold_applied=result.threshold_applied,
            chunks=[
                RetrievedChunkOut(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    chunk_index=c.chunk_index,
                    distance=c.distance,
                    text=c.text,
                )
                for c in result.chunks
            ],
        )
