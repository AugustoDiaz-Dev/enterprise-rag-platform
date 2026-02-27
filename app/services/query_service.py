from __future__ import annotations

import logging
import math
import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.llm.base import BaseLLM
from app.rag.embedding import EmbeddingProvider
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.schemas import Citation, QueryResponse, RetrievedChunkOut

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a knowledgeable assistant. Your job is to answer the user's question \
using ONLY the context passages provided below. Follow these rules:

1. Base your answer exclusively on the provided context — do not add outside knowledge.
2. If the context does not contain enough information to answer, say so clearly.
3. Be concise, accurate, and well-structured.
4. Respond in the same language as the user's question.
5. When citing evidence, reference passages using the exact labels shown (e.g., [Passage 1]). End your response with a "Sources: [Passage X]" sentence that lists the passages you relied on.
"""


def _format_citation_text(text: str, *, max_len: int = 400) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    truncated = cleaned[:max_len]
    if " " in truncated:
        truncated = truncated[: truncated.rfind(" ")]
    return truncated.rstrip() + "…"

# GPT-4o-mini pricing (per 1M tokens, as of 2024)
_COST_PER_1M_INPUT = 0.15
_COST_PER_1M_OUTPUT = 0.60


def _make_llm() -> BaseLLM:
    """Factory: returns the configured LLM provider (#11)."""
    if settings.llm_provider == "local":
        from app.llm.local_provider import LocalModelProvider
        return LocalModelProvider(
            model=settings.local_model_name,
            base_url=settings.local_model_url,
        )
    from app.llm.openai_provider import OpenAILLM
    return OpenAILLM()


class QueryService:
    def __init__(self, session: AsyncSession, embedder: EmbeddingProvider):
        self._session = session
        self._retriever = Retriever(session, embedder)
        self._store = VectorStore(session)
        self._llm: BaseLLM = _make_llm()

    async def query(
        self,
        *,
        query: str,
        top_k: int,
        score_threshold: float | None = None,
        document_id: uuid.UUID | None = None,   # #9
        debug: bool = False,                     # #10
        prompt_name: str = "default",            # #8
    ) -> QueryResponse:
        t0 = time.monotonic()

        # 1. Retrieve chunks ─────────────────────────────────────────────
        result = await self._retriever.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            document_id=document_id,
        )

        # 2. Resolve system prompt (#8) ───────────────────────────────────
        active_prompt = await self._store.get_active_prompt(prompt_name)
        system_prompt = active_prompt.content if active_prompt else _SYSTEM_PROMPT

        # 3. Call LLM ────────────────────────────────────────────────────
        prompt_tokens = completion_tokens = total_tokens = 0
        answer: str

        if not result.chunks:
            answer = "No relevant information found in the knowledge base."
        else:
            context_blocks = "\n\n---\n\n".join(
                f"[Passage {i + 1}]\n{c.text}" for i, c in enumerate(result.chunks)
            )
            user_message = f"Context passages:\n\n{context_blocks}\n\nQuestion: {query}"

            llm_response = await self._llm.complete(system=system_prompt, user=user_message)
            answer = llm_response.content
            prompt_tokens = llm_response.prompt_tokens
            completion_tokens = llm_response.completion_tokens
            total_tokens = llm_response.total_tokens

        latency_ms = int((time.monotonic() - t0) * 1000)

        # 4. Cost estimate — only meaningful for OpenAI (#7) ─────────────
        estimated_cost: float | None = None
        if settings.llm_provider == "openai" and total_tokens > 0:
            estimated_cost = round(
                (prompt_tokens / 1_000_000) * _COST_PER_1M_INPUT
                + (completion_tokens / 1_000_000) * _COST_PER_1M_OUTPUT,
                8,
            )

        # 5. Persist query log (#7) ───────────────────────────────────────
        await self._store.log_query(
            query_text=query,
            chunks_used=len(result.chunks),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
            latency_ms=latency_ms,
        )

        logger.info(
            "query_answered",
            extra={
                "chunks_used": len(result.chunks),
                "threshold_applied": result.threshold_applied,
                "total_tokens": total_tokens,
                "latency_ms": latency_ms,
            },
        )

        # 6. Debug payload (#10) ──────────────────────────────────────────
        debug_info = None
        if debug and result.query_embedding:
            emb = result.query_embedding
            norm = math.sqrt(sum(x * x for x in emb))
            debug_info = {
                "embedding_dim": len(emb),
                "embedding_norm": round(norm, 6),
                "threshold_applied": result.threshold_applied,
                "document_filter": str(result.document_id) if result.document_id else None,
                "chunks_before_threshold": top_k,
                "chunks_after_threshold": len(result.chunks),
                "per_chunk_scores": [
                    {
                        "chunk_id": str(c.chunk_id),
                        "chunk_index": c.chunk_index,
                        "distance": round(c.distance, 6),
                        "similarity": round(1 - c.distance, 6),
                    }
                    for c in result.chunks
                ],
                "prompt_source": (
                    f"db:{active_prompt.name}@v{active_prompt.version}"
                    if active_prompt else "hardcoded_default"
                ),
            }

        citations = [
            Citation(
                label=f"[Passage {i + 1}]",
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                text=_format_citation_text(c.text),
            )
            for i, c in enumerate(result.chunks)
        ]

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
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
            latency_ms=latency_ms,
            debug_info=debug_info,
            citations=citations,
        )
