from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Select, delete, func, select
from dataclasses import fields as dc_fields
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk as ChunkModel
from app.db.models import Document as DocumentModel
from app.db.models import QueryLog as QueryLogModel
from app.db.models import SystemPrompt as SystemPromptModel


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    text: str
    distance: float


@dataclass(frozen=True)
class DocumentRecord:
    id: uuid.UUID
    filename: str
    content_type: str
    created_at: datetime
    chunk_count: int


@dataclass(frozen=True)
class ServiceMetrics:
    """Aggregated service-level stats returned by GET /metrics (#14)."""
    total_queries: int
    total_documents: int
    total_chunks: int
    avg_latency_ms: float | None
    total_tokens: int
    avg_tokens_per_query: float | None
    total_estimated_cost_usd: float | None


class VectorStore:
    def __init__(self, session: AsyncSession):
        self._session = session

    # ------------------------------------------------------------------ #
    #  Ingestion                                                           #
    # ------------------------------------------------------------------ #

    async def find_document_by_hash(self, file_hash: str) -> uuid.UUID | None:
        """Return the document_id if a file with this SHA-256 was already ingested."""
        stmt = select(DocumentModel.id).where(DocumentModel.file_hash == file_hash)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def create_document(
        self, *, filename: str, content_type: str, file_hash: str | None = None
    ) -> uuid.UUID:
        doc = DocumentModel(filename=filename, content_type=content_type, file_hash=file_hash)
        self._session.add(doc)
        await self._session.flush()
        return doc.id

    async def add_chunks(
        self,
        *,
        document_id: uuid.UUID,
        chunks: list[tuple[int, str]],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for (chunk_index, text), embedding in zip(chunks, embeddings, strict=True):
            self._session.add(
                ChunkModel(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=text,
                    embedding=embedding,
                )
            )

    # ------------------------------------------------------------------ #
    #  Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def _similarity_query(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        document_id: uuid.UUID | None = None,  # #9 metadata filter
    ) -> Select[tuple[ChunkModel, float]]:
        distance_expr = ChunkModel.embedding.cosine_distance(query_embedding).label("distance")
        stmt = select(ChunkModel, distance_expr).order_by(distance_expr.asc()).limit(top_k)
        if document_id is not None:
            stmt = stmt.where(ChunkModel.document_id == document_id)
        return stmt

    async def query(
        self,
        *,
        query_embedding: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
        document_id: uuid.UUID | None = None,  # #9 metadata filter
    ) -> list[RetrievedChunk]:
        stmt = self._similarity_query(query_embedding, top_k=top_k, document_id=document_id)
        rows = (await self._session.execute(stmt)).all()

        out: list[RetrievedChunk] = []
        for chunk, distance in rows:
            if score_threshold is not None and float(distance) > score_threshold:
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    distance=float(distance),
                )
            )
        return out

    # ------------------------------------------------------------------ #
    #  Document management                                                 #
    # ------------------------------------------------------------------ #

    async def list_documents(self) -> list[DocumentRecord]:
        """Return all documents with their chunk counts."""
        stmt = (
            select(
                DocumentModel.id,
                DocumentModel.filename,
                DocumentModel.content_type,
                DocumentModel.created_at,
                func.count(ChunkModel.id).label("chunk_count"),
            )
            .outerjoin(ChunkModel, ChunkModel.document_id == DocumentModel.id)
            .group_by(
                DocumentModel.id,
                DocumentModel.filename,
                DocumentModel.content_type,
                DocumentModel.created_at,
            )
            .order_by(DocumentModel.created_at.desc())
        )
        rows = (await self._session.execute(stmt)).all()
        return [
            DocumentRecord(
                id=row.id,
                filename=row.filename,
                content_type=row.content_type,
                created_at=row.created_at,
                chunk_count=row.chunk_count,
            )
            for row in rows
        ]

    async def get_document(self, document_id: uuid.UUID) -> DocumentModel | None:
        return await self._session.get(DocumentModel, document_id)

    async def delete_document(self, document_id: uuid.UUID) -> bool:
        """Delete a document and all its chunks (CASCADE). Returns True if found."""
        stmt = delete(DocumentModel).where(DocumentModel.id == document_id)
        result = await self._session.execute(stmt)
        return result.rowcount > 0

    # ------------------------------------------------------------------ #
    #  #7 Token usage / query logs                                        #
    # ------------------------------------------------------------------ #

    async def log_query(
        self,
        *,
        query_text: str,
        chunks_used: int,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        estimated_cost_usd: float | None,
        latency_ms: int,
    ) -> uuid.UUID:
        """Persist per-query telemetry to the query_logs table."""
        entry = QueryLogModel(
            query_text=query_text,
            chunks_used=chunks_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            latency_ms=latency_ms,
        )
        self._session.add(entry)
        await self._session.flush()
        return entry.id

    async def list_query_logs(self, *, limit: int = 100) -> list[QueryLogModel]:
        """Return the most recent query logs."""
        stmt = (
            select(QueryLogModel)
            .order_by(QueryLogModel.created_at.desc())
            .limit(limit)
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return list(rows)

    # ------------------------------------------------------------------ #
    #  #8 Prompt versioning registry                                      #
    # ------------------------------------------------------------------ #

    async def create_prompt(
        self,
        *,
        name: str,
        content: str,
        author: str | None = None,
    ) -> SystemPromptModel:
        """Insert a new prompt version (auto-increments version number)."""
        # Determine next version number for this name
        stmt = select(func.max(SystemPromptModel.version)).where(SystemPromptModel.name == name)
        max_ver: int | None = (await self._session.execute(stmt)).scalar_one_or_none()
        next_ver = (max_ver or 0) + 1

        prompt = SystemPromptModel(
            name=name,
            version=next_ver,
            content=content,
            author=author,
            is_active=False,
        )
        self._session.add(prompt)
        await self._session.flush()
        return prompt

    async def activate_prompt(self, prompt_id: uuid.UUID) -> SystemPromptModel | None:
        """Set one prompt as active; deactivate all others with the same name."""
        prompt = await self._session.get(SystemPromptModel, prompt_id)
        if prompt is None:
            return None
        # Deactivate siblings
        from sqlalchemy import update as sa_update
        await self._session.execute(
            sa_update(SystemPromptModel)
            .where(SystemPromptModel.name == prompt.name, SystemPromptModel.id != prompt_id)
            .values(is_active=False)
        )
        prompt.is_active = True
        await self._session.flush()
        return prompt

    async def get_active_prompt(self, name: str = "default") -> SystemPromptModel | None:
        """Return the currently active prompt for the given name."""
        stmt = (
            select(SystemPromptModel)
            .where(SystemPromptModel.name == name, SystemPromptModel.is_active.is_(True))
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list_prompts(self, name: str | None = None) -> list[SystemPromptModel]:
        stmt = select(SystemPromptModel).order_by(
            SystemPromptModel.name, SystemPromptModel.version.desc()
        )
        if name:
            stmt = stmt.where(SystemPromptModel.name == name)
        return list((await self._session.execute(stmt)).scalars().all())

    # ------------------------------------------------------------------ #
    #  #14 Service-level metrics                                           #
    # ------------------------------------------------------------------ #

    async def get_metrics(self) -> ServiceMetrics:
        """Return aggregate service stats across all query logs, documents, and chunks."""
        # Query log aggregates
        log_stmt = select(
            func.count(QueryLogModel.id).label("total_queries"),
            func.avg(QueryLogModel.latency_ms).label("avg_latency_ms"),
            func.sum(QueryLogModel.total_tokens).label("total_tokens"),
            func.avg(QueryLogModel.total_tokens).label("avg_tokens_per_query"),
            func.sum(QueryLogModel.estimated_cost_usd).label("total_cost"),
        )
        log_row = (await self._session.execute(log_stmt)).one()

        # Document count
        doc_count: int = (
            await self._session.execute(select(func.count(DocumentModel.id)))
        ).scalar_one() or 0

        # Chunk count
        chunk_count: int = (
            await self._session.execute(select(func.count(ChunkModel.id)))
        ).scalar_one() or 0

        return ServiceMetrics(
            total_queries=int(log_row.total_queries or 0),
            total_documents=doc_count,
            total_chunks=chunk_count,
            avg_latency_ms=round(float(log_row.avg_latency_ms), 2) if log_row.avg_latency_ms else None,
            total_tokens=int(log_row.total_tokens or 0),
            avg_tokens_per_query=round(float(log_row.avg_tokens_per_query), 2) if log_row.avg_tokens_per_query else None,
            total_estimated_cost_usd=round(float(log_row.total_cost), 8) if log_row.total_cost else None,
        )
