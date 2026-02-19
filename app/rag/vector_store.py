from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Select, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk as ChunkModel
from app.db.models import Document as DocumentModel


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
        row = result.scalar_one_or_none()
        return row

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
        self, query_embedding: list[float], *, top_k: int
    ) -> Select[tuple[ChunkModel, float]]:
        distance_expr = ChunkModel.embedding.cosine_distance(query_embedding).label("distance")
        stmt = select(ChunkModel, distance_expr).order_by(distance_expr.asc()).limit(top_k)
        return stmt

    async def query(
        self,
        *,
        query_embedding: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        stmt = self._similarity_query(query_embedding, top_k=top_k)
        rows = (await self._session.execute(stmt)).all()

        out: list[RetrievedChunk] = []
        for chunk, distance in rows:
            # #3 Score thresholding: skip chunks that exceed the distance threshold
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
    #  Document management (#4 list, #5 delete)                           #
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
        """Fetch a single document by id."""
        return await self._session.get(DocumentModel, document_id)

    async def delete_document(self, document_id: uuid.UUID) -> bool:
        """Delete a document and all its chunks (CASCADE handles chunks). Returns True if found."""
        stmt = delete(DocumentModel).where(DocumentModel.id == document_id)
        result = await self._session.execute(stmt)
        return result.rowcount > 0
