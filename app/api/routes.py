from __future__ import annotations

import hashlib
import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.rag.chunker import chunk_text
from app.rag.embedding import get_embedding_provider
from app.rag.ingestion import extract_text_from_pdf
from app.rag.vector_store import VectorStore
from app.schemas import (
    DocumentIngestResponse,
    DocumentListResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.query_service import QueryService

logger = logging.getLogger(__name__)
router = APIRouter()


# --------------------------------------------------------------------------- #
#  Health                                                                      #
# --------------------------------------------------------------------------- #

@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
#  Documents — ingest (#1 idempotency), list (#4), delete (#5)               #
# --------------------------------------------------------------------------- #

@router.post("/documents", response_model=DocumentIngestResponse, status_code=201)
async def upload_document(
    session: AsyncSession = Depends(get_session),
    file: UploadFile = File(...),
) -> DocumentIngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    content_type = file.content_type or "application/pdf"
    if content_type not in {"application/pdf"}:
        raise HTTPException(status_code=415, detail=f"Unsupported content_type={content_type!r}")

    pdf_bytes = await file.read()

    # #1 Idempotent ingestion — compute SHA-256 and skip if already stored
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    store = VectorStore(session)
    existing_id = await store.find_document_by_hash(file_hash)
    if existing_id is not None:
        logger.info("document_already_exists", extra={"document_id": str(existing_id), "hash": file_hash})
        return DocumentIngestResponse(
            document_id=existing_id,
            chunks_ingested=0,
            already_existed=True,
        )

    text = extract_text_from_pdf(pdf_bytes)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in PDF")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Unable to chunk extracted text")

    embedder = get_embedding_provider()
    embeddings = await embedder.embed_texts([c.text for c in chunks])

    doc_id = await store.create_document(
        filename=file.filename,
        content_type=content_type,
        file_hash=file_hash,
    )
    await store.add_chunks(
        document_id=doc_id,
        chunks=[(c.index, c.text) for c in chunks],
        embeddings=embeddings,
    )
    await session.commit()

    logger.info("document_ingested", extra={"document_id": str(doc_id), "chunks": len(chunks)})
    return DocumentIngestResponse(document_id=doc_id, chunks_ingested=len(chunks), already_existed=False)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    session: AsyncSession = Depends(get_session),
) -> DocumentListResponse:
    """#4 — List all ingested documents with metadata and chunk counts."""
    store = VectorStore(session)
    docs = await store.list_documents()
    return DocumentListResponse(
        documents=[
            {
                "id": d.id,
                "filename": d.filename,
                "content_type": d.content_type,
                "created_at": d.created_at,
                "chunk_count": d.chunk_count,
            }
            for d in docs
        ],
        total=len(docs),
    )


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    """#5 — Delete a document and all its chunks."""
    store = VectorStore(session)
    found = await store.delete_document(document_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    await session.commit()
    logger.info("document_deleted", extra={"document_id": str(document_id)})


# --------------------------------------------------------------------------- #
#  Query (#3 score threshold applied inside QueryService)                     #
# --------------------------------------------------------------------------- #

@router.post("/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    session: AsyncSession = Depends(get_session),
) -> QueryResponse:
    embedder = get_embedding_provider()
    service = QueryService(session, embedder)
    return await service.query(
        query=payload.query,
        top_k=payload.top_k,
        score_threshold=payload.score_threshold,
    )
