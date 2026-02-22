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
from app.api.deps import get_api_key
from app.rag.vector_store import VectorStore
from app.schemas import (
    DocumentIngestResponse,
    DocumentListResponse,
    PromptCreate,
    PromptListResponse,
    PromptOut,
    QueryLogOut,
    QueryRequest,
    QueryResponse,
    ServiceMetricsOut,
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
#  Metrics  #14                                                               #
# --------------------------------------------------------------------------- #

@router.get("/metrics", response_model=ServiceMetricsOut)
async def get_metrics(
    session: AsyncSession = Depends(get_session),
    _ = Depends(get_api_key),
) -> ServiceMetricsOut:
    """#14 — Service-level metrics: query count, avg latency, tokens, cost, docs, chunks."""
    store = VectorStore(session)
    m = await store.get_metrics()
    return ServiceMetricsOut(
        total_queries=m.total_queries,
        total_documents=m.total_documents,
        total_chunks=m.total_chunks,
        avg_latency_ms=m.avg_latency_ms,
        total_tokens=m.total_tokens,
        avg_tokens_per_query=m.avg_tokens_per_query,
        total_estimated_cost_usd=m.total_estimated_cost_usd,
    )


# --------------------------------------------------------------------------- #
#  Documents — ingest (#1), list (#4), delete (#5)                            #
# --------------------------------------------------------------------------- #

@router.post("/documents", response_model=DocumentIngestResponse, status_code=201)
async def upload_document(
    session: AsyncSession = Depends(get_session),
    file: UploadFile = File(...),
    _ = Depends(get_api_key),
) -> DocumentIngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    content_type = file.content_type or "application/pdf"
    if content_type not in {"application/pdf"}:
        raise HTTPException(status_code=415, detail=f"Unsupported content_type={content_type!r}")

    pdf_bytes = await file.read()

    # #1 Idempotent ingestion
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    store = VectorStore(session)
    existing_id = await store.find_document_by_hash(file_hash)
    if existing_id is not None:
        logger.info("document_already_exists", extra={"document_id": str(existing_id)})
        return DocumentIngestResponse(
            document_id=existing_id,
            chunks_ingested=0,
            already_existed=True,
        )

    # #6 OCR fallback is handled transparently inside extract_text_from_pdf
    text, ocr_used = extract_text_from_pdf(pdf_bytes)
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

    logger.info("document_ingested", extra={"document_id": str(doc_id), "chunks": len(chunks), "ocr": ocr_used})
    return DocumentIngestResponse(
        document_id=doc_id,
        chunks_ingested=len(chunks),
        already_existed=False,
        ocr_used=ocr_used,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    session: AsyncSession = Depends(get_session),
    _ = Depends(get_api_key),
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
    _ = Depends(get_api_key),
) -> None:
    """#5 — Delete a document and all its chunks."""
    store = VectorStore(session)
    found = await store.delete_document(document_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    await session.commit()
    logger.info("document_deleted", extra={"document_id": str(document_id)})


# --------------------------------------------------------------------------- #
#  Query (#3 threshold, #9 filter, #10 debug)                                 #
# --------------------------------------------------------------------------- #

@router.post("/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    session: AsyncSession = Depends(get_session),
    _ = Depends(get_api_key),
) -> QueryResponse:
    embedder = get_embedding_provider()
    service = QueryService(session, embedder)
    return await service.query(
        query=payload.query,
        top_k=payload.top_k,
        score_threshold=payload.score_threshold,
        document_id=payload.document_id,   # #9
        debug=payload.debug,               # #10
    )


# --------------------------------------------------------------------------- #
#  Query logs — #7 token usage history                                        #
# --------------------------------------------------------------------------- #

@router.get("/query-logs", response_model=list[QueryLogOut])
async def get_query_logs(
    limit: int = 100,
    session: AsyncSession = Depends(get_session),
    _ = Depends(get_api_key),
) -> list[QueryLogOut]:
    """#7 — Return the most recent query logs with token usage and cost estimates."""
    store = VectorStore(session)
    logs = await store.list_query_logs(limit=limit)
    return [
        QueryLogOut(
            id=log.id,
            query_text=log.query_text,
            chunks_used=log.chunks_used,
            prompt_tokens=log.prompt_tokens,
            completion_tokens=log.completion_tokens,
            total_tokens=log.total_tokens,
            estimated_cost_usd=log.estimated_cost_usd,
            latency_ms=log.latency_ms,
            created_at=log.created_at,
        )
        for log in logs
    ]


# --------------------------------------------------------------------------- #
#  Prompt registry — #8 versioned system prompts                              #
# --------------------------------------------------------------------------- #

@router.post("/prompts", response_model=PromptOut, status_code=201)
async def create_prompt(
    payload: PromptCreate,
    session: AsyncSession = Depends(get_session),
    _ = Depends(get_api_key),
) -> PromptOut:
    """#8 — Create a new versioned system prompt."""
    store = VectorStore(session)
    prompt = await store.create_prompt(
        name=payload.name,
        content=payload.content,
        author=payload.author,
    )
    await session.commit()
    logger.info("prompt_created", extra={"name": prompt.name, "version": prompt.version})
    return PromptOut(
        id=prompt.id,
        name=prompt.name,
        version=prompt.version,
        content=prompt.content,
        author=prompt.author,
        is_active=prompt.is_active,
        created_at=prompt.created_at,
    )


@router.get("/prompts", response_model=PromptListResponse)
async def list_prompts(
    name: str | None = None,
    session: AsyncSession = Depends(get_session),
    _ = Depends(get_api_key),
) -> PromptListResponse:
    """#8 — List all prompt versions (optionally filtered by name)."""
    store = VectorStore(session)
    prompts = await store.list_prompts(name=name)
    return PromptListResponse(
        prompts=[
            PromptOut(
                id=p.id,
                name=p.name,
                version=p.version,
                content=p.content,
                author=p.author,
                is_active=p.is_active,
                created_at=p.created_at,
            )
            for p in prompts
        ],
        total=len(prompts),
    )


@router.put("/prompts/{prompt_id}/activate", response_model=PromptOut)
async def activate_prompt(
    prompt_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    _ = Depends(get_api_key),
) -> PromptOut:
    """#8 — Mark a prompt version as active (deactivates all others with the same name)."""
    store = VectorStore(session)
    prompt = await store.activate_prompt(prompt_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")
    await session.commit()
    logger.info("prompt_activated", extra={"id": str(prompt.id), "name": prompt.name, "version": prompt.version})
    return PromptOut(
        id=prompt.id,
        name=prompt.name,
        version=prompt.version,
        content=prompt.content,
        author=prompt.author,
        is_active=prompt.is_active,
        created_at=prompt.created_at,
    )
