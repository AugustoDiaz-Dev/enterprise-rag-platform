# Architecture Overview — Enterprise RAG Platform

> **Project 1** · `project-1/enterprise-rag/`

## System Overview

The Enterprise RAG Platform is a production-ready **Retrieval-Augmented Generation** API that enables organisations to upload private documents, index them as semantic embeddings, and query them with natural language — all without exposing data to external vector databases.

```
                 ┌──────────────┐
   PDF Upload    │              │  POST /documents
  ──────────────▶│  FastAPI API │──────────────────────▶  Ingestion Pipeline
                 │   (async)   │                           │
   NL Query      │              │  POST /query              ▼
  ──────────────▶│              │──────────────┐    ┌─────────────────┐
                 └──────────────┘              │    │  Chunker (#16)  │
                                               │    │  Embedder       │
                                               │    │  VectorStore    │
                                               ▼    └────────┬────────┘
                                      ┌──────────────┐       │ write
                                      │    LLM Layer │       ▼
                                      │  (OpenAI /   │  ┌──────────────────┐
                                      │   Ollama)    │  │  PostgreSQL      │
                                      └──────┬───────┘  │  + pgvector      │
                                             │           │                  │
                                             │ answer    │  documents       │
                                             ▼           │  chunks (vec)   │
                                        Response         │  query_logs     │
                                                         │  system_prompts │
                                                         └──────────────────┘
```

---

## Component Breakdown

### API Layer — `app/api/routes.py`

FastAPI async router exposing:

| Endpoint | Feature # | Purpose |
|---|---|---|
| `GET /health` | — | Liveness check |
| `GET /metrics` | #14 | Aggregate service stats |
| `POST /documents` | #1, #6 | PDF ingestion with OCR fallback |
| `GET /documents` | #4 | List all indexed documents |
| `DELETE /documents/{id}` | #5 | Remove document + chunks |
| `POST /query` | #3, #9, #10 | Semantic query with filtering + debug |
| `GET /query-logs` | #7 | Historical token usage logs |
| `POST /prompts` | #8 | Create versioned system prompt |
| `GET /prompts` | #8 | List prompt versions |
| `PUT /prompts/{id}/activate` | #8 | Promote a prompt version |

### RAG Pipeline — `app/rag/`

```
PDF bytes
   │
   ▼
extract_text_from_pdf()          ← pypdf first; Tesseract OCR fallback (#6)
   │
   ▼
chunk_text()                     ← Token-aware, sentence-boundary chunker (#16)
   │ list[Chunk]
   ▼
EmbeddingProvider.embed_texts()  ← OpenAI text-embedding-3-small (1536-dim)
   │ list[list[float]]
   ▼
VectorStore.add_chunks()         ← INSERT into chunks table (pgvector)
```

### Query Pipeline — `app/services/query_service.py`

```
User query string
   │
   ▼
EmbeddingProvider.embed_query()  ← 1536-dim query vector
   │
   ▼
Retriever.retrieve()             ← cosine similarity top-k (#3 threshold, #9 filter)
   │ list[RetrievedChunk]
   ▼
QueryService.query()
   ├─ get_active_prompt()        ← #8 prompt registry
   ├─ BaseLLM.complete()         ← #11 OpenAI / Ollama
   ├─ log_query()               ← #7 token + cost tracking
   └─ debug_info assembly       ← #10 retrieval explain
   │
   ▼
QueryResponse (JSON)
```

### LLM Abstraction — `app/llm/`

| Class | Provider | Notes |
|---|---|---|
| `OpenAILLM` | OpenAI GPT-4o-mini | tenacity retry/backoff |
| `LocalModelProvider` | Ollama HTTP API | any locally-pulled model |

Selected via `LLM_PROVIDER` env var.

### Database — `app/db/`

| Table | Purpose |
|---|---|
| `documents` | Document metadata + SHA-256 hash for deduplication |
| `chunks` | Text chunks + 1536-dim embeddings (`pgvector`) |
| `query_logs` | Per-query telemetry: tokens, cost, latency |
| `system_prompts` | Versioned system prompts with active flag |

---

## Request Lifecycle (Query)

```
Client
  │  POST /query  { query, top_k, document_id?, score_threshold?, debug? }
  ▼
routes.py::query()
  │
  │  embed query
  ▼
EmbeddingProvider (OpenAI / hash)
  │  [0.12, -0.44, …]  (1536-dim)
  ▼
VectorStore.query()
  │  SELECT chunks WHERE distance < threshold  [+ WHERE document_id = ?]
  │  ORDER BY embedding <=> query_embedding
  │  LIMIT top_k
  ▼
QueryService
  │  Assemble context passages
  │  Load active system prompt from DB
  │  Call LLM (OpenAI / Ollama)
  │  Compute cost estimate
  │  INSERT query_logs
  ▼
QueryResponse  { answer, chunks, tokens, cost, debug_info? }
  │
Client
```

---

## Data Flow Diagram (Ingestion)

```
Client
  │  POST /documents  (multipart PDF)
  ▼
routes.py::upload_document()
  │  sha256(pdf_bytes) → check duplicate → skip if exists (#1)
  │
  │  extract_text_from_pdf(pdf_bytes)
  │    ├─ pypdf.PdfReader  → text  (fast path)
  │    └─ pdf2image + pytesseract → OCR fallback (#6)
  │
  │  chunk_text(text)   ← sentence-boundary, token-aware (#16)
  │
  │  embedder.embed_texts(chunks)  ← OpenAI batch call
  │
  │  INSERT documents  (filename, hash, content_type)
  │  INSERT chunks[]   (document_id, text, embedding)
  ▼
DocumentIngestResponse  { document_id, chunks_ingested, ocr_used }
```
