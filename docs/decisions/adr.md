# Design Decisions — Enterprise RAG Platform

This document captures the key architectural and implementation choices made during development. Decisions are recorded here so future contributors understand not just *what* was done but *why*.

---

## ADR-001 · PostgreSQL + pgvector instead of a dedicated vector DB

**Decision**: Use `pgvector` as an extension on PostgreSQL rather than adopting Pinecone, Weaviate, or Qdrant.

**Context**: The project needed persistent vector storage with metadata. Dedicated vector databases offer advanced ANN indices but add operational complexity (another service to manage, another billing dimension).

**Rationale**:
- Single data store — documents, chunks, query logs, and prompts all live in one PostgreSQL instance.
- pgvector's cosine distance is fast enough for the expected document volumes (<100k chunks).
- Available as a Docker image (`pgvector/pgvector:pg16`) with zero extra config.
- SQLAlchemy asyncio + pgvector has first-class Python support.

**Trade-off**: At millions of chunks, HNSW indexing on pgvector may need tuning. Migration to a dedicated vector DB is possible later without changing the retrieval interface.

---

## ADR-002 · Sentence-boundary token-aware chunker over fixed-character splitting

**Decision**: Replace the original character-based chunker with a token-aware, sentence-boundary chunker (#16).

**Context**: Character-based splitting can break sentences mid-word, destroying semantic coherence, and its "size" has no relationship to what the LLM sees (token budget).

**Rationale**:
- Sentence-boundary splits produce more semantically coherent passages.
- Token estimation (words × 1.33) approximates GPT-2 tokenisation without requiring `tiktoken` at import time.
- Overlap is now specified in tokens rather than characters, which maps directly to the LLM context window.

**Trade-off**: The GPT-2 heuristic slightly overestimates non-English tokens. Exact tiktoken-based counting is possible as a future upgrade without changing the public API.

---

## ADR-003 · LLM abstraction layer (`BaseLLM`)

**Decision**: Define a `BaseLLM` abstract class with a single `async complete(system, user) → LLMResponse` method.

**Context**: The initial implementation only targeted OpenAI. However, cost concerns and data-sovereignty requirements may demand local models.

**Rationale**:
- Swapping providers requires only changing `LLM_PROVIDER` in `.env` — no code changes.
- `LLMResponse` is a standard dataclass with `content`, `prompt_tokens`, `completion_tokens`, `total_tokens` — decoupled from any provider SDK.
- `LocalModelProvider` (Ollama) implements the same interface, enabling zero-code switching.

---

## ADR-004 · Prompt versioning stored in the database

**Decision**: Store system prompts in the `system_prompts` table with auto-incrementing version numbers per name.

**Context**: Prompt engineering is iterative. Without versioning, there's no audit trail of what prompt produced a given answer.

**Rationale**:
- DB storage makes prompts queryable, auditable, and rollback-able.
- Auto-incrementing version per `name` means you can have multiple named prompt families (e.g., `default`, `finance`, `legal`).
- `is_active` flag allows instant promotion/rollback via the API without redeployment.

**Trade-off**: Prompts in the DB are separated from code, so a Git diff won't show prompt changes. Mitigation: export prompts to `datasets/prompts/` as part of a release process.

---

## ADR-005 · SHA-256 hash for idempotent ingestion

**Decision**: Compute `sha256(pdf_bytes)` on every upload and skip ingestion if the hash already exists.

**Context**: Re-uploading the same document should not duplicate chunks in the vector store or cause duplicate embeddings.

**Rationale**:
- SHA-256 is collision-resistant and fast for typical PDF sizes.
- The hash is stored in the `documents.file_hash` column with a unique index.
- Returns `already_existed=true` with the existing document ID immediately — zero DB writes.

---

## ADR-006 · Token usage stored per query (not aggregated)

**Decision**: Persist raw `prompt_tokens`, `completion_tokens`, `total_tokens`, and `estimated_cost_usd` per query in `query_logs`.

**Context**: Budget tracking requires historical granularity.

**Rationale**:
- Raw per-query logs allow any aggregation downstream (`GET /metrics` sums/averages them).
- Latency (`latency_ms`) is captured end-to-end including retrieval + LLM call.
- Cost estimation uses the current GPT-4o-mini pricing constants and is `NULL` for local models.

---

## ADR-007 · OCR as a transparent fallback (not a separate endpoint)

**Decision**: OCR is invoked inside `extract_text_from_pdf()` automatically when `pypdf` returns empty text. A single `POST /documents` endpoint handles both normal and scanned PDFs.

**Context**: Callers shouldn't need to know whether a PDF is text-based or scanned.

**Rationale**:
- Transparent fallback reduces API surface area.
- `ocr_used` in the response allows callers to observe when OCR was triggered.
- System dependencies (`tesseract`, `poppler`) are installed in the `Dockerfile` and documented in the README.

---

## ADR-008 · Ragas for evaluation (not custom metrics)

**Decision**: Use the open-source Ragas library for automated evaluation rather than building bespoke metrics.

**Context**: Meaningful RAG evaluation requires LLM-graded metrics (faithfulness, relevancy), which are non-trivial to implement from scratch.

**Rationale**:
- Ragas provides the industry-standard set of RAG metrics.
- Evaluation dependencies (`ragas`, `datasets`, `langchain-openai`) are kept separate from the main package to avoid inflating production image size.
- The evaluation script (`eval/ragas_eval.py`) queries the running API, making it environment-agnostic.

---

## ADR-009 · GitHub Actions CI with three jobs (lint → unit → integration)

**Decision**: Three-job CI pipeline: ruff lint → unit tests + coverage gate → integration smoke test.

**Rationale**:
- Fast feedback: lint and unit tests are cheap and run first; integration (real DB) only runs if they pass.
- The 80% coverage gate is enforced by `pytest-cov --cov-fail-under=80`.
- Integration job uses the `pgvector/pgvector:pg16` service container — identical to local dev.
- `EMBEDDING_PROVIDER=hash` in CI avoids real OpenAI calls during smoke tests.
