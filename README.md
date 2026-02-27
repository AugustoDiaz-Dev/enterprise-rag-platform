# Enterprise RAG Platform

A production-ready **Retrieval-Augmented Generation (RAG)** API built with FastAPI, PostgreSQL (pgvector), and OpenAI.

Upload PDF documents, chunk and embed them into a vector store, then query the knowledge base with natural language — all via a clean REST API.

## Features

- 📄 **PDF Ingestion** — Upload PDFs via multipart form; they are automatically chunked and embedded
- 🔍 **Semantic Search** — Top-k retrieval using pgvector cosine similarity
- 🤖 **LLM Answer Synthesis** — Pluggable LLM layer (OpenAI by default, easily swappable)
- ⚡ **Async FastAPI** — Fully async stack with SQLAlchemy asyncio
- 🐘 **PostgreSQL + pgvector** — Persistent vector storage with no external vector DB required
- 🐳 **Docker-ready** — One command to spin up the database
- 🧾 **Source citations** — Responses reference `[Passage N]` labels, and the API exposes a `citations` array so the UI can link back to the retrieved evidence.

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| ORM | SQLAlchemy (async) |
| Database | PostgreSQL + pgvector |
| Embeddings | OpenAI `text-embedding-3-small` (configurable) |
| LLM | OpenAI GPT (pluggable via `LLMProvider` base class) |
| Config | Pydantic Settings + `.env` |

## Requirements

- Python **3.11+**
- Docker (for PostgreSQL + pgvector)

## Quickstart

### 1. Start the database

```bash
docker compose up -d
```

### 2. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The `pyproject.toml` now uses `tool.setuptools.packages.find`, so the editable install succeeds even with the multi-package layout (`app/`, `eval/`, `datasets/`).

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your values
```

### 4. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

The interactive docs will be available at **http://localhost:8000/docs**

### Ports

- `8000` — FastAPI API + Vue dashboard (`/api` routes + `/dashboard` static SPA)
- `5433` — PostgreSQL + pgvector (mapped from Docker service `postgres`)

### Sample demo asset

A representative PDF is bundled at `demo/sample-enterprise-rag.pdf`. Upload it once you've started the API (and `docker compose up -d` for Postgres) to see ingestion, chunking, citations, and the dashboard chat in action:

```bash
curl -X POST "http://localhost:8000/api/documents" \
  -H "X-API-Key: rag-admin-secret" \
  -F "file=@demo/sample-enterprise-rag.pdf"
```

Use the same API key (`X-API-Key: rag-admin-secret`) for all protected endpoints such as `/api/query`, `/api/metrics`, and `/api/prompts`.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/documents` | Upload a PDF (multipart: `file=<pdf>`) |
| `POST` | `/query` | Query the knowledge base |

### Example: Upload a document

```bash
curl -X POST "http://localhost:8000/documents" \
  -F "file=@your-document.pdf"
```

### Example: Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "top_k": 5}'
```

## Project Structure

```
enterprise-rag/
├── app/
│   ├── api/          # Route handlers
│   ├── core/         # Config & logging
│   ├── db/           # Models, session, DB init
│   ├── llm/          # LLM provider abstraction
│   ├── rag/          # Chunker, embedder, retriever, vector store
│   ├── services/     # Query orchestration
│   ├── main.py       # FastAPI app entry point
│   └── schemas.py    # Pydantic request/response schemas
├── tests/
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

## Environment Variables

See `.env.example` for all required variables:

```env
DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5432/rag
APP_ENV=dev
LOG_LEVEL=INFO
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your-key-here
```

## Running Tests

```bash
pytest
```

## License

MIT
