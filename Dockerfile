# ── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build tools needed to compile Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first (layer cache: only reinstall when deps change)
COPY pyproject.toml ./

# Install production dependencies into an isolated prefix
RUN pip install --upgrade pip && \
    pip install --prefix=/install ".[dev]"

# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# System-level packages required at runtime:
#   - libpq5: Postgres client lib (asyncpg needs it)
#   - tesseract-ocr + poppler-utils: OCR fallback for scanned PDFs (#6)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/
COPY pyproject.toml ./

# Non-root user for security
RUN useradd --no-create-home --shell /bin/false appuser
USER appuser

EXPOSE 8000

# Entrypoint: production-grade uvicorn (no --reload in production)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
