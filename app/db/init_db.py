from __future__ import annotations

import logging

from sqlalchemy import text

from app.db.models import Base
from app.db.session import engine

logger = logging.getLogger(__name__)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        # create_all is idempotent — safe to run on every startup
        await conn.run_sync(Base.metadata.create_all)

        # ── Inline migrations for previously existing databases ─────────
        # Add file_hash column to documents if not present (#1)
        await conn.execute(
            text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_hash VARCHAR(64) UNIQUE;")
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_documents_file_hash ON documents (file_hash);")
        )

    logger.info("database_initialized")
