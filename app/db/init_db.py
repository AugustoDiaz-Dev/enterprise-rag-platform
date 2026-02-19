from __future__ import annotations

import logging

from sqlalchemy import text

from app.db.models import Base
from app.db.session import engine

logger = logging.getLogger(__name__)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)

        # Migration: add file_hash column if it doesn't exist yet (idempotent).
        # Safe to run on both fresh and existing databases.
        await conn.execute(
            text("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS file_hash VARCHAR(64) UNIQUE;
            """)
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_documents_file_hash ON documents (file_hash);")
        )

    logger.info("database_initialized")
