from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from app.core.config import settings


Embedding = list[float]


class EmbeddingProvider:
    dim: int

    async def embed_texts(self, texts: list[str]) -> list[Embedding]:
        raise NotImplementedError

    async def embed_query(self, text: str) -> Embedding:
        (vec,) = await self.embed_texts([text])
        return vec


# ---------------------------------------------------------------------------
# Hash provider (dev / offline)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HashEmbeddingProvider(EmbeddingProvider):
    dim: int = 128

    async def embed_texts(self, texts: list[str]) -> list[Embedding]:
        vectors: list[Embedding] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            buf = bytearray()
            cur = h
            while len(buf) < self.dim:
                buf.extend(cur)
                cur = hashlib.sha256(cur).digest()
            raw = buf[: self.dim]
            vec = [(b - 128) / 128.0 for b in raw]
            vectors.append(vec)
        return vectors


# ---------------------------------------------------------------------------
# OpenAI provider (production)
# ---------------------------------------------------------------------------

@dataclass
class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Uses OpenAI text-embedding-3-small (1536-dim) via the official SDK."""

    model: str = "text-embedding-3-small"
    dim: int = field(init=False)

    # Dimension map — extend if you switch models
    _DIM_MAP: dict[str, int] = field(
        default_factory=lambda: {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        },
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.dim = self._DIM_MAP.get(self.model, 1536)

    async def embed_texts(self, texts: list[str]) -> list[Embedding]:
        from openai import AsyncOpenAI  # lazy import — not required at module load

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        # OpenAI recommends replacing newlines with spaces
        cleaned = [t.replace("\n", " ") for t in texts]
        response = await client.embeddings.create(model=self.model, input=cleaned)
        return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedding_provider() -> EmbeddingProvider:
    provider = settings.embedding_provider.lower().strip()

    if provider in {"hash", "dev", "local"}:
        return HashEmbeddingProvider()

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        return OpenAIEmbeddingProvider()

    raise ValueError(f"Unknown EMBEDDING_PROVIDER={settings.embedding_provider!r}")
