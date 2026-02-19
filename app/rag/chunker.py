from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


def chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 200) -> list[Chunk]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        segment = cleaned[start:end].strip()
        if segment:
            chunks.append(Chunk(index=idx, text=segment))
            idx += 1
        if end == len(cleaned):
            break
        start = max(0, end - overlap)

    return chunks

