"""#16 Robust token-aware chunker with sentence-boundary respect.

Strategy
--------
1.  Estimate token count with the GPT-2 heuristic: ``len(text.split()) * 1.33``.
    This avoids pulling in the ``tiktoken`` library at import time (optional).
2.  Split on sentence boundaries first (``. ! ?`` followed by whitespace).
3.  Accumulate sentences into chunks that stay below ``max_tokens``.
4.  Apply an overlap of ``overlap_tokens`` from the tail of the previous chunk.
5.  Multilingual-safe: sentence splitting is done on unicode whitespace + common
    end-punctuation so it works for most European languages.

The public API (``chunk_text``) is backward-compatible with the old version —
callers that pass ``chunk_size`` / ``overlap`` still work via the compat shim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    token_estimate: int = 0   # approximate GPT token count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _estimate_tokens(text: str) -> int:
    """GPT-2 heuristic: words × 1.33 rounds up to the nearest integer."""
    return max(1, round(len(text.split()) * 1.33))


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on [.!?] + whitespace boundaries."""
    cleaned = " ".join(text.split())          # normalise whitespace
    parts = _SENTENCE_SPLIT_RE.split(cleaned)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    *,
    max_tokens: int = 400,
    overlap_tokens: int = 80,
    # ── backward-compat aliases (old callers used char-based sizes) ──
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Split *text* into overlapping chunks bounded by token count.

    Parameters
    ----------
    text:
        The raw document text to split.
    max_tokens:
        Maximum **estimated** GPT tokens per chunk (default 400 ≈ ~300 words).
    overlap_tokens:
        How many tokens from the end of the previous chunk to prepend to the
        next one, preserving context across boundaries (default 80).
    chunk_size / overlap:
        Legacy character-based parameters. When provided they are converted to
        a rough token estimate via ``chunk_size // 4`` and ``overlap // 4``.
    """
    # -- compat shim ---------------------------------------------------------
    if chunk_size is not None:
        max_tokens = max(1, chunk_size // 4)
    if overlap is not None:
        overlap_tokens = max(0, overlap // 4)
    # ------------------------------------------------------------------------

    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    sentences = _split_sentences(cleaned)

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    idx = 0

    for sentence in sentences:
        sent_tokens = _estimate_tokens(sentence)

        # If a single sentence is larger than the budget, split it hard
        if sent_tokens > max_tokens:
            # Flush what we have
            if current_sentences:
                body = " ".join(current_sentences)
                chunks.append(Chunk(index=idx, text=body, token_estimate=current_tokens))
                idx += 1
                current_sentences = []
                current_tokens = 0

            # Hard-split the giant sentence by words
            words = sentence.split()
            buf: list[str] = []
            buf_tokens = 0
            for word in words:
                wt = _estimate_tokens(word)
                if buf_tokens + wt > max_tokens and buf:
                    body = " ".join(buf)
                    chunks.append(Chunk(index=idx, text=body, token_estimate=buf_tokens))
                    idx += 1
                    # overlap from buf tail
                    overlap_words = _tail_words(buf, overlap_tokens)
                    buf = overlap_words
                    buf_tokens = _estimate_tokens(" ".join(buf)) if buf else 0
                buf.append(word)
                buf_tokens += wt
            if buf:
                current_sentences = [" ".join(buf)]
                current_tokens = buf_tokens
            continue

        # Normal case: would adding this sentence overflow the chunk?
        if current_tokens + sent_tokens > max_tokens and current_sentences:
            body = " ".join(current_sentences)
            chunks.append(Chunk(index=idx, text=body, token_estimate=current_tokens))
            idx += 1

            # Carry-over overlap — take sentences from the tail
            overlap_sents = _tail_sentences(current_sentences, overlap_tokens)
            current_sentences = overlap_sents
            current_tokens = _estimate_tokens(" ".join(current_sentences)) if current_sentences else 0

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    # Flush remainder
    if current_sentences:
        body = " ".join(current_sentences)
        chunks.append(Chunk(index=idx, text=body, token_estimate=current_tokens))

    return chunks


# ---------------------------------------------------------------------------
# Overlap helpers
# ---------------------------------------------------------------------------

def _tail_sentences(sentences: list[str], overlap_tokens: int) -> list[str]:
    """Return sentences from the tail of *sentences* that fit in *overlap_tokens*."""
    if not sentences or overlap_tokens <= 0:
        return []
    result: list[str] = []
    used = 0
    for s in reversed(sentences):
        t = _estimate_tokens(s)
        if used + t > overlap_tokens:
            break
        result.insert(0, s)
        used += t
    return result


def _tail_words(words: list[str], overlap_tokens: int) -> list[str]:
    """Return words from the tail of *words* that fit in *overlap_tokens*."""
    if not words or overlap_tokens <= 0:
        return []
    result: list[str] = []
    used = 0
    for w in reversed(words):
        t = _estimate_tokens(w)
        if used + t > overlap_tokens:
            break
        result.insert(0, w)
        used += t
    return result
