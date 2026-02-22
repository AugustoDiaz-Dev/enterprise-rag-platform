"""#17 — Chunker unit tests (token-aware, sentence-boundary chunker)."""
from __future__ import annotations

import pytest

from app.rag.chunker import Chunk, _estimate_tokens, _split_sentences, chunk_text


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------

def test_estimate_tokens_empty() -> None:
    assert _estimate_tokens("") == 1   # max(1, …)


def test_estimate_tokens_single_word() -> None:
    result = _estimate_tokens("hello")
    assert result >= 1


def test_estimate_tokens_proportional() -> None:
    short = _estimate_tokens("hello world")
    long = _estimate_tokens("hello world foo bar baz qux quux corge grault garply")
    assert long > short


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------

def test_split_sentences_basic() -> None:
    text = "Hello world. How are you? I am fine!"
    parts = _split_sentences(text)
    assert len(parts) == 3
    assert parts[0] == "Hello world."
    assert parts[1] == "How are you?"
    assert parts[2] == "I am fine!"


def test_split_sentences_no_punctuation() -> None:
    text = "just one long run-on sentence without ending"
    parts = _split_sentences(text)
    assert len(parts) == 1


def test_split_sentences_normalises_whitespace() -> None:
    text = "  First sentence.   Second   sentence.  "
    parts = _split_sentences(text)
    assert all(p == p.strip() for p in parts)


# ---------------------------------------------------------------------------
# chunk_text — basic
# ---------------------------------------------------------------------------

def test_chunk_text_empty_returns_empty() -> None:
    assert chunk_text("") == []


def test_chunk_text_whitespace_only_returns_empty() -> None:
    assert chunk_text("   \n\t  ") == []


def test_chunk_text_single_short_sentence() -> None:
    chunks = chunk_text("Hello world.", max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert "Hello world" in chunks[0].text


def test_chunk_text_indices_are_sequential() -> None:
    text = " ".join(["Word" + str(i) + "." for i in range(200)])
    chunks = chunk_text(text, max_tokens=50, overlap_tokens=10)
    for i, c in enumerate(chunks):
        assert c.index == i


def test_chunk_text_no_empty_chunks() -> None:
    text = "a" * 5000
    chunks = chunk_text(text, max_tokens=100, overlap_tokens=20)
    assert all(c.text.strip() for c in chunks)


def test_chunk_text_respects_max_tokens() -> None:
    """Each chunk's token_estimate should not greatly exceed max_tokens."""
    text = " ".join(["Word" + str(i) + "." for i in range(500)])
    max_t = 100
    chunks = chunk_text(text, max_tokens=max_t, overlap_tokens=20)
    for c in chunks:
        # Allow a small overshoot for single giant sentences
        assert c.token_estimate <= max_t * 1.5


def test_chunk_text_overlap_carries_context() -> None:
    """The start of chunk N should share some text with the end of chunk N-1."""
    sentences = [f"This is sentence number {i}." for i in range(30)]
    text = " ".join(sentences)
    chunks = chunk_text(text, max_tokens=30, overlap_tokens=10)
    assert len(chunks) >= 2
    # Verify consecutive chunks share at least one word
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_words = set(prev.text.split())
        nxt_words = set(nxt.text.split())
        assert prev_words & nxt_words, "No word overlap between consecutive chunks"


def test_chunk_text_overlap_must_be_smaller_than_max() -> None:
    with pytest.raises(ValueError, match="overlap_tokens must be smaller"):
        chunk_text("some text", max_tokens=50, overlap_tokens=50)


# ---------------------------------------------------------------------------
# chunk_text — backward-compat shim (char-based args)
# ---------------------------------------------------------------------------

def test_chunk_text_compat_chunk_size() -> None:
    text = "a" * 5000
    chunks = chunk_text(text, chunk_size=1000, overlap=100)
    assert len(chunks) >= 1
    assert chunks[0].index == 0
    assert all(c.text for c in chunks)


# ---------------------------------------------------------------------------
# chunk_text — giant sentence (hard word-split path)
# ---------------------------------------------------------------------------

def test_chunk_text_giant_sentence_is_split() -> None:
    """A single sentence with > max_tokens should still produce multiple chunks."""
    giant = " ".join(["word"] * 1000)
    chunks = chunk_text(giant, max_tokens=50, overlap_tokens=10)
    assert len(chunks) > 1
    assert all(c.text for c in chunks)


# ---------------------------------------------------------------------------
# chunk_text — multilingual text
# ---------------------------------------------------------------------------

def test_chunk_text_multilingual() -> None:
    """Basic smoke test for non-ASCII (Spanish) input."""
    text = (
        "El cielo es azul. La tierra gira alrededor del sol. "
        "Los árboles crecen en el bosque. El agua corre por el río."
    )
    chunks = chunk_text(text, max_tokens=15, overlap_tokens=3)
    assert len(chunks) >= 1
    assert all(c.text for c in chunks)


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

def test_chunk_is_frozen() -> None:
    c = Chunk(index=0, text="hello")
    with pytest.raises((AttributeError, TypeError)):
        c.index = 99  # type: ignore[misc]
