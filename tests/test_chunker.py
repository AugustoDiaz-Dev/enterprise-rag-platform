from app.rag.chunker import chunk_text


def test_chunk_text_basic() -> None:
    text = "a" * 5000
    chunks = chunk_text(text, chunk_size=1000, overlap=100)
    assert len(chunks) >= 5
    assert chunks[0].index == 0
    assert all(c.text for c in chunks)
