# Evaluation Methodology — Enterprise RAG Platform

## Overview

Quality evaluation for a RAG system covers two orthogonal dimensions:

| Dimension | Question | Tooling |
|---|---|---|
| **Retrieval quality** | Are the right chunks being retrieved? | Ragas `context_precision`, `context_recall` |
| **Generation quality** | Is the LLM answer faithful and relevant? | Ragas `faithfulness`, `answer_relevancy` |

---

## Ragas Metrics

### Faithfulness
Measures whether every claim in the generated answer is supported by the retrieved context. Computed by the Ragas evaluation LLM by cross-checking each statement in the answer against the context passages.

- **Threshold**: ≥ 0.80
- **Why it matters**: Low faithfulness = hallucinations. The RAG system's primary value proposition is grounding answers in evidence.

### Answer Relevancy
Measures how directly the answer addresses the question. A long answer that goes off-topic scores low.

- **Threshold**: ≥ 0.75
- **Why it matters**: Ensures the LLM is focused on the actual question rather than producing verbose, off-topic responses.

### Context Precision
Measures whether the retrieved chunks are ranked correctly — i.e., the most relevant chunks appear at the top of the list.

- **Threshold**: ≥ 0.70
- **Why it matters**: The LLM attends more to earlier context passages. Poor precision wastes the context window.

### Context Recall
Measures whether the golden ground-truth answer is actually supported by the retrieved chunks. Low recall means relevant information exists in the document but isn't being retrieved.

- **Threshold**: ≥ 0.70
- **Why it matters**: Low recall = the chunking or embedding strategy is losing information.

---

## Golden Dataset

Located at `datasets/rag/golden_set.json`.

Each entry contains:

```json
{
  "id": "gs-001",
  "question": "...",
  "ground_truth": "...",       ← concise factual answer for recall scoring
  "contexts": ["...", "..."],  ← ideal context passages
  "reference_answer": "...",   ← longer reference for faithfulness scoring
  "tags": ["definition"],
  "difficulty": "easy|medium|hard"
}
```

### Distribution

| Difficulty | Count |
|---|---|
| Easy | 3 |
| Medium | 5 |
| Hard | 2 |
| **Total** | **10** |

Topics covered: RAG overview, pgvector, chunking, idempotent ingestion, OCR fallback, metadata filtering, prompt versioning, token tracking, debug mode, local LLM.

---

## Running an Evaluation

### Prerequisites

```bash
# 1. Install eval dependencies (not part of main package)
pip install ragas datasets langchain-openai

# 2. Ensure the API is running with a document already ingested
docker compose up -d
# ... ingest a PDF ...
```

### Execute

```bash
python -m eval.ragas_eval \
    --api-url http://localhost:8000 \
    --golden-set datasets/rag/golden_set.json \
    --output eval/reports/ragas_report.json \
    --top-k 5
```

### Output

A JSON report is written to `eval/reports/ragas_report.json`:

```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "scores": {
    "faithfulness": 0.87,
    "answer_relevancy": 0.82,
    "context_precision": 0.74,
    "context_recall": 0.71
  },
  "thresholds": { ... },
  "passed": true
}
```

---

## Latency & Token Budget

Beyond Ragas scores, monitor operational metrics via `GET /metrics`:

| Metric | Target |
|---|---|
| `avg_latency_ms` | < 3000ms (p50) |
| `avg_tokens_per_query` | < 2000 tokens |
| `total_estimated_cost_usd` | Track week-over-week |

---

## Iteration Loop

```
1. Run evaluation  →  identify failing metric
2. If faithfulness ↓  →  review system prompt, reduce top_k, tighten score_threshold
3. If context_recall ↓  →  lower score_threshold, increase top_k, review chunking
4. If context_precision ↓  →  review embedding model, try reranker (future)
5. If answer_relevancy ↓  →  refine system prompt instructions
6. Update golden_set with new question categories as needed
7. Re-run evaluation  →  confirm improvement
```
