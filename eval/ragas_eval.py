"""#12 Ragas evaluation pipeline.

Usage
-----
1. Ensure the Enterprise RAG API is running and a document has been ingested.
2. Install evaluation dependencies:

       pip install ragas datasets langchain-openai

3. Run:

       python -m eval.ragas_eval \\
           --api-url http://localhost:8000 \\
           --golden-set datasets/rag/golden_set.json \\
           --output eval/reports/ragas_report.json

Environment
-----------
``OPENAI_API_KEY`` must be set — Ragas uses it to compute LLM-graded metrics.
``EVAL_TOP_K`` controls how many chunks to retrieve per question (default 5).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Lazy imports — optional deps not required for the core API to run
# ---------------------------------------------------------------------------

def _require(package: str) -> Any:
    """Import *package* or abort with a helpful message."""
    import importlib
    try:
        return importlib.import_module(package)
    except ModuleNotFoundError:
        print(
            f"[ragas_eval] Missing dependency: '{package}'. "
            f"Install with:  pip install ragas datasets langchain-openai",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Query the running API
# ---------------------------------------------------------------------------

def query_api(api_url: str, question: str, top_k: int = 5) -> dict[str, Any]:
    """POST /query and return the parsed response dict."""
    resp = httpx.post(
        f"{api_url}/query",
        json={"query": question, "top_k": top_k},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Build Ragas dataset
# ---------------------------------------------------------------------------

def build_ragas_dataset(
    api_url: str,
    golden_set: list[dict[str, Any]],
    top_k: int = 5,
) -> Any:
    """Query the API for each golden question and return a Ragas-compatible Dataset."""

    datasets = _require("datasets")

    questions: list[str] = []
    answers: list[str] = []
    contexts_list: list[list[str]] = []
    ground_truths: list[str] = []

    print(f"[ragas_eval] Querying API at {api_url} for {len(golden_set)} questions…")
    for i, entry in enumerate(golden_set, 1):
        q = entry["question"]
        gt = entry["ground_truth"]
        print(f"  [{i}/{len(golden_set)}] {q[:60]}…")
        try:
            resp = query_api(api_url, q, top_k=top_k)
        except httpx.HTTPError as exc:
            print(f"    ⚠ API error: {exc} — skipping", file=sys.stderr)
            continue

        retrieved_contexts = [c["text"] for c in resp.get("chunks", [])]
        questions.append(q)
        answers.append(resp.get("answer", ""))
        contexts_list.append(retrieved_contexts if retrieved_contexts else [""])
        ground_truths.append(gt)
        time.sleep(0.2)   # be gentle on the API

    return datasets.Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    )


# ---------------------------------------------------------------------------
# Run Ragas evaluation
# ---------------------------------------------------------------------------

def run_ragas_evaluation(dataset: Any) -> dict[str, float]:
    """Run Ragas metrics and return {metric_name: score} dict."""
    ragas = _require("ragas")
    from ragas import evaluate  # type: ignore[import]
    from ragas.metrics import (  # type: ignore[import]
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    print("\n[ragas_eval] Running Ragas evaluation (this calls OpenAI)…")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )
    return dict(result)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def save_report(scores: dict[str, float], output_path: Path) -> None:
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scores": scores,
        "thresholds": {
            "faithfulness": 0.80,
            "answer_relevancy": 0.75,
            "context_precision": 0.70,
            "context_recall": 0.70,
        },
        "passed": all(
            scores.get(k, 0) >= v
            for k, v in {
                "faithfulness": 0.80,
                "answer_relevancy": 0.75,
                "context_precision": 0.70,
                "context_recall": 0.70,
            }.items()
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\n[ragas_eval] Report saved → {output_path}")

    print("\n── Ragas Scores ──────────────────────────────────")
    for metric, score in scores.items():
        threshold = report["thresholds"].get(metric, 0)
        status = "✅" if score >= threshold else "❌"
        print(f"  {status}  {metric:<25} {score:.3f}  (threshold: {threshold:.2f})")
    print("──────────────────────────────────────────────────")
    print(f"  Overall: {'PASSED ✅' if report['passed'] else 'FAILED ❌'}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Ragas evaluation against the Enterprise RAG API."
    )
    parser.add_argument(
        "--api-url", default="http://localhost:8000", help="Base URL of the RAG API"
    )
    parser.add_argument(
        "--golden-set",
        default="datasets/rag/golden_set.json",
        help="Path to golden_set.json",
    )
    parser.add_argument(
        "--output",
        default="eval/reports/ragas_report.json",
        help="Where to write the JSON report",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Chunks to retrieve per question"
    )
    args = parser.parse_args()

    golden_path = Path(args.golden_set)
    if not golden_path.exists():
        print(f"[ragas_eval] Golden set not found: {golden_path}", file=sys.stderr)
        sys.exit(1)

    golden_set: list[dict[str, Any]] = json.loads(golden_path.read_text())
    dataset = build_ragas_dataset(args.api_url, golden_set, top_k=args.top_k)

    if len(dataset) == 0:
        print("[ragas_eval] No successful queries — cannot evaluate.", file=sys.stderr)
        sys.exit(1)

    scores = run_ragas_evaluation(dataset)
    save_report(scores, Path(args.output))


if __name__ == "__main__":
    main()
