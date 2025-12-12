#!/usr/bin/env python3
"""Utility script to inspect benchmark outputs and diff troublesome queries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import load_law_corpus
from src.utils.reporting import make_snippet


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LegalRAG benchmark outputs")
    parser.add_argument("--predictions", default="outputs/reports/predictions.json", help="Predictions JSON path")
    parser.add_argument("--metrics", default="outputs/reports/metrics.json", help="Aggregated metrics JSON path")
    parser.add_argument("--diff", default="outputs/bad_cases/diff_cases.json", help="Bad-case diff file")
    parser.add_argument("--law-corpus", default="data/法律法规.jsonl", help="Law corpus JSONL path")
    parser.add_argument("--limit", type=int, default=5, help="Number of diff cases to display")
    return parser.parse_args()


def _build_diff_cases(predictions: List[dict], corpus_path: Path, limit: int) -> List[dict]:
    corpus = load_law_corpus(corpus_path)
    cases: List[dict] = []
    for entry in predictions:
        gt_ids = {int(doc_id) for doc_id in entry.get("law_ids", []) if doc_id is not None}
        preds = entry.get("predictions", [])
        if not gt_ids or not preds:
            continue
        if any(int(pred.get("law_id")) in gt_ids for pred in preds if pred.get("law_id") is not None):
            continue
        ground_truth_docs = []
        for law_id in gt_ids:
            doc = corpus.get(law_id)
            if not doc:
                continue
            ground_truth_docs.append(
                {
                    "law_id": doc.doc_id,
                    "law_name": doc.law_name,
                    "snippet": make_snippet(doc.content),
                }
            )
        mistakes = []
        for pred in preds:
            law_id = pred.get("law_id")
            if law_id is None:
                continue
            doc = corpus.get(int(law_id))
            if not doc:
                continue
            mistakes.append(
                {
                    "law_id": doc.doc_id,
                    "law_name": doc.law_name,
                    "snippet": make_snippet(doc.content),
                    "score": pred.get("score"),
                }
            )
            if len(mistakes) >= 3:
                break
        cases.append(
            {
                "query": entry.get("query"),
                "ground_truth": ground_truth_docs,
                "mistakes": mistakes,
                "bench_source": entry.get("bench_source"),
                "law_texts": entry.get("law_texts"),
            }
        )
        if len(cases) >= limit:
            break
    return cases


def main() -> None:
    args = parse_args()
    predictions_path = PROJECT_ROOT / args.predictions
    metrics_path = PROJECT_ROOT / args.metrics
    diff_path = PROJECT_ROOT / args.diff
    corpus_path = PROJECT_ROOT / args.law_corpus

    predictions = _load_json(predictions_path)
    metrics = _load_json(metrics_path) if metrics_path.exists() else {}

    if diff_path.exists():
        diff_cases = _load_json(diff_path)[: args.limit]
    else:
        diff_cases = _build_diff_cases(predictions, corpus_path, args.limit)

    if metrics:
        print("===== Aggregate Metrics =====")
        for key in ("ndcg", "recall", "mrr", "hit_rate", "total_queries"):
            value = metrics.get(key)
            if value is None:
                continue
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        metadata = metrics.get("metadata") or {}
        duration_value = metadata.get("evaluation_duration_seconds")
        if duration_value is not None:
            print(f"evaluation_duration_seconds: {duration_value:.2f}")
        if metadata.get("retriever_type"):
            print(f"retriever_type: {metadata.get('retriever_type')}")
        if metadata.get("endpoint"):
            print(f"endpoint: {metadata.get('endpoint')}")
        per_source = metrics.get("per_source") or {}
        if per_source:
            print("\n===== Per-Source Metrics =====")
            for source, stats in per_source.items():
                print(f"[{source}]")
                for key in ("ndcg", "recall", "mrr", "hit_rate", "total_queries"):
                    value = stats.get(key)
                    if value is None:
                        continue
                    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        print()

    if not diff_cases:
        print("No failing cases to display.")
        return

    print("===== Diff Viewer =====")
    for idx, case in enumerate(diff_cases, 1):
        print(f"\nCase {idx}: Query -> {case.get('query')}")
        bench_source = case.get("bench_source")
        if bench_source:
            if isinstance(bench_source, dict):
                label = bench_source.get("source") or "unspecified"
            else:
                label = bench_source
            print(f"  Bench Source: {label}")
        print("  Ground Truth:")
        for doc in case.get("ground_truth", []):
            print(f"    - [{doc.get('law_id')}] {doc.get('law_name')}")
            print(f"      {doc.get('snippet')}")
        print("  Wrong Retrievals:")
        mistakes = case.get("mistakes", [])
        if not mistakes:
            print("    (no predictions available)")
            continue
        for doc in mistakes:
            print(f"    - [{doc.get('law_id')}] {doc.get('law_name')} (score={doc.get('score')})")
            print(f"      {doc.get('snippet')}")


if __name__ == "__main__":
    main()
