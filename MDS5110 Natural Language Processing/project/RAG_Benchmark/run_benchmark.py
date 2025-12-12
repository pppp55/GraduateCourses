#!/usr/bin/env python3
"""Entry point for running the LegalRAG benchmark end-to-end."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
import time

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.scoring import aggregate_metrics, evaluate_query
from src.retrievers import build_retriever
from src.utils import data_loader
from src.utils.reporting import export_bad_cases, make_snippet, save_csv, save_json


def _ensure_absolute(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def _load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LegalRAG benchmark runner")
    parser.add_argument("--config", default="configs/default.json", help="Path to config JSON")
    parser.add_argument("--top_k", type=int, default=None, help="Override retriever top-k")
    parser.add_argument("--max_queries", type=int, default=None, help="Limit number of benchmark queries")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    config = _load_config(config_path)

    run_name = config.get("run_name") or config.get("experiment_name") or config.get("outputs", {}).get("run_name") or config_path.stem

    outputs_cfg = config.get("outputs", {})
    root_base = outputs_cfg.get("root", "outputs")
    root_dir = _ensure_absolute(root_base)
    run_dir = root_dir / run_name

    def _resolve_path(key: str, default_rel: str) -> Path:
        configured = outputs_cfg.get(key)
        if configured:
            path = _ensure_absolute(configured)
        else:
            path = run_dir / default_rel
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    log_path = _resolve_path("log_path", "logs/benchmark.log")
    _setup_logging(log_path)

    data_cfg = config.get("data", {})
    queries_path = PROJECT_ROOT / data_cfg.get("queries_path", "data/query_law_ids_validated.json")
    law_corpus_path = PROJECT_ROOT / data_cfg.get("law_corpus_path", "data/法律法规.jsonl")
    max_queries = args.max_queries if args.max_queries is not None else data_cfg.get("max_queries")

    logging.info("Loading queries from %s", queries_path)
    queries = data_loader.load_queries(queries_path, limit=max_queries)
    logging.info("Loaded %d queries", len(queries))

    logging.info("Loading law corpus from %s", law_corpus_path)
    corpus = data_loader.load_law_corpus(law_corpus_path)
    logging.info("Loaded %d law documents", len(corpus))

    retriever_cfg = config.get("retriever", {})
    retriever_type = retriever_cfg.get("type", "lexical")
    top_k = args.top_k if args.top_k is not None else retriever_cfg.get("top_k", 10)
    logging.info("Initializing '%s' retriever with top_k=%d", retriever_type, top_k)
    retriever = build_retriever(retriever_type, corpus, params=retriever_cfg)

    per_query_metrics: List[dict] = []
    per_source_metrics: Dict[str, List[dict]] = defaultdict(list)
    results: List[dict] = []

    bench_start = time.perf_counter()

    for idx, example in enumerate(queries, 1):
        retrieved = retriever.search(example.query, top_k=top_k)
        predicted_ids = [doc.law_id for doc in retrieved]
        metrics = evaluate_query(example.law_ids, predicted_ids, k=top_k)
        per_query_metrics.append(metrics)
        source_key = example.source or "unspecified"
        per_source_metrics[source_key].append(metrics)
        predictions_payload = [
            {
                "law_id": doc.law_id,
                "law_name": doc.law_name,
                "score": round(doc.score, 6),
                "snippet": make_snippet(doc.content),
            }
            for doc in retrieved
        ]
        bench_info = {
            "source": example.source,
            "detailed_source": example.detailed_source,
        }
        ground_truth_laws = example.law_contents if example.law_contents is not None else []
        results.append(
            {
                "query": example.query,
                "law_ids": example.law_ids,
                "law_texts": ground_truth_laws,
                "bench_source": bench_info,
                "predictions": predictions_payload,
                "metrics": metrics,
            }
        )
        if idx % 25 == 0:
            logging.info("Processed %d/%d queries", idx, len(queries))

    aggregated = aggregate_metrics(per_query_metrics)
    per_source_scores = {
        source: aggregate_metrics(entries)
        for source, entries in per_source_metrics.items()
    }
    logging.info("Benchmark finished: NDCG=%.4f Recall=%.4f MRR=%.4f", aggregated.get("ndcg") or 0.0, aggregated.get("recall") or 0.0, aggregated.get("mrr") or 0.0)

    duration_seconds = time.perf_counter() - bench_start

    metadata = {
        "config": str(args.config),
        "run_name": run_name,
        "retriever_type": retriever_type,
        "top_k": top_k,
        "query_file": str(queries_path),
        "evaluation_duration_seconds": duration_seconds,
    }
    if retriever_cfg.get("endpoint"):
        metadata["endpoint"] = retriever_cfg.get("endpoint")
    user_metadata = config.get("metadata", {})
    metadata.update(user_metadata)
    aggregated["metadata"] = metadata
    aggregated["per_source"] = per_source_scores

    predictions_path = _resolve_path("predictions_path", "reports/predictions.json")
    metrics_json_path = _resolve_path("metrics_json", "reports/metrics.json")
    metrics_csv_path = _resolve_path("metrics_csv", "reports/metrics.csv")
    per_source_csv_path = _resolve_path("per_source_csv", "reports/per_source_metrics.csv")
    bad_cases_path = _resolve_path("bad_cases_path", "bad_cases/diff_cases.json")

    save_json(results, predictions_path)
    save_json(aggregated, metrics_json_path)
    save_csv(
        [
            {"metric": "ndcg", "value": aggregated.get("ndcg")},
            {"metric": "recall", "value": aggregated.get("recall")},
            {"metric": "mrr", "value": aggregated.get("mrr")},
            {"metric": "hit_rate", "value": aggregated.get("hit_rate")},
            {"metric": "total_queries", "value": aggregated.get("total_queries")},
            {"metric": "evaluation_duration_seconds", "value": round(duration_seconds, 4)},
        ],
        metrics_csv_path,
        fieldnames=["metric", "value"],
    )
    if per_source_scores:
        per_source_rows = []
        for source, scores in per_source_scores.items():
            per_source_rows.append(
                {
                    "source": source,
                    "ndcg": scores.get("ndcg"),
                    "recall": scores.get("recall"),
                    "mrr": scores.get("mrr"),
                    "hit_rate": scores.get("hit_rate"),
                    "total_queries": scores.get("total_queries"),
                }
            )
        save_csv(
            per_source_rows,
            per_source_csv_path,
            fieldnames=["source", "ndcg", "recall", "mrr", "hit_rate", "total_queries"],
        )
    export_bad_cases(results, corpus, bad_cases_path, top_errors=3, max_cases=200)
    logging.info("Artifacts saved under %s", run_dir)


if __name__ == "__main__":
    main()
