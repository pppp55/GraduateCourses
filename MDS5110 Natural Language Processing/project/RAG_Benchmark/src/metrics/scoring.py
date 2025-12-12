"""Metric computation helpers (NDCG, Recall, MRR)."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence


def _dcg(relevances: Sequence[int]) -> float:
    score = 0.0
    for idx, rel in enumerate(relevances):
        if rel <= 0:
            continue
        score += rel / math.log2(idx + 2)
    return score


def compute_ndcg(ground_truth: Sequence[int], predictions: Sequence[int], k: int | None = None) -> float | None:
    """Compute binary NDCG for a single query."""

    gt_set = {int(doc_id) for doc_id in ground_truth if doc_id is not None}
    if not gt_set:
        return None
    if k is None:
        k = len(predictions)
    truncated = list(predictions)[:k]
    relevances = [1 if int(doc_id) in gt_set else 0 for doc_id in truncated]
    ideal = sorted(relevances, reverse=True)
    # If all predictions are zero relevance, ideal DCG should still reflect optimal scenario.
    if sum(relevances) < len(gt_set):
        ideal = [1] * min(len(gt_set), k)
    actual_dcg = _dcg(relevances)
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def compute_recall(ground_truth: Sequence[int], predictions: Sequence[int], k: int | None = None) -> float | None:
    gt_set = {int(doc_id) for doc_id in ground_truth if doc_id is not None}
    if not gt_set:
        return None
    truncated = set(int(doc_id) for doc_id in (predictions[:k] if k is not None else predictions))
    return len(gt_set & truncated) / len(gt_set)


def compute_mrr(ground_truth: Sequence[int], predictions: Sequence[int], k: int | None = None) -> float | None:
    gt_set = {int(doc_id) for doc_id in ground_truth if doc_id is not None}
    if not gt_set:
        return None
    search_space = predictions if k is None else predictions[:k]
    for idx, doc_id in enumerate(search_space):
        if int(doc_id) in gt_set:
            return 1.0 / (idx + 1)
    return 0.0


def evaluate_query(ground_truth: Sequence[int], predictions: Sequence[int], k: int | None = None) -> dict:
    """Return a dictionary of per-query metrics."""

    metrics = {
        "ndcg": compute_ndcg(ground_truth, predictions, k=k),
        "recall": compute_recall(ground_truth, predictions, k=k),
        "mrr": compute_mrr(ground_truth, predictions, k=k),
    }
    return metrics


def aggregate_metrics(per_query: Iterable[dict]) -> dict:
    """Average metrics while skipping queries without ground-truth labels."""

    totals = {"ndcg": 0.0, "recall": 0.0, "mrr": 0.0}
    counts = {"ndcg": 0, "recall": 0, "mrr": 0}
    num_queries = 0
    hit_count = 0

    for entry in per_query:
        num_queries += 1
        recall_value = entry.get("recall")
        if recall_value and recall_value > 0:
            hit_count += 1
        for key in ("ndcg", "recall", "mrr"):
            value = entry.get(key)
            if value is None:
                continue
            totals[key] += value
            counts[key] += 1

    averaged = {
        key: (totals[key] / counts[key] if counts[key] else None)
        for key in totals
    }
    averaged["evaluated_queries"] = max(counts.values(), default=0)
    averaged["total_queries"] = num_queries
    averaged["hit_rate"] = hit_count / num_queries if num_queries else None
    return averaged
