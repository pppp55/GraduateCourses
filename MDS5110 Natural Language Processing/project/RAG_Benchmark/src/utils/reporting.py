"""Reporting utilities for benchmark outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Sequence

from src.utils.data_loader import LawCorpus, LawDocument


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(payload, path: Path) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(rows: Sequence[dict], path: Path, fieldnames: Sequence[str]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_snippet(text: str, limit: int = 200) -> str:
    """Condense text to a single line snippet."""

    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _serialize_doc(doc: LawDocument | None) -> dict | None:
    if doc is None:
        return None
    return {
        "law_id": doc.doc_id,
        "law_name": doc.law_name,
        "duration": doc.duration,
        "snippet": make_snippet(doc.content, limit=260),
    }


def export_bad_cases(
    results: Iterable[dict],
    corpus: LawCorpus,
    output_path: Path,
    top_errors: int = 3,
    max_cases: int | None = None,
) -> None:
    """Create a diff-style file for the trickiest queries."""

    diff_cases: List[dict] = []
    for entry in results:
        gt_ids = {int(law_id) for law_id in entry.get("law_ids", []) if law_id is not None}
        predictions = entry.get("predictions", [])
        if not predictions:
            continue
        hit = any(int(pred.get("law_id")) in gt_ids for pred in predictions if pred.get("law_id") is not None)
        if hit:
            continue
        bench_source = entry.get("bench_source")
        law_texts = entry.get("law_texts") or []
        ground_truth_docs = [doc for doc in (_serialize_doc(corpus.get(law_id)) for law_id in gt_ids) if doc]
        if not ground_truth_docs and law_texts:
            ground_truth_docs = law_texts
        wrong_docs = []
        for pred in predictions:
            law_id = pred.get("law_id")
            if law_id is None:
                continue
            doc = _serialize_doc(corpus.get(int(law_id)))
            if not doc:
                continue
            wrong_docs.append({
                **doc,
                "score": pred.get("score"),
            })
            if len(wrong_docs) >= top_errors:
                break
        case_payload = {
            "query": entry.get("query"),
            "ground_truth": ground_truth_docs,
            "mistakes": wrong_docs,
        }
        if bench_source:
            case_payload["bench_source"] = bench_source
        if law_texts:
            case_payload["law_texts"] = law_texts
        diff_cases.append(case_payload)
        if max_cases is not None and len(diff_cases) >= max_cases:
            break
    save_json(diff_cases, output_path)
