#!/usr/bin/env python3
"""Filter questions whose law_ids agree across multiple response files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_responses(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    return {str(item.get("question_id")): item for item in items if item.get("question_id")}


def collect_aligned(
    datasets: List[Dict[str, dict]],
) -> List[dict]:
    common_ids = set.intersection(*(set(data.keys()) for data in datasets))
    aligned: List[dict] = []
    for qid in sorted(common_ids):
        entries = [data[qid] for data in datasets]
        law_id_lists = [entry.get("law_ids") or [] for entry in entries]
        if not law_id_lists[0]:
            continue
        if any(law_ids != law_id_lists[0] for law_ids in law_id_lists[1:]):
            continue
        aligned.append(
            {
                "question_id": qid,
                "question": entries[0].get("question", ""),
                "law_ids": law_id_lists[0],
            }
        )
    return aligned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep entries whose law_ids align across response files")
    parser.add_argument("--responses", nargs=4, type=Path, help="Paths to sample_responses_*.json files")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deepseek_pipeline/aligned_law_ids.json"),
        help="Where to store filtered results (default: deepseek_pipeline/aligned_law_ids.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [load_responses(path) for path in args.responses]
    aligned = collect_aligned(datasets)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)
    print(f"Kept {len(aligned)} aligned questions -> {args.output}")


if __name__ == "__main__":
    main()
