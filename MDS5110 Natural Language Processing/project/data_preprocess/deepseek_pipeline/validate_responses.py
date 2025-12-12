"""Validation utilities for sample_responses.json outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def safe_int(value: Any, default: int | None = 0) -> int | None:
    """Convert arbitrary inputs to int with fallback."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_responses(
    responses: List[Dict[str, object]],
    law_ids: Dict[int, Dict[str, object]],
) -> Tuple[List[str], Dict[str, int]]:
    issues: List[str] = []
    stats = {
        "total": len(responses),
        "law_required": 0,
        "law_optional": 0,
        "law_ids_matched": 0,
        "law_ids_missing": 0,
        "unknown_law_ids": 0,
    }

    for idx, item in enumerate(responses, start=1):
        question_id = item.get("question_id", f"<unknown-{idx}>")
        raw_flag = item.get("law_appearance")
        law_flag = safe_int(raw_flag, 0)
        legal_text = item.get("legal_text") or []
        if isinstance(legal_text, str):
            legal_text = [legal_text] if legal_text else []

        if law_flag:
            stats["law_required"] += 1
            if not legal_text:
                issues.append(f"{question_id}: law_appearance=1 but legal_text empty")
        else:
            stats["law_optional"] += 1
            if legal_text:
                issues.append(f"{question_id}: law_appearance=0 but legal_text not empty")

        raw_law_ids = item.get("law_ids")
        if isinstance(raw_law_ids, list):
            law_ids_list = raw_law_ids
        elif raw_law_ids is None:
            law_ids_list = []
        else:
            law_ids_list = [raw_law_ids]
        if law_flag and not law_ids_list:
            stats["law_ids_missing"] += 1
            issues.append(f"{question_id}: law_appearance=1 but law_ids missing")
        elif law_flag:
            stats["law_ids_matched"] += 1

        for lid in law_ids_list:
            lid_int = safe_int(lid, default=None)
            if lid_int is None:
                issues.append(f"{question_id}: law_id {lid} is not an integer")
                continue
            if lid_int not in law_ids:
                stats["unknown_law_ids"] += 1
                issues.append(f"{question_id}: law_id {lid_int} not in law database")
    return issues, stats


def index_laws(law_records: List[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    indexed: Dict[int, Dict[str, object]] = {}
    for record in law_records:
        law_id = record.get("id")
        lid_int = safe_int(law_id, default=None)
        if lid_int is None:
            continue
        indexed[lid_int] = record
    return indexed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DeepSeek sample responses against law database")
    parser.add_argument(
        "--responses",
        default="deepseek_pipeline/sample_responses.json",
        help="Path to sample_responses.json",
    )
    parser.add_argument(
        "--laws",
        default="法律法规.jsonl",
        help="Path to 法律法规.jsonl",
    )
    parser.add_argument(
        "--max_issues",
        type=int,
        default=50,
        help="Maximum number of issues to print (default 50)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    responses = load_json(Path(args.responses))
    law_records = [json.loads(line) for line in Path(args.laws).read_text(encoding="utf-8").splitlines() if line.strip()]
    law_index = index_laws(law_records)

    issues, stats = validate_responses(responses, law_index)

    print("Validation summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if not issues:
        print("No issues detected. ✅")
        return

    print(f"\nFirst {min(len(issues), args.max_issues)} issues:")
    for issue in issues[: args.max_issues]:
        print(f"  - {issue}")
    if len(issues) > args.max_issues:
        print(f"  ... ({len(issues) - args.max_issues} more)" )


if __name__ == "__main__":
    main()
