"""Map legal_text citations to law IDs from 法律法规.jsonl."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

LAW_CITATION_PATTERN = re.compile(
    r"(?:《([^》]+)》)?第([零〇○一二三四五六七八九十百千万亿0-9]+)条(之[零〇○一二三四五六七八九十百千万0-9]+)?"
)
ARTICLE_PATTERN = re.compile(
    r"第([零〇○一二三四五六七八九十百千万亿0-9]+)条(之[零〇○一二三四五六七八九十百千万0-9]+)?"
)
CHAPTER_PATTERN = re.compile(r"(第[零〇○一二三四五六七八九十百千万亿0-9]+章)\s*$")

CHINESE_DIGITS: Dict[str, int] = {
    "零": 0,
    "〇": 0,
    "○": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
UNIT_MAP: Dict[str, int] = {"十": 10, "百": 100, "千": 1000}
SECTION_UNIT_MAP: Dict[str, int] = {"万": 10_000, "亿": 100_000_000}


@dataclass(frozen=True)
class LawEntry:
    id: int
    law_name: str
    normalized_name: str
    article_number: Optional[int]
    suffix_number: Optional[int]


@dataclass(frozen=True)
class Citation:
    law_name: str
    article_number: Optional[int]
    suffix_number: Optional[int]


def normalize_name(text: str) -> str:
    cleaned = re.sub(r"[\s《》“”\"'·﹑、（）()\\-]", "", text)
    return cleaned


def parse_numeral(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    token = text.strip()
    if not token:
        return None
    token = token.replace("第", "").replace("条", "")
    if token.isdigit():
        return int(token)
    return chinese_to_int(token)


def chinese_to_int(text: str) -> Optional[int]:
    if not text:
        return None
    total = 0
    section = 0
    number = 0
    for char in text:
        if char in CHINESE_DIGITS:
            number = CHINESE_DIGITS[char]
        elif char in UNIT_MAP:
            unit_value = UNIT_MAP[char]
            if number == 0:
                number = 1
            section += number * unit_value
            number = 0
        elif char in SECTION_UNIT_MAP:
            section += number
            if section == 0:
                section = 1
            total += section * SECTION_UNIT_MAP[char]
            section = 0
            number = 0
        else:
            number = 0
    return total + section + number


def parse_suffix(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    value = text.lstrip("之")
    return parse_numeral(value)


def extract_article_meta(content: str) -> Tuple[Optional[int], Optional[int]]:
    match = ARTICLE_PATTERN.search(content)
    if not match:
        return None, None
    article = parse_numeral(match.group(1))
    suffix = parse_suffix(match.group(2))
    return article, suffix


def load_law_entries(law_path: Path) -> List[LawEntry]:
    entries: List[LawEntry] = []
    with law_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            article, suffix = extract_article_meta(data.get("content", ""))
            entries.append(
                LawEntry(
                    id=int(data["id"]),
                    law_name=data["law_name"],
                    normalized_name=normalize_name(data["law_name"]),
                    article_number=article,
                    suffix_number=suffix,
                )
            )
    return entries


def extract_citations(text: str) -> List[Citation]:
    citations: List[Citation] = []
    if not text:
        return citations
    last_law: Optional[str] = None
    for match in LAW_CITATION_PATTERN.finditer(text):
        law = match.group(1) or last_law
        if not law:
            continue
        law = law.strip()
        last_law = law
        article_num = parse_numeral(match.group(2))
        suffix_num = parse_suffix(match.group(3))
        citations.append(Citation(law_name=law, article_number=article_num, suffix_number=suffix_num))
    return citations


def parse_structured_citation(text: str) -> Optional[Citation]:
    if not text:
        return None
    text = text.strip()
    if not text:
        return None
    article_match = ARTICLE_PATTERN.search(text)
    if not article_match:
        return None
    article_num = parse_numeral(article_match.group(1))
    suffix_num = parse_suffix(article_match.group(2))
    prefix = text[: article_match.start()].rstrip()
    if not prefix:
        return None
    chapter_match = CHAPTER_PATTERN.search(prefix)
    if chapter_match:
        law_name = prefix[: chapter_match.start()].strip()
    else:
        law_name = prefix
    if not law_name:
        return None
    return Citation(law_name=law_name, article_number=article_num, suffix_number=suffix_num)


def find_candidates(law_fragment: str, entries: Sequence[LawEntry]) -> List[LawEntry]:
    fragment_norm = normalize_name(law_fragment)
    if not fragment_norm:
        return [entry for entry in entries if law_fragment in entry.law_name]

    def matches(norm_value: str) -> bool:
        return fragment_norm in norm_value or norm_value in fragment_norm

    direct = [entry for entry in entries if matches(entry.normalized_name)]
    if direct:
        return direct

    fragment_no_digits = re.sub(r"\d+", "", fragment_norm)
    if fragment_no_digits:
        direct = [
            entry
            for entry in entries
            if fragment_no_digits in re.sub(r"\d+", "", entry.normalized_name)
            or re.sub(r"\d+", "", entry.normalized_name) in fragment_no_digits
        ]
        if direct:
            return direct

    return [entry for entry in entries if law_fragment in entry.law_name]


def filter_by_article(candidates: Sequence[LawEntry], article_number: Optional[int], suffix_number: Optional[int]) -> List[LawEntry]:
    if article_number is None:
        return list(candidates)
    exact = [entry for entry in candidates if entry.article_number == article_number]
    if not exact:
        return []
    if suffix_number is None:
        without_suffix = [entry for entry in exact if entry.suffix_number is None]
        return without_suffix or exact
    with_suffix = [entry for entry in exact if entry.suffix_number == suffix_number]
    return with_suffix or exact


def enrich_responses(responses: List[Dict[str, object]], entries: Sequence[LawEntry]) -> Tuple[int, int]:
    total_citations = 0
    matched_citations = 0
    for record in responses:
        citations = collect_citations(record.get("legal_text"))
        total_citations += len(citations)
        seen: set[int] = set()
        law_ids: List[int] = []
        for citation in citations:
            candidates = find_candidates(citation.law_name, entries)
            if not candidates:
                continue
            matched = filter_by_article(candidates, citation.article_number, citation.suffix_number)
            if not matched:
                continue
            matched_citations += 1
            for entry in matched:
                if entry.id not in seen:
                    seen.add(entry.id)
                    law_ids.append(entry.id)
        record["law_ids"] = law_ids
    return total_citations, matched_citations


def collect_citations(legal_text_field: object) -> List[Citation]:
    if isinstance(legal_text_field, list):
        citations: List[Citation] = []
        for item in legal_text_field:
            citation = parse_structured_citation(str(item))
            if citation:
                citations.append(citation)
        if citations:
            return citations
        merged = " ".join(str(item) for item in legal_text_field if str(item).strip())
        return extract_citations(merged)
    text = str(legal_text_field or "")
    return extract_citations(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach law IDs to sample responses based on legal_text citations")
    parser.add_argument(
        "--responses",
        default="deepseek_pipeline/sample_responses.json",
        help="Path to sample responses JSON file",
    )
    parser.add_argument(
        "--laws",
        default="法律法规.jsonl",
        help="Path to law JSONL source file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path (defaults to overwriting --responses)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    responses_path = Path(args.responses)
    law_path = Path(args.laws)
    output_path = Path(args.output) if args.output else responses_path

    with responses_path.open("r", encoding="utf-8") as handle:
        responses = json.load(handle)

    law_entries = load_law_entries(law_path)
    total, matched = enrich_responses(responses, law_entries)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(responses, handle, ensure_ascii=False, indent=2)

    print(f"[INFO] Annotated {matched}/{total} cited articles; updated file saved to {output_path}")


if __name__ == "__main__":
    main()
