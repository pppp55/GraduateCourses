"""Data loading helpers for LegalRAG-Bench."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class QueryExample:
    """Container for a single benchmark query along with its labels."""

    query: str
    law_ids: List[int]
    source: Optional[str] = None
    detailed_source: Optional[str] = None
    law_contents: Optional[List[Dict[str, Any]]] = None


@dataclass
class LawDocument:
    """Law corpus entry used for retrieval and inspection."""

    doc_id: int
    law_name: str
    content: str
    duration: Optional[str] = None


class LawCorpus:
    """In-memory representation of the statutory corpus."""

    def __init__(self, documents: Sequence[LawDocument]):
        self._documents: List[LawDocument] = list(documents)
        self._by_id: Dict[int, LawDocument] = {doc.doc_id: doc for doc in self._documents}

    def __len__(self) -> int:  # pragma: no cover - simple property
        return len(self._documents)

    @property
    def documents(self) -> List[LawDocument]:
        return self._documents

    def get(self, doc_id: int) -> Optional[LawDocument]:
        return self._by_id.get(doc_id)


def _normalize_law_ids(raw_ids: Iterable[int | str | None]) -> List[int]:
    normalized: List[int] = []
    for value in raw_ids:
        if value is None:
            continue
        try:
            normalized.append(int(value))
        except (TypeError, ValueError):
            continue
    return normalized


def load_queries(path: Path, limit: Optional[int] = None) -> List[QueryExample]:
    """Load benchmark queries from JSON produced by the alignment script."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Query file must contain a JSON array")

    examples: List[QueryExample] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        query = (item.get("query") or item.get("question") or "").strip()
        if not query:
            continue
        law_ids = _normalize_law_ids(item.get("law_ids", []))
        source = item.get("source")
        detailed_source = item.get("detailed_source") or item.get("description")
        law_contents_raw = item.get("law_contents")
        law_contents = law_contents_raw if isinstance(law_contents_raw, list) else None
        examples.append(
            QueryExample(
                query=query,
                law_ids=law_ids,
                source=str(source).strip() if source else None,
                detailed_source=str(detailed_source).strip() if detailed_source else None,
                law_contents=law_contents,
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


def load_law_corpus(path: Path, limit: Optional[int] = None) -> LawCorpus:
    """Load the law corpus from a JSONL file where each line is a document."""

    documents: List[LawDocument] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            doc_id = payload.get("id")
            if doc_id is None:
                continue
            try:
                law_id = int(doc_id)
            except (TypeError, ValueError):
                continue
            doc = LawDocument(
                doc_id=law_id,
                law_name=str(payload.get("law_name", "")).strip() or "未知法条",
                content=str(payload.get("content", "")),
                duration=payload.get("law_duration"),
            )
            documents.append(doc)
            if limit is not None and len(documents) >= limit:
                break
    if not documents:
        raise ValueError(f"No documents parsed from {path}")
    return LawCorpus(documents)
