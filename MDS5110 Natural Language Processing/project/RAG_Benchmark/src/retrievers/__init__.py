"""Retriever factory for LegalRAG-Bench."""

from __future__ import annotations

from typing import Any, Dict

from src.retrievers.lexical import LexicalRetriever
from src.retrievers.remote import RemoteRetriever
from src.utils.data_loader import LawCorpus


def build_retriever(name: str, corpus: LawCorpus, params: Dict[str, Any] | None = None):
    normalized = name.lower()
    params = params or {}

    if normalized == "lexical":
        return LexicalRetriever(corpus)
    if normalized == "remote":
        endpoint = params.get("endpoint")
        timeout = params.get("timeout", 10.0)
        proxies = params.get("proxies")
        return RemoteRetriever(corpus, endpoint=endpoint, timeout=timeout, proxies=proxies)
    raise ValueError(f"Unknown retriever type: {name}")
