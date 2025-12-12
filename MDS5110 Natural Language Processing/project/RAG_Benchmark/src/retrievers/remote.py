"""HTTP-based retriever that proxies queries to an external RAG service."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import requests

from src.utils.data_loader import LawCorpus, LawDocument

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """Normalized retrieval output consumed by the benchmark."""

    law_id: int
    law_name: str
    score: float
    content: str


class RemoteRetriever:
    """Wrap an existing HTTP retriever (same schema as rag_request_for_bench)."""

    def __init__(
        self,
        corpus: LawCorpus,
        endpoint: str,
        timeout: float = 10.0,
        proxies: Optional[dict] = None,
    ) -> None:
        if not endpoint:
            raise ValueError("RemoteRetriever requires a non-empty endpoint URL")
        self.corpus = corpus
        self.endpoint = endpoint
        self.timeout = timeout
        self.proxies = proxies if proxies is not None else {"http": None, "https": None}

    def _call_service(self, query: str, top_k: int) -> List[dict]:
        payload = {"queries": [query], "topk": top_k, "return_scores": True}
        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
                proxies=self.proxies,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            logger.warning("Retriever timeout for query: %s", query[:50])
            return []
        except requests.exceptions.RequestException as exc:
            logger.error("Retriever request failed: %s", exc)
            return []
        except ValueError as exc:
            logger.error("Failed to decode retriever JSON: %s", exc)
            return []

        results = data.get("result", [])
        if not results:
            return []
        return results[0] or []

    def _materialize_doc(self, law_id: int) -> tuple[str, str]:
        doc: Optional[LawDocument] = self.corpus.get(law_id)
        if not doc:
            return (f"法条 {law_id}", "")
        return (doc.law_name, doc.content)

    def search(self, query: str, top_k: int = 10) -> List[RetrievedDocument]:
        if not query.strip():
            return []
        raw_results = self._call_service(query, top_k)
        retrieved: List[RetrievedDocument] = []
        for entry in raw_results:
            document = entry.get("document", {}) if isinstance(entry, dict) else {}
            doc_id = document.get("id") or document.get("doc_id")
            if doc_id is None:
                continue
            try:
                law_id = int(doc_id)
            except (TypeError, ValueError):
                continue
            law_name, content = self._materialize_doc(law_id)
            score = float(entry.get("score", 0.0)) if isinstance(entry, dict) else 0.0
            retrieved.append(
                RetrievedDocument(
                    law_id=law_id,
                    law_name=law_name,
                    score=score,
                    content=content,
                )
            )
            if len(retrieved) >= top_k:
                break
        return retrieved
