"""Simple lexical retriever baseline using TF-IDF style scoring."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List

from src.utils.data_loader import LawCorpus, LawDocument

_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    tokens = [token.lower() for token in _TOKEN_PATTERN.findall(text)]
    cjk_chars = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
    return tokens + cjk_chars


@dataclass
class RetrievedDocument:
    law_id: int
    law_name: str
    score: float
    content: str


class LexicalRetriever:
    """Naive TF-IDF retriever that runs fully offline for benchmarking."""

    def __init__(self, corpus: LawCorpus):
        self.corpus = corpus
        self._index = []
        self._build_index()

    def _build_index(self) -> None:
        documents = self.corpus.documents
        doc_count = len(documents)
        if doc_count == 0:
            return
        df_counter: Counter[str] = Counter()
        counted_docs = []
        for doc in documents:
            tokens = _tokenize(doc.content)
            counts = Counter(tokens)
            counted_docs.append((doc, counts))
            df_counter.update(counts.keys())
        idf = {term: math.log((doc_count + 1) / (freq + 1)) + 1.0 for term, freq in df_counter.items()}
        for doc, counts in counted_docs:
            vector = {}
            norm = 0.0
            for term, freq in counts.items():
                tf = 1.0 + math.log(freq)
                weight = tf * idf.get(term, 1.0)
                vector[term] = weight
                norm += weight * weight
            norm = math.sqrt(norm) if norm > 0 else 1.0
            self._index.append(
                {
                    "doc": doc,
                    "vector": vector,
                    "norm": norm,
                }
            )

    def _vectorize_query(self, query: str) -> tuple[dict, float]:
        tokens = _tokenize(query)
        counts = Counter(tokens)
        vector = {}
        norm = 0.0
        for term, freq in counts.items():
            tf = 1.0 + math.log(freq)
            vector[term] = tf
            norm += tf * tf
        return vector, math.sqrt(norm) if norm > 0 else 1.0

    def search(self, query: str, top_k: int = 10) -> List[RetrievedDocument]:
        if not query.strip():
            return []
        query_vec, query_norm = self._vectorize_query(query)
        scores: List[RetrievedDocument] = []
        for node in self._index:
            doc_vec = node["vector"]
            score = 0.0
            for term, weight in query_vec.items():
                doc_weight = doc_vec.get(term)
                if doc_weight is None:
                    continue
                score += weight * doc_weight
            if score == 0.0:
                continue
            score /= (query_norm * node["norm"])
            doc: LawDocument = node["doc"]
            scores.append(
                RetrievedDocument(
                    law_id=doc.doc_id,
                    law_name=doc.law_name,
                    score=score,
                    content=doc.content,
                )
            )
        scores.sort(key=lambda item: item.score, reverse=True)
        return scores[:top_k]
