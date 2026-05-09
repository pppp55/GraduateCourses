from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Embeddings:
    words: list[str]
    vectors: np.ndarray
    normalized: np.ndarray
    word_to_index: dict[str, int]

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def has(self, word: str) -> bool:
        return normalize_token(word) in self.word_to_index

    def index(self, word: str) -> int:
        return self.word_to_index[normalize_token(word)]

    def vector(self, word: str) -> np.ndarray:
        return self.vectors[self.index(word)]

    def normalized_vector(self, word: str) -> np.ndarray:
        return self.normalized[self.index(word)]


def normalize_token(token: str) -> str:
    return token.strip().lower()


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def load_embeddings(path: str | Path) -> Embeddings:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    words: list[str] = []
    vectors: list[list[float]] = []
    expected_dim: int | None = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        first = f.readline().lstrip("\ufeff")
        first_parts = first.strip().split()
        has_header = len(first_parts) == 2 and all(part.isdigit() for part in first_parts)
        if has_header:
            expected_dim = int(first_parts[1])
        else:
            _parse_embedding_line(first, words, vectors, expected_dim)
            if vectors:
                expected_dim = len(vectors[-1])

        for line in f:
            _parse_embedding_line(line, words, vectors, expected_dim)
            if vectors and expected_dim is None:
                expected_dim = len(vectors[-1])

    if not vectors:
        raise ValueError(f"No embedding vectors could be loaded from {path}")

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)

    word_to_index: dict[str, int] = {}
    for idx, word in enumerate(words):
        word_to_index.setdefault(normalize_token(word), idx)

    return Embeddings(words=words, vectors=matrix, normalized=normalized, word_to_index=word_to_index)


def _parse_embedding_line(
    line: str,
    words: list[str],
    vectors: list[list[float]],
    expected_dim: int | None,
) -> None:
    parts = line.strip().split()
    if len(parts) < 2:
        return

    word = parts[0]
    try:
        vector = [float(value) for value in parts[1:]]
    except ValueError:
        return

    if expected_dim is not None and len(vector) != expected_dim:
        return

    words.append(word)
    vectors.append(vector)


def cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def top_k_neighbors(
    embeddings: Embeddings,
    query: str,
    k: int = 10,
    exclude: Iterable[str] = (),
) -> list[tuple[str, float]]:
    query_norm = embeddings.normalized_vector(query)
    similarities = embeddings.normalized @ query_norm
    excluded = {normalize_token(query), *(normalize_token(word) for word in exclude)}
    for token in excluded:
        idx = embeddings.word_to_index.get(token)
        if idx is not None:
            similarities[idx] = -np.inf

    candidate_count = min(k, len(embeddings.words) - len(excluded))
    if candidate_count <= 0:
        return []

    nearest_idx = np.argpartition(-similarities, candidate_count - 1)[:candidate_count]
    nearest_idx = nearest_idx[np.argsort(-similarities[nearest_idx])]
    return [(embeddings.words[int(idx)], float(similarities[int(idx)])) for idx in nearest_idx]


def nearest_to_vector(
    embeddings: Embeddings,
    vector: np.ndarray,
    exclude: Iterable[str] = (),
) -> tuple[str, float, int]:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot search nearest word for a zero vector")

    similarities = embeddings.normalized @ (vector / norm)
    for token in {normalize_token(word) for word in exclude}:
        idx = embeddings.word_to_index.get(token)
        if idx is not None:
            similarities[idx] = -np.inf

    best_idx = int(np.argmax(similarities))
    return embeddings.words[best_idx], float(similarities[best_idx]), best_idx


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Spearman inputs must have the same length")
    if len(xs) < 2:
        return math.nan

    rank_x = _rankdata(xs)
    rank_y = _rankdata(ys)
    return cosine(rank_x - rank_x.mean(), rank_y - rank_y.mean())


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=np.float64)

    start = 0
    while start < len(values_array):
        end = start + 1
        while end < len(values_array) and values_array[order[end]] == values_array[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def deterministic_sample(items: Sequence, count: int, seed: int) -> list:
    if count >= len(items):
        return list(items)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(items)), count))
    return [items[idx] for idx in indices]


def write_csv(path: str | Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_seed(cli_student_id: str | None) -> int:
    raw = cli_student_id or "0"
    digits = "".join(ch for ch in raw if ch.isdigit())
    if digits:
        return int(digits) % (2**32)
    return sum(ord(ch) for ch in raw) % (2**32)


def model_name_from_path(path: str | Path) -> str:
    return Path(path).stem
