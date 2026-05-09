from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from embedding_utils import (
    deterministic_sample,
    load_embeddings,
    model_name_from_path,
    project_path,
    resolve_seed,
    top_k_neighbors,
    write_csv,
)


WORD_LIST = [
    "july", "reliable", "play", "willing", "good", "very", "patient", "concerned",
    "important", "powerful", "quickly", "generally", "gradually", "happy", "able",
    "close", "near", "saturday", "friend", "company", "road", "plane", "war",
    "politics", "building", "student", "university", "realm", "china", "experience",
    "police", "give", "create", "tell", "become", "lack", "win", "help", "gain",
    "get", "take", "use", "set", "find", "increase", "difficult", "go", "man",
    "ten", "year",
]


def evaluate_model(model_path: Path, k: int, detail_words: list[str], out_dir: Path) -> dict:
    model_name = model_name_from_path(model_path)
    embeddings = load_embeddings(model_path)

    detail_rows: list[dict] = []
    knn_rows: list[dict] = []
    summary_rows: list[dict] = []
    query_averages: list[float] = []
    covered = 0

    for query in WORD_LIST:
        if not embeddings.has(query):
            summary_rows.append({
                "model": model_name,
                "query": query,
                "covered": 0,
                "average_similarity": "",
            })
            continue

        neighbors = top_k_neighbors(embeddings, query, k=k)
        covered += 1
        avg_similarity = float(np.mean([sim for _, sim in neighbors])) if neighbors else float("nan")
        query_averages.append(avg_similarity)

        summary_rows.append({
            "model": model_name,
            "query": query,
            "covered": 1,
            "average_similarity": f"{avg_similarity:.6f}",
        })

        for rank, (neighbor, similarity) in enumerate(neighbors, start=1):
            row = {
                "model": model_name,
                "query": query,
                "rank": rank,
                "neighbor": neighbor,
                "cosine_similarity": f"{similarity:.6f}",
            }
            knn_rows.append(row)
            if query in detail_words:
                detail_rows.append(row)

    overall_average = float(np.mean(query_averages)) if query_averages else float("nan")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / f"{model_name}_knn_all.csv",
        knn_rows,
        ["model", "query", "rank", "neighbor", "cosine_similarity"],
    )
    write_csv(
        out_dir / f"{model_name}_knn_summary.csv",
        summary_rows,
        ["model", "query", "covered", "average_similarity"],
    )
    write_csv(
        out_dir / f"{model_name}_knn_detail_words.csv",
        detail_rows,
        ["model", "query", "rank", "neighbor", "cosine_similarity"],
    )

    return {
        "model": model_name,
        "covered_queries": covered,
        "total_queries": len(WORD_LIST),
        "overall_average_similarity": overall_average,
        "detail_words": detail_words,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KNN evaluation for CBOW and Skip-Gram embeddings.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[str(project_path("embeddings", "cbow.vec")), str(project_path("embeddings", "skipgram.vec"))],
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--student-id", default=os.environ.get("STUDENT_ID", "0"))
    parser.add_argument("--detail-count", type=int, default=4)
    parser.add_argument("--out-dir", default=str(project_path("results")))
    args = parser.parse_args()

    seed = resolve_seed(args.student_id)
    detail_words = deterministic_sample(WORD_LIST, args.detail_count, seed)
    print(f"Student seed: {seed}")
    print(f"Detailed KNN words: {', '.join(detail_words)}")

    summaries = []
    for model in args.models:
        model_path = Path(model)
        if not model_path.exists():
            print(f"[SKIP] Missing embedding file: {model_path}")
            continue
        summary = evaluate_model(model_path, args.k, detail_words, Path(args.out_dir))
        summaries.append(summary)
        print(
            f"{summary['model']}: covered {summary['covered_queries']}/{summary['total_queries']} "
            f"queries, overall average similarity = {summary['overall_average_similarity']:.6f}"
        )

    if not summaries:
        raise SystemExit("No embedding files were evaluated. Put cbow.vec and skipgram.vec in embeddings/ first.")


if __name__ == "__main__":
    main()
