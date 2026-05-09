from __future__ import annotations

import argparse
import os
from pathlib import Path

from embedding_utils import (
    cosine,
    deterministic_sample,
    load_embeddings,
    model_name_from_path,
    project_path,
    resolve_seed,
    spearman_correlation,
    write_csv,
)


def read_simlex(path: Path) -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3 or parts[0].lower() == "word1":
                continue
            pairs.append((parts[0].lower(), parts[1].lower(), float(parts[2])))
    return pairs


def evaluate_model(model_path: Path, simlex_path: Path, sample_count: int, seed: int, out_dir: Path) -> dict:
    model_name = model_name_from_path(model_path)
    embeddings = load_embeddings(model_path)
    pairs = read_simlex(simlex_path)

    all_rows: list[dict] = []
    standard_scores: list[float] = []
    scaled_scores: list[float] = []

    for word1, word2, standard in pairs:
        covered = embeddings.has(word1) and embeddings.has(word2)
        row = {
            "model": model_name,
            "word1": word1,
            "word2": word2,
            "standard_similarity": f"{standard:.6f}",
            "covered": int(covered),
            "cosine_similarity": "",
            "scaled_similarity": "",
        }
        if covered:
            raw_cosine = cosine(embeddings.vector(word1), embeddings.vector(word2))
            scaled = (raw_cosine + 1.0) * 5.0
            row["cosine_similarity"] = f"{raw_cosine:.6f}"
            row["scaled_similarity"] = f"{scaled:.6f}"
            standard_scores.append(standard)
            scaled_scores.append(scaled)
        all_rows.append(row)

    covered_rows = [row for row in all_rows if row["covered"]]
    sample_rows = deterministic_sample(covered_rows, sample_count, seed)
    spearman = spearman_correlation(standard_scores, scaled_scores)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / f"{model_name}_simlex_all.csv",
        all_rows,
        ["model", "word1", "word2", "standard_similarity", "covered", "cosine_similarity", "scaled_similarity"],
    )
    write_csv(
        out_dir / f"{model_name}_simlex_sample_{sample_count}.csv",
        sample_rows,
        ["model", "word1", "word2", "standard_similarity", "covered", "cosine_similarity", "scaled_similarity"],
    )

    return {
        "model": model_name,
        "covered_pairs": len(covered_rows),
        "total_pairs": len(pairs),
        "spearman": spearman,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SimLex-999 golden standard evaluation.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[str(project_path("embeddings", "cbow.vec")), str(project_path("embeddings", "skipgram.vec"))],
    )
    parser.add_argument("--simlex", default=str(project_path("data", "simlex-999.txt")))
    parser.add_argument("--student-id", default=os.environ.get("STUDENT_ID", "0"))
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--out-dir", default=str(project_path("results")))
    args = parser.parse_args()

    seed = resolve_seed(args.student_id)
    summaries = []
    for model in args.models:
        model_path = Path(model)
        if not model_path.exists():
            print(f"[SKIP] Missing embedding file: {model_path}")
            continue
        summary = evaluate_model(model_path, Path(args.simlex), args.sample_count, seed, Path(args.out_dir))
        summaries.append(summary)
        print(
            f"{summary['model']}: covered {summary['covered_pairs']}/{summary['total_pairs']} pairs, "
            f"Spearman = {summary['spearman']:.6f}"
        )

    if not summaries:
        raise SystemExit("No embedding files were evaluated. Put cbow.vec and skipgram.vec in embeddings/ first.")


if __name__ == "__main__":
    main()
